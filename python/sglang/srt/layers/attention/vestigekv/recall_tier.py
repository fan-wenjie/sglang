"""GPU-resident recall tier for VestigeKV (PREREG19/20).

Originally vendored from the validated mini-sglang stack
(minisgl/kimi/tier2.py); deliberately changed since (conformal closed-form
zp, storage-dtype scoring, eigh basis, live-archive caches -- see the
vestigekv-dev history for each why). The equivalence net is self-contained:
test/manual/test_vestigekv_equiv.py asserts this module equal to an in-test
naive reference encoding the same math; change the math only in lockstep
with that reference, never one-sided.

Index per MLA slot, built once at a compression event: exact 64-dim sidecar
summand over archived rows, rank-r sketch of the 512-dim content (PCA basis of
prefix queries), residual-norm certificate self-calibrated (z) on prefix queries
with full-cache labels, entropy gate threshold from the same calibration
(auto-off when it cannot separate, PREREG24). Per decode query:
score = sidecar + sketch + z*certificate; fire where score beats the tier-1 max
and the gate is open; fetch top-j fired archived rows.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.defaults import ieee_fp32
from sglang.srt.layers.attention.vestigekv.scan_kernel import vestige_scan

# Every logging call in this module ran as a NameError until now, and the build
# worker catches Exception and keeps the provisional index -- so the three
# branches that log (the tier-2 skip, the archive-bound regime, and the rank
# report) each turned into "calibration silently did not happen" instead of
# into a message. Two ablation rounds came back as clean baselines that way.
logger = logging.getLogger(__name__)

_ZP_SEEN = [0]


def _log_zp(method, target, zp, n, zmax, rate=float('nan')):
    """One reading per process of what the certificate actually fitted.

    Three arms shipped this month as silent no-ops, and an arm that changes
    nothing reads exactly like an arm that ran and changed nothing. zp is the
    one number that says which fit ran and what it produced, and the sample
    maximum beside it is the ceiling the order statistic cannot pass.
    """
    _ZP_SEEN[0] += 1
    if _ZP_SEEN[0] % 200 == 1:
        logging.getLogger(__name__).info(
            "VKZP fit=%s target=%.4f zp=%.4f n_cal=%d sample_max=%.4f "
            "hard_rate=%.3f (build %d)",
            method,
            target,
            zp,
            n,
            zmax,
            rate,
            _ZP_SEEN[0],
        )



def _cal_dtype():
    """Storage dtype for the calibration's exact-score labels, or None for fp32.

    The pool is bf16, so fp32 here buys nothing on the archive operand and
    costs an upcast copy of every chunk; measured at [2048,576]x[576,16384],
    909.6 us fp32 against 237.3 bf16 and 256.2 fp16. What separates the two low
    forms is the QUERY: a bf16 value is exactly representable in fp16, so
    converting the pool loses nothing and only fp16 keeps the query's mantissa
    near fp32's -- 1 label changes against fp32, where bf16 changes 28 of 2048.
    """
    name = envs.SGLANG_DEBUG_VESTIGEKV_CAL_DTYPE.get()
    if not name:
        return None
    try:
        return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]
    except KeyError:
        raise SystemExit(
            f"ABORT: SGLANG_DEBUG_VESTIGEKV_CAL_DTYPE={name!r} is not one of "
            "'', 'bf16', 'fp16'; a typo here would silently keep fp32 and the "
            "ablation would report the baseline twice")


class RecallTier:
    def __init__(
        self,
        r: int = D.INDEX_RANK,
        recall_target: float = D.RECALL_TARGET,
        scale: float = D.ATTN_SCALE,
        margin: float = 0.0,
        threshold: str = "max",
        ent_gain: float = 0.0,
        fence_rows: int = 0,
        gauss_target: float = 0.0,
    ):
        self.r = r
        self.recall_target = recall_target
        self.scale = scale
        self.margin = margin  # scan threshold = base - margin (see VestigeKVConfig)
        self.threshold = threshold  # base: "max" kept score or "lse" of kept scores
        # Extra margin per nat of kept-distribution flatness. The fused
        # prologue applies it inside the kernel; these reference forms have to
        # apply the same one or the equivalence tests compare two thresholds.
        self.ent_gain = ent_gain
        # Rows fired above which the lane is fenced to its full row set; 0 =
        # only the buffer overflowing fences. See pre-registration 7.
        self.fence_rows = fence_rows
        # >0 fits zp as mu + Phi^-1(target) * sigma instead of taking the
        # conformal order statistic. See build().
        self.gauss_target = gauss_target
        self.built = False
        # live-archive projection caches: None until the first decode-time
        # close backfills them (extend_closed); _pos_all doubles as the fill
        # watermark read by the close path.
        self._pos_all = self._csk_all = self._rho_all = None
        self._side_mat = None  # lazily materialised; see the `side` property
        self._kept_mat = None  # lazily materialised; see `kept_rows`
        self._csk_mat = self._rho_mat = None  # selections over the _all caches
        self._arch_idx = None  # positions of the archive in the closed prefix
        # Index tables. These are what the tier STORES; the row tables
        # (side, kept_rows) and the archive selections (csk, rho) are
        # properties over them.
        self.arch = None  # pool row ids of the archive
        self.kept_slots = None  # pool row ids tier-1 keeps
        self._kbuf = None  # the layer's pool buffer, to re-read from
        self.version = 0  # bumped on in-place membership refresh (pack sync key)
        self.hard_rate = float("nan")   # set at each calibrated build
        self.archive_bound = False      # regime detector's verdict
        self._scatter_buf = None  # reused static-shape scatter target (query_fixed)
        # fixed-address staging for the fused scan (capturable)
        self._qside_t = self._qsk_t = self._hit_buf = self._inf = None

    def _from_all(self, name, cache):
        """Select the archive's rows out of a closed-prefix cache."""
        src = getattr(self, name)
        if src is None or self._arch_idx is None:
            raise RuntimeError(
                f"{name} is not populated; build()/extend_closed() must fill "
                "the closed-prefix caches before the archive can be selected"
            )
        return src.index_select(0, self._arch_idx.to(torch.int64)).contiguous()

    @property
    def csk(self):
        """The archive's sketch projections, [A, r] fp16 -- a selection over
        _csk_all, materialised on demand. Tier-1 re-decides membership at
        every block close, so the closed-prefix cache is the store and the
        archive is a view of it; keeping a compacted copy alongside is the
        same numbers twice."""
        if self._csk_mat is None:
            self._csk_mat = self._from_all("_csk_all", None)
        return self._csk_mat

    @csk.setter
    def csk(self, v):
        self._csk_mat = v

    @property
    def rho(self):
        """The archive's residual norms, [A] fp32 -- a selection over
        _rho_all, on the same terms as `csk`."""
        if self._rho_mat is None:
            self._rho_mat = self._from_all("_rho_all", None)
        return self._rho_mat

    @rho.setter
    def rho(self, v):
        self._rho_mat = v

    def _operand_builder(self):
        """The kernel that fills the closed-prefix caches. branch_tier swaps it
        for one with no basis, which is half the content traffic and no dot."""
        from sglang.srt.layers.attention.vestigekv.operand_fused import (
            build_operands_fused,
        )

        return build_operands_fused

    def drop_operands(self):
        """Release the materialised archive selections; the properties
        re-derive them from the closed-prefix caches."""
        self._csk_mat = self._rho_mat = None

    @property
    def kept_rows(self):
        """Tier-1's kept rows, [nk, 576] bf16 -- lazily materialised.

        Same story as `side`: kept_slots names them and the pool holds them, so
        a resident bf16 copy is 1152 bytes per row of nothing new. The in-graph
        prologue scores them straight out of the pool; the eager reference path
        and the max1 competition in query_fixed are what still want the rows.
        """
        if self._kept_mat is not None:
            return self._kept_mat
        if self._kbuf is None or self.kept_slots is None:
            raise RuntimeError(
                "tier has neither materialised kept rows nor a pool to read "
                "them from; build()/refresh_membership() must record kbuf"
            )
        self._kept_mat = self._kbuf[self.kept_slots.to(torch.int64)].to(torch.bfloat16)
        return self._kept_mat

    @kept_rows.setter
    def kept_rows(self, v):
        self._kept_mat = v

    def drop_kept_rows(self):
        """Release the materialised kept rows; the property re-derives them."""
        self._kept_mat = None

    @property
    def side(self):
        """The archive's sidecars, [A, 64] bf16.

        Materialised on demand rather than stored. The sidecar is the pool
        row's own tail at KV_LORA_RANK and `arch` already names the row, so a
        resident copy is bytes the pool still holds -- and it grows with the
        context. The in-graph scan reads the pool directly and never asks for
        this; only the eager reference path and calibration do, and both are
        off the per-step path.
        """
        if self._side_mat is not None:
            return self._side_mat
        if self._kbuf is None or self.arch is None:
            raise RuntimeError(
                "tier has neither a materialised sidecar nor a pool to read it "
                "from; build()/refresh_membership() must record kbuf"
            )
        self._side_mat = self._kbuf[self.arch][:, D.KV_LORA_RANK :].contiguous()
        return self._side_mat

    @side.setter
    def side(self, v):
        self._side_mat = v

    def drop_side(self):
        """Release the materialised sidecar; the property re-derives it."""
        self._side_mat = None

    @ieee_fp32
    @torch.inference_mode()
    def build(
        self,
        kbuf: torch.Tensor,
        row_slots: torch.Tensor,
        keep: torch.Tensor,
        q_cal: torch.Tensor,
        q_pos: torch.Tensor,
        conservative: bool = False,
        v_init: torch.Tensor | None = None,
        operands_from: RecallTier | None = None,
        diag: bool = False,
    ) -> dict:
        """rows: [T,576] pool rows (fp32). keep: [T] bool tier-1 mask.
        q_cal: [n,H,576] expanded calibration queries; q_pos: [n] positions.
        Returns stats. All thresholds derive from the prefix itself.

        operands_from: a previous tier for the SAME (slot, layer) whose
        archive row-set is unchanged (tier-1 selection is frozen between
        block closes, so every calibrated rebuild inside the calibration
        window sees the identical archive). The scan operands (V, csk, rho,
        side, arch) are pure functions of (prefix rows, basis, keep mask)
        and are INDEPENDENT of the calibration queries, so reusing them is
        bit-exact -- the rebuild then only refits the calibration scalars
        (zp, thr_g) and regathers the (small, growing) kept rows. This is
        what turns the per-request calibration ladder from 5-9 full archive
        passes into one: measured 10 ms -> ~1.5 ms per rebuild at S=8k.
        The caller owns the row-set-unchanged guard."""
        sc_ = self.scale
        T = row_slots.numel()
        dev = row_slots.device
        H = q_cal.shape[1]
        self.keep = keep
        arch_idx = (~keep).nonzero().flatten()
        self._arch_idx = arch_idx
        qe = q_cal.reshape(-1, D.LATENT_DIM).float()  # [n*H, 576]

        qcal_c = qe[:, : D.KV_LORA_RANK] - qe[:, : D.KV_LORA_RANK].mean(0, keepdim=True)
        # Top-r right singular vectors via the covariance eigendecomposition:
        # V(SVD) == eigenvectors of C^T C, and only r=64 of 512 are used, so the
        # full Jacobi SVD computes 448 vectors that are thrown away. Measured on
        # the real shape (512x512): 17.4 ms -> 4.3 ms, subspace alignment 1.0000
        # (torch.svd_lowrank was faster still but only 0.87-aligned: rejected).
        if v_init is not None:
            V = v_init
        elif conservative:
            # The PROVISIONAL build must never block the first decode step on a
            # cusolver eigendecomposition (measured ~4 ms/layer, ~20 ms/request
            # on the token path, and a one-shot ~230 ms cusolver init on the
            # very first call). Its basis is thrown away n_cal steps later by
            # the async calibrated build, and this index over-fetches anyway
            # (zp clamped, gate open), so basis QUALITY is irrelevant here --
            # only orthonormality matters for the Cauchy-Schwarz certificate.
            # Use the leading r rows of the identity: orthonormal by
            # construction, allocation-only, no factorization. The calibrated
            # build (async, off the token path) still fits the real PCA basis
            # and caches it, so every subsequent provisional build reuses that.
            V = torch.zeros(self.r, D.KV_LORA_RANK, device=dev, dtype=qe.dtype)
            V[torch.arange(self.r, device=dev), torch.arange(self.r, device=dev)] = 1.0
        else:
            # Fitted on THIS request's calibration queries, every calibrated
            # build. The certificate is sound for any orthonormal basis, but
            # its tightness is not: a basis carried over from another request
            # left another model family's queries with 2x the residual norm of
            # their own PCA and
            # the certificate fired the whole archive (fallback 0.6-0.8).
            evals, evecs = torch.linalg.eigh(qcal_c.T @ qcal_c)
            V = evecs[:, -self.r :].T.flip(0)  # descending singular value order
        if envs.SGLANG_DEBUG_VESTIGEKV_BRANCH_ONLY.get():
            V = torch.zeros_like(V)   # branch-only recall; see the env doc
        self.V = V
        # Chunked, and the pool rows are never materialized in fp32 as a whole.
        # The unchunked form allocated the full [T, 576] fp32 copy plus TWO
        # [A, 512] content copies (csk and rho each indexed it), ~429 MB per
        # build at S=64k. Ten builds per request of that, against a serving
        # mem-fraction, is why an in-server build cost 3.8x its isolated time.
        A = int(arch_idx.numel())
        if operands_from is not None:
            # Bit-exact adoption (see the docstring): same prefix rows, same
            # basis, same keep mask => same operands, no recompute. The guard
            # is the caller's; this assert catches a broken one.
            # Two conditions, not one. The archive must have the same size,
            # AND the donor's closed-prefix cache must cover THIS tier's
            # prefix -- _arch_idx indexes into that cache, so a donor built on
            # a shorter prefix makes every selection an out-of-bounds gather.
            # The old code adopted the already-compacted selection, where the
            # size check alone was sufficient; adopting the cache is what makes
            # the second condition load-bearing.
            donor_pos = operands_from._pos_all
            assert int(operands_from._arch_idx.numel()) == A, (
                "operand reuse across a changed archive row-set"
            )
            assert donor_pos is not None and donor_pos.numel() == int(
                row_slots.numel()
            ), (
                "operand reuse from a tier whose closed prefix differs: donor "
                f"{None if donor_pos is None else donor_pos.numel()} rows vs "
                f"{int(row_slots.numel())} here"
            )
            assert torch.equal(donor_pos, row_slots), (
                "operand reuse from a tier holding a different prefix row-set"
            )
            self.V = V = operands_from.V
            if envs.SGLANG_DEBUG_VESTIGEKV_BRANCH_ONLY.get():
                # the inherited basis needs the same treatment, or a request
                # that reuses one silently keeps the sketch term
                self.V = V = torch.zeros_like(V)
            # Adopt the closed-prefix CACHES, not the archive selections over
            # them: the selections are properties now, and a tier that owns
            # only a selection cannot re-derive one after a drop or a
            # membership refresh. The precondition (same rows, same basis)
            # makes the caches identical, which is what makes this bit-exact.
            self._csk_all = operands_from._csk_all
            self._rho_all = operands_from._rho_all
            assert self._csk_all.shape[0] == int(row_slots.numel()), (
                "adopted cache does not cover the prefix it will be indexed by"
            )
        else:
            # Project the WHOLE closed prefix, not just the archive. Tier-1's
            # membership is re-decided at every block close, so a row that is
            # kept now can be archived later; projecting only today's archive
            # means the close has to re-project, and holding both a
            # [closed, r] cache and a [A, r] compacted copy of it means
            # storing the same numbers twice. One store, and the archive is a
            # selection over it. The extra rows are the kept fraction, ~3%.
            #
            # The caches are filled TOGETHER with _pos_all here. An earlier
            # attempt set _pos_all eagerly while side/csk stayed lazy, and the
            # close path -- which reads _pos_all.shape[0] as its backfill
            # watermark -- then indexed a cache that did not cover it.
            if kbuf.is_cuda and kbuf.dtype == torch.bfloat16:
                # Fused single-kernel operand build: gather + project +
                # residual + casts (operand_fused.py). Same fp32-ieee
                # arithmetic; ~1 ulp reduction-order difference vs cuBLAS,
                # gated by fire-set stability + retrieval. Per row, so the
                # values on the archived subset do not depend on the row set.
                self._csk_all, self._rho_all, _side = self._operand_builder()(
                    kbuf, row_slots, V
                )
                del _side
            else:
                # Storage precision (gated): side at bf16 is BIT-EXACT relative to the
                # bf16 pool it is copied from (the old fp32 store was an uninformative
                # upcast); csk keeps fp16 -- it is an fp32 GEMM product and the fire
                # decision compares scores near a threshold, so the 11-bit mantissa
                # (vs bf16's 8) matters, while its magnitude sits far below the fp16
                # range cap, asserted below; rho stays fp32 (4 B/row, why touch it).
                # Scan traffic drops 516 -> 260 B/row: slope ratio 0.475 -> 0.256.
                # Accumulation everywhere stays fp32/ieee (the tf32 lesson).
                T = int(row_slots.numel())
                self._csk_all = torch.empty(T, self.r, device=dev, dtype=torch.float16)
                self._rho_all = torch.empty(T, device=dev, dtype=torch.float32)
                for a0 in range(0, T, D.BUILD_ROW_CHUNK):
                    a1 = min(a0 + D.BUILD_ROW_CHUNK, T)
                    blk = kbuf[row_slots[a0:a1]].float()
                    content = blk[:, : D.KV_LORA_RANK]
                    c = content @ V.T
                    torch._assert_async(
                        (c.abs().amax() < 6e4).to(torch.bool)
                    )  # fp16 range guard: a violation here is a model-scale anomaly
                    self._csk_all[a0:a1] = c.half()
                    self._rho_all[a0:a1] = (content - c @ V).norm(dim=-1)
                    del blk, content, c
        # Index tables in place, and the closed-prefix caches now cover the
        # built prefix, so _pos_all is an honest backfill watermark for the
        # close path. Everything below (calibration included) reads the row
        # tables and the archive selections through these.
        self._kbuf = kbuf
        self._pos_all = row_slots
        # int32 throughout: a pool row id indexes a pool with far fewer
        # than 2^31 slots, and these two are 15.5 B/token at int64 --
        # 9% of the whole index.
        self._arch_idx = arch_idx.to(torch.int32)
        self.arch = row_slots.index_select(0, arch_idx).to(torch.int32)
        # Pool row ids for the kept set, and nothing else: a latent row is
        # written once when its token enters the pool and never rewritten, so
        # a consumer that can address the pool wants these 4 bytes, not the
        # 1152-byte copy. `kept_rows` is a property over these.
        self.kept_slots = row_slots[keep].to(torch.int32)
        self._kept_mat = self._side_mat = None
        self._csk_mat = self._rho_mat = None

        if conservative:
            # Provisional index: serve immediately, calibrate nothing. Fitting
            # zp on cache-row proxies looked like calibration and was not --
            # measured on the port it returned zp anywhere from 0.0 to 8.0 for
            # the same layer across consecutive requests, and zp=0 means NO
            # certificate inflation, i.e. it can fire too little and miss the
            # target. The honest provisional setting is the most conservative
            # rung with the gate open: it over-fetches, which costs latency and
            # never recall. The real index replaces it n_cal decode steps later.
            self.thr_g = float("-inf")
            self.zp = D.Z_MAX
            self.need_more_hard = True
            self.built = True
            return {
                "zp": self.zp,
                "gate_off": True,
                "n_hard": None,
                "arch": int(arch_idx.numel()),
                "conservative": True,
            }
        # calibration: full-cache causal labels (pool is complete by invariant).
        # CHUNKED over keys to avoid a [n*H, T] materialization (OOMs at 512k).
        qpos_r = q_pos.repeat_interleave(H)
        nq = qe.shape[0]
        best_val = torch.full((nq,), torch.finfo(torch.float32).min, device=dev)
        tgt = torch.zeros(nq, dtype=torch.long, device=dev)
        # Best ARCHIVED row per query (true score, causal): the certificate is
        # calibrated on it for every query, not only for the queries whose
        # global argmax is archived -- see the zp block below.
        abest_val = torch.full((nq,), torch.finfo(torch.float32).min, device=dev)
        atgt = torch.zeros(nq, dtype=torch.long, device=dev)
        KB = D.BUILD_KEY_CHUNK
        cal_dt = _cal_dtype()
        q_cal_lo = None if cal_dt is None else qe.to(cal_dt)
        if cal_dt is torch.float16:
            # fp16 tops out at 65504 and the pool is bf16, whose range is
            # fp32's; the same guard csk carries, for the same reason.
            torch._assert_async((qe.abs().amax() < 6e4).to(torch.bool))
        if diag:
            # Debug telemetry: how many ARCHIVED rows each calibration query
            # truly prefers to its best kept row (the recall need), against
            # what the certificate fires below.
            max1_d = (
                (qe.to(torch.bfloat16) @ self.kept_rows.T)
                .float()
                .mul_(sc_)
                .max(-1)
                .values
            )
            need = torch.zeros(nq, dtype=torch.int64, device=dev)
        for k0 in range(0, T, KB):
            k1 = min(k0 + KB, T)
            # In place throughout: `* sc_` and `masked_fill` each copied the
            # whole [n*H, KB] block, 67 MB per chunk that the allocator then had
            # to find under a serving mem-fraction.
            if cal_dt is None:
                cblk = kbuf[row_slots[k0:k1]].float()
                blk = qe @ cblk.T
                del cblk
            else:
                kblk = kbuf[row_slots[k0:k1]]
                if kblk.dtype != cal_dt:
                    kblk = kblk.to(cal_dt)
                blk = (q_cal_lo @ kblk.T).float()
                del kblk
            blk.mul_(sc_)
            colk = torch.arange(k0, k1, device=dev)[None, :]
            blk.masked_fill_(colk > qpos_r[:, None], torch.finfo(torch.float32).min)
            if diag:
                arch_col = (~keep[k0:k1])[None, :]
                need += ((blk > max1_d[:, None]) & arch_col).sum(-1)
            bval, bidx = blk.max(-1)
            take = bval > best_val
            best_val = torch.where(take, bval, best_val)
            tgt = torch.where(take, bidx + k0, tgt)
            blk.masked_fill_(keep[k0:k1][None, :], torch.finfo(torch.float32).min)
            abval, abidx = blk.max(-1)
            atake = abval > abest_val
            abest_val = torch.where(atake, abval, abest_val)
            atgt = torch.where(atake, abidx + k0, atgt)
            del blk
        del best_val
        hard = ~keep[tgt]

        skept = (qe.to(torch.bfloat16) @ self.kept_rows.T).float() * sc_
        p1 = torch.softmax(skept, -1)
        ent = -(p1 * p1.clamp_min(D.ENTROPY_EPS).log()).sum(-1)
        max1 = self._thr_base(skept, skept.max(-1).values)
        del skept, p1

        n_hard = int(hard.sum())
        alpha = D.gate_alpha(self.recall_target)
        thr_g = (
            float(ent[hard].quantile(alpha))
            if n_hard >= D.min_hard(self.recall_target)
            else float("-inf")
        )
        if float((ent > thr_g).float().mean()) > D.GATE_SELF_DISABLE_FRACTION:
            thr_g = float("-inf")
        if envs.SGLANG_DEBUG_VESTIGEKV_NO_GATE.get():
            thr_g = float("-inf")     # ablation: gate never closes
        self.thr_g = thr_g

        # zp in closed form. Requirement per calibration query q: the certified
        # score of its best ARCHIVED row t must reach that row's true score,
        # idxs_t + z*cert_t >= true_t, i.e. z >= z_q := (true_t - idxs_t)/cert_t;
        # then whenever an archived row truly beats the kept set, its certified
        # score does too and the scan fires it. Over the n exchangeable queries
        # the k-th order statistic of z_q at k = ceil((n+1)*scan_target) is the
        # conformal quantile (distribution-free marginal guarantee). Every
        # query contributes, not only the "hard" ones whose global argmax is
        # archived: on a model whose kept set is good those are too few to
        # calibrate on and the certificate collapsed to the Z_MAX clamp,
        # firing the whole archive.
        # Positional, not pool ids: atgt indexes the closed prefix and
        # _arch_idx is the archive's positions in it, ascending.
        has_arch = abest_val > torch.finfo(torch.float32).min
        n_cal_q = int(has_arch.sum())
        # Regime detection, from evidence the calibration already gathered.
        # `hard` marks calibration queries whose best row is ARCHIVED. A needle
        # decode wants the kept tier most steps; a request reading a long
        # answer out of context wants the archive nearly every step, and its
        # answer is a conjunction over hundreds of them, which no per-step
        # recall target reaches while staying sparse. Detect it here, where the
        # rate is already computed, and let the request keep the clamp.
        self.hard_rate = n_hard / max(int(hard.numel()), 1)
        _rr = envs.SGLANG_DEBUG_VESTIGEKV_REGIME_RATE.get()
        self.archive_bound = bool(_rr) and self.hard_rate >= _rr
        _skip = envs.SGLANG_DEBUG_VESTIGEKV_SKIP_TIER2_HARD.get()
        if _skip >= 0.0 and self.hard_rate <= _skip:
            # Vacuous guarantee: the archive holds the winner on none of the
            # calibration queries, so the width this tier's gate pays for buys
            # nothing. +inf is what the scan already reads as "no row can
            # fire"; an empty kept set still forces the gate open, so a
            # request with nothing in tier 1 is not stranded.
            #
            # This sits AFTER hard_rate is assigned, which is the whole point:
            # the first version read it four statements too early, got the
            # float("nan") from __init__, and never fired -- nan <= 0.0 is
            # False. Three ablation jobs came back as clean baselines and only
            # the silence of the log line below said so.
            #
            # The standing risk is the one that retired hard_rate as a
            # detector: calibration queries are the prompt's last few, answer
            # steps are not them, 0.1% against 22.85% miss.
            self.thr_g = float("inf")
            logger.info(
                "VKSKIP tier2 off: hard_rate=%.4f <= %.4f (arch=%d)",
                self.hard_rate, _skip, int(arch_idx.numel()),
            )
        self.need_more_hard = n_cal_q < D.min_hard(self.recall_target)
        if self.need_more_hard:
            # Not enough evidence for the guarantee yet: serve with the safety
            # clamp (over-fetches, never under-recalls) and tell the caller to
            # keep collecting.
            self.zp = D.Z_MAX
        elif self.archive_bound:
            # Escalate for THIS request: the fitted quantile is sound per step
            # and irrelevant to a conjunction this long, so the clamp stays.
            self.zp = D.Z_MAX
            logger.info(
                "VKREGIME archive-bound: hard_rate=%.3f >= %.3f, keeping the "
                "Z_MAX clamp (n_hard=%d/%d, n_cal_q=%d)",
                self.hard_rate, _rr, n_hard, int(hard.numel()), n_cal_q,
            )
        else:
            qh = qe[has_arch]
            qskh = qh[:, : D.KV_LORA_RANK] @ V.T
            qresh = (qh[:, : D.KV_LORA_RANK] - qskh @ V).norm(dim=-1)
            pa = torch.searchsorted(self._arch_idx.to(torch.int64), atgt[has_arch])
            # Gather the target rows out of the caches directly. Going through
            # self.side / self.csk / self.rho would materialise the WHOLE
            # archive's selection to read a few dozen rows of it, and those
            # three views are the entire difference between the steady-state
            # footprint and the peak -- which is the figure that decides
            # whether a long request fits.
            pa64 = pa.to(torch.int64)
            arch_rows = self.arch.to(torch.int64).index_select(0, pa64)
            tgt_side = self._kbuf.index_select(0, arch_rows)[
                :, D.KV_LORA_RANK : D.LATENT_DIM
            ]  # [n, side_dim]
            src = self._arch_idx.to(torch.int64).index_select(0, pa64)
            tgt_csk = self._csk_all.index_select(0, src)  # [n, r]
            tgt_rho = self._rho_all.index_select(0, src)  # [n]
            # Same rounding as the serve-time scoring path: query operands
            # rounded to the storage dtypes before the (exact-in-fp32)
            # products, so zp is calibrated on exactly what the kernel scores.
            idxs_t = (
                qh[:, D.KV_LORA_RANK :].to(torch.bfloat16).float() * tgt_side.float()
            ).sum(-1) + (qskh.half().float() * tgt_csk.float()).sum(-1)
            idxs_t = idxs_t * sc_
            cert_t = (
                qresh * tgt_rho * sc_ / (D.KV_LORA_RANK - self.r) ** 0.5
            ).clamp_min(D.ENTROPY_EPS)
            z_req = (abest_val[has_arch] - idxs_t) / cert_t
            if self.gauss_target > 0.0:
                # A parametric fit instead of an order statistic. z_req is the
                # inner product of two residuals in the (kv - r)-dimensional
                # orthogonal complement, divided by sqrt(kv - r), so under
                # isotropy it is standard normal by construction -- and it
                # measures that way: skewness -0.18, excess kurtosis -0.00 over
                # the Kimi dumps.
                #
                # Three things the order statistic cannot do. It is the MAXIMUM
                # of the sample at the min_hard bar, where it varies by 1.35
                # across layers about a median of 1.88; it can never exceed the
                # sample max, which is the 0.99 Gaussian target, so a higher
                # target is inexpressible; and conformal's one advantage, a
                # distribution-free guarantee under exchangeability, is bought
                # at that price while exchangeability demonstrably fails --
                # calibration queries miss the top archived row on 0.1% of
                # cases and answer steps on 22.85%.
                mu, sd = z_req.mean(), z_req.std()
                q = torch.special.ndtri(
                    torch.tensor(
                        self.gauss_target, device=z_req.device, dtype=z_req.dtype
                    )
                )
                self.zp = min(float(mu + q * sd), D.Z_MAX)
                _log_zp(
                    "gauss", self.gauss_target, self.zp, n_cal_q,
                    float(z_req.max()), self.hard_rate,
                )
            else:
                k = D.conformal_k(n_cal_q, self.recall_target)
                self.zp = min(float(z_req.kthvalue(k).values), D.Z_MAX)
                _log_zp(
                    "conformal",
                    self.recall_target,
                    self.zp,
                    n_cal_q,
                    float(z_req.max()),
                    self.hard_rate,
                )
        zp = self.zp
        # Per-row projection caches over ALL closed rows (kept and archived
        # alike), so a decode-time block close -- which rebalances membership
        # globally -- refreshes the index by index selection only: no
        # re-projection GEMM, no stale archive, no unrecallable rows. Mirrors
        # the reference engine's live-archive semantics.
        # The caches now COVER the built prefix, so _pos_all is set to match
        # and the close path backfills only [built, c1). The earlier contract
        # (all three None, first close re-projects 0..c1) existed because an
        # eager _pos_all with lazy side/csk left the watermark ahead of the
        # cache contents; filling them together is what makes the watermark
        # honest. _arch_idx indexes the archive INTO that prefix; `arch`
        # carries the pool row ids the scan and the fetch output use.
        self.built = True
        stats = {
            "zp": zp,
            "gate_off": thr_g == float("-inf"),
            "n_hard": n_hard,
            "need_more_hard": self.need_more_hard,
            "arch": int(arch_idx.numel()),
        }
        if diag:
            stats.update(self._diag_fire(qe, max1, zp, sc_, need))
        return stats

    def _ent_margin(self, skept):
        """How flat the kept distribution is, in nats: log-sum-exp of the kept
        scores less their maximum. 0 when one kept row holds the attention
        mass, log(n) when none does. Scaled by ent_gain it widens the scan's
        threshold exactly where the maximum says least about the query; the
        kernel computes the same quantity as `lse - e_max`.
        """
        if self.ent_gain == 0.0 or skept is None:
            return 0.0
        return self.ent_gain * (torch.logsumexp(skept, -1) - skept.max(-1).values)

    def _adaptive_margin(self, qe, qsk, qres, max1, sc_, lse_kept=None):
        """gamma from the archive's own mass, per head. NOT ON THE SERVED PATH.

        This runs inside query_fixed, the per-(layer, slot) form. Decode does
        not use it: batched_step's fused prologue computes skept, the softmax,
        the entropy, the gate, qsk, qres and max1g in one kernel, and max1g is
        the threshold. Setting the env flag therefore changed nothing, and the
        leakage telemetry below never logged a line -- which is how this was
        found, after the two adaptive runs came back at 0.343 and 0.390 where
        an equivalent fixed margin reads 0.865.

        The architecture is the finding, not the bug. max1g is computed BEFORE
        the archive is scanned, so a threshold cannot depend on the archive's
        score distribution without a second pass over it, and the scan is the
        cost. That is why the entropy margin keys on the KEPT distribution:
        at prologue time it is the only distribution that exists.

        What fits: have the scan accumulate lse over the certified scores it
        already visits and discards -- one register reduction, no extra
        traffic -- and let the NEXT step's prologue use it. Recall is stale by
        one step by construction, so this adds no staleness that is not there.
        """
        eps = envs.SGLANG_DEBUG_VESTIGEKV_ADAPTIVE_MARGIN.get()
        if not eps:
            return 0.0
        import math

        idxs = (qsk.half().float() @ self.csk.float().T) * sc_
        if D.SIDECAR_DIM:
            # The branch term is half the certified score; without it the
            # log-sum-exp below is of a different quantity than the scan
            # thresholds on, and the margin would be derived from a score the
            # kernel never computes.
            idxs = (
                idxs
                + (
                    qe[:, D.KV_LORA_RANK :].to(torch.bfloat16).float()
                    @ self.side.float().T
                )
                * sc_
            )
        cert = (qres[:, None] * self.rho[None, :]) * sc_ / (
            D.KV_LORA_RANK - self.r
        ) ** 0.5
        s_arch = idxs + self.zp * cert
        lse = torch.logsumexp(s_arch, dim=-1)
        n_arch = s_arch.shape[-1]
        s_keep = s_arch if envs.SGLANG_DEBUG_VESTIGEKV_LEAKAGE.get() else None
        del idxs, cert
        if s_keep is None:
            del s_arch
        # Two branches, because only two are provable. The excluded rows all
        # sit below max1 - gamma, so their mass is at most
        # n_arch * exp(max1 - gamma); holding that under eps * exp(max1) needs
        # gamma >= ln(n_arch / eps). The log-sum-exp buys exactly one thing:
        # when the WHOLE archive holds less than eps of the kept maximum's
        # weight, excluding all of it is already within budget and gamma is 0.
        #
        # A smooth interpolation between the two -- gamma = lse - max1 +
        # ln(1/eps) -- was written here first and is WRONG. Its fixed point
        # against an archive whose rows cluster just under the cut is
        # gamma = ln(n_arch/eps)/2, half of what the bound needs, and the
        # excluded mass then exceeds eps by more than two orders of magnitude
        # at n_arch = 16k. That clustering is not a corner case here: it is
        # what a verbatim copy looks like, which is the regime the margin was
        # being derived for.
        if envs.SGLANG_DEBUG_VESTIGEKV_ADAPTIVE_MODE.get() == "smooth":
            g = (lse - max1 + math.log(1.0 / eps)).clamp_min(0.0)
        else:
            g = torch.where(
                lse - max1 <= math.log(eps),
                torch.zeros_like(lse),
                torch.full_like(lse, math.log(n_arch / eps)),
            )
        if envs.SGLANG_DEBUG_VESTIGEKV_LEAKAGE.get():
            # What the rule actually leaves outside, relative to exp(max1):
            # the algebra says the smooth rule can exceed eps here, and this is
            # where that either shows up on real traffic or does not.
            cut = (max1 - self.margin)[:, None] - g[:, None]
            left = torch.where(s_keep <= cut, s_keep, s_keep.new_full((), -1e30))
            # Normalise by the INCLUDED set's log-sum-exp, not by max1. The
            # online-softmax merge identity says the output is
            #   sigma(L_F - L_E) o_F + sigma(L_E - L_F) o_E,
            # so the weight the omitted rows actually carry in the answer is
            # sigma(L_E - L_F). max1 is a single row and overstates that
            # weight whenever the kept set is diffuse, which is most steps.
            ref = max1 if lse_kept is None else lse_kept
            lse_excl = torch.logsumexp(left, -1)
            leak = torch.sigmoid(lse_excl - ref)
            self._leak_hist = getattr(self, "_leak_hist", [])
            self._leak_hist.append(float(leak.median()))
            if len(self._leak_hist) % 256 == 0:
                import statistics as _st
                logger.info(
                    "VKLEAK mode=%s eps=%.3f scans=%d leak_median=%.4g leak_p90=%.4g "
                    "gamma_median=%.2f",
                    envs.SGLANG_DEBUG_VESTIGEKV_ADAPTIVE_MODE.get(), eps,
                    len(self._leak_hist), _st.median(self._leak_hist),
                    sorted(self._leak_hist)[int(0.9 * len(self._leak_hist))],
                    float(g.median()),
                )
        return g

    def _thr_base(self, skept, max1):
        # The score the margin is taken from: the best kept row, or the kept
        # set's log-sum-exp (>= max1; equal when one row holds the mass).
        if self.threshold == "lse":
            return torch.logsumexp(skept, -1)
        return max1

    def _diag_fire(self, qe, max1, zp, sc_, need):
        """Debug telemetry for one calibrated build: per calibration query, the
        rows the certificate fires at the fitted zp against the rows the query
        truly needs (true score above its best kept row); medians and maxima
        over the queries, plus the certificate's own two terms."""
        A = self.arch.shape[0]
        qsk = qe[:, : D.KV_LORA_RANK] @ self.V.T
        qres = (qe[:, : D.KV_LORA_RANK] - qsk @ self.V).norm(dim=-1)
        idxs = (qsk.half().float() @ self.csk.float().T) * sc_
        if D.SIDECAR_DIM:
            idxs = (
                idxs
                + (
                    qe[:, D.KV_LORA_RANK :].to(torch.bfloat16).float()
                    @ self.side.float().T
                )
                * sc_
            )
        cert = (
            (qres[:, None] * self.rho[None, :]) * sc_ / (D.KV_LORA_RANK - self.r) ** 0.5
        )
        fire = ((idxs + zp * cert) > (max1 - self.margin)[:, None]).sum(-1)
        fire0 = (idxs > (max1 - self.margin)[:, None]).sum(-1)  # sketch term alone
        q = lambda t, p: float(t.float().quantile(p))
        # Would a position-pooled (4 rows) certificate be usable? Group bound:
        # q_sk . mean(c) + |q_sk| max|c_i - mean(c)| + zp cert(max rho); a
        # fired group fetches its 4 rows.
        A4 = (A // 4) * 4
        cg = self.csk[:A4].float().view(-1, 4, self.r)
        cbar = cg.mean(1)
        delta = (cg - cbar[:, None, :]).norm(dim=-1).max(1).values
        rho_g = self.rho[:A4].view(-1, 4).max(1).values
        bound_g = (qsk @ cbar.T) * sc_ + (
            qsk.norm(dim=-1)[:, None] * delta[None, :]
        ) * sc_
        bound_g = (
            bound_g
            + (qres[:, None] * rho_g[None, :])
            * sc_
            * zp
            / (D.KV_LORA_RANK - self.r) ** 0.5
        )
        gfire = (bound_g > (max1 - self.margin)[:, None]).sum(-1) * 4
        return {
            "gfire_p50": q(gfire, 0.5),
            "gfire_p90": q(gfire, 0.9),
            "delta_over_c_p50": q(delta / cbar.norm(dim=-1).clamp_min(1e-6), 0.5),
            "A": A,
            "need_p50": q(need, 0.5),
            "need_p90": q(need, 0.9),
            "need_max": int(need.max()),
            "fire_p50": q(fire, 0.5),
            "fire_p90": q(fire, 0.9),
            "fire_sketch_only_p50": q(fire0, 0.5),
            "max1_p50": q(max1, 0.5),
            "cert_p50": q(cert.median(dim=-1).values * zp, 0.5),
            "rho_p50": q(self.rho, 0.5),
            "qres_p50": q(qres, 0.5),
        }

    @torch.inference_mode()
    @ieee_fp32
    def query_fixed(
        self,
        qe: torch.Tensor,
        out: torch.Tensor,
        out_len: torch.Tensor,
        out_ovf: torch.Tensor,
        slot: int,
    ) -> None:
        """Device-only variant of query(): writes the fired pool rows into
        out[slot, :W], their count (at most W) into out_len[slot] and whether
        the fire exceeded W into out_ovf[slot], with NO host sync (no
        .numel(), no bool() early-exit). Serving decode uses this; query()
        stays as the reference/equivalence-tested form. An overflow keeps the
        first W rows in position order; the flag is what the pack's fence
        reads."""
        sc_ = self.scale
        qe = qe.float()
        H = qe.shape[0]
        if self.kept_slots.shape[0] == 0:
            # No tier-1 row kept => there is no max1 baseline to beat, so every
            # archived row is eligible: attend the whole archive this step
            # (degenerates to full attention, never under-recalls). A kept set
            # this small is itself anomalous (see the close/build invariant);
            # the +inf-open form keeps serving correct while it is investigated.
            max1 = qe.new_full((H,), float("-inf"))
            gate = qe.new_ones(H, dtype=torch.bool)
            emarg = 0.0
            skept = None      # no kept scores exist on this branch
        else:
            skept = (qe.to(torch.bfloat16) @ self.kept_rows.T).float() * sc_
            max1 = self._thr_base(skept, skept.max(-1).values)
            emarg = self._ent_margin(skept)
            p1 = torch.softmax(skept, -1)
            ent = -(p1 * p1.clamp_min(D.ENTROPY_EPS).log()).sum(-1)
            gate = ent > self.thr_g
        qsk = qe[:, : D.KV_LORA_RANK] @ self.V.T
        qres = (qe[:, : D.KV_LORA_RANK] - qsk @ self.V).norm(dim=-1)
        if self._qside_t is None:
            H = qe.shape[0]
            # Storage dtypes, matching the kernel's native-dtype dots: bf16 for
            # the sidecar branch (exact -- qbuf is bf16), fp16 for the sketch
            # projection (rounded here AND at calibration, so zp covers it).
            self._qside_t = qe.new_empty(D.SIDECAR_DIM, H, dtype=torch.bfloat16)
            self._qsk_t = qe.new_empty(self.r, H, dtype=torch.float16)
            self._hit_buf = torch.empty(
                self.side.shape[0], dtype=torch.int32, device=qe.device
            )
            self._inf = qe.new_full((), float("inf"))
        self._qside_t.copy_(qe[:, D.KV_LORA_RANK :].T)
        self._qsk_t.copy_(qsk.T)
        # Fold the gate into the threshold: a closed head can never fire, so
        # +inf makes it lose every comparison and the kernel needs no second
        # predicate. Scores stay in registers -- see scan_kernel for why
        # that, not arithmetic, is what the scan costs.
        amarg = self._adaptive_margin(
            qe, qsk, qres, max1, sc_,
            lse_kept=None if skept is None else torch.logsumexp(skept, -1))
        max1g = torch.where(gate, max1 - self.margin - emarg - amarg, self._inf)
        hit = (
            vestige_scan(
                self._qside_t,
                self._qsk_t,
                qres,
                max1g,
                self.side,
                self.csk,
                self.rho,
                sc_,
                self.zp * sc_ / (D.KV_LORA_RANK - self.r) ** 0.5,
                out=self._hit_buf,
            )
            != 0
        )
        k_succ = envs.SGLANG_DEBUG_VESTIGEKV_SUCCESSOR.get()
        if k_succ:
            # Ablation: dilate the fire mask forward along the archive so a
            # stale query that points at row p also admits p+1..p+k.
            dil = hit.clone()
            for _j in range(1, int(k_succ) + 1):
                dil[_j:] |= hit[:-_j]
            hit = dil
        W = out.shape[1]
        total = hit.sum()  # [] int64 on device
        n = total.clamp(max=W)
        # Rank the hit rows by position and keep the first W. Everything here is
        # static-shape on purpose: boolean-mask indexing (`pos[sel]`) is a
        # masked_select, whose output size is only known on the device, so eager
        # mode reads it back and drains the pipeline. Two such reads per call x
        # one call per local MLA layer per request was 8.0 of the 10.5 ms/step
        # of host-side cost (VKSTATS, S=61k bs=1). Instead, scatter every archive
        # row -- misses and overflow all target a scratch slot W that is dropped.
        pos = torch.cumsum(hit.to(torch.int64), 0) - 1
        dst = torch.where(hit & (pos < W), pos, W)
        scratch = self._scatter_buf
        if scratch is None or scratch.shape[0] != W + 1:
            scratch = out.new_zeros(W + 1)
            self._scatter_buf = scratch
        scratch.zero_()
        scratch.scatter_(0, dst, self.arch.to(out.dtype))
        out[slot].copy_(scratch[:W])
        out_len[slot] = n
        # See compact_fired: the flag's threshold is separate from the buffer
        # cap, so a multi-key fence can raise it early. A fenced lane attends
        # its full row set and never reads the buffer, so this is safe.
        out_ovf[slot] = total > (min(W, self.fence_rows) if self.fence_rows else W)

    @torch.inference_mode()
    def step_attribution(self, qe: torch.Tensor) -> dict:
        """Where this step's attention mass actually went, against dense.

        Every offline study so far conditions on the kept set as given and
        asks only whether an archived row that beats max1 is fired. That
        question is answered -- held-out recall is 99.9% with no decay in k --
        and the multi-key accuracy gap survives it, so the next question is
        the one nothing has asked: of the mass DENSE puts somewhere, how much
        does VestigeKV attend, and is the row dense leans on even in the
        attended set?

        Computes the true scores over the whole closed prefix (kept plus
        archive), which is what makes this a debug-only path: it is the dense
        attention the method exists to avoid. Returns per-head medians, small
        enough to keep one record per (step, layer, lane).
        """
        sc_ = self.scale
        qe = qe.float()
        kv = D.KV_LORA_RANK
        if self.arch is None or self.arch.numel() == 0:
            return {}
        skept = (qe.to(torch.bfloat16) @ self.kept_rows.T).float() * sc_
        max1 = skept.max(-1).values
        rows = self.arch.to(torch.int64)
        arch_rows = self._kbuf.index_select(0, rows).float()
        strue = (qe @ arch_rows.T) * sc_
        qsk = qe[:, :kv] @ self.V.T
        qres = (qe[:, :kv] - qsk @ self.V).norm(dim=-1)
        idxs = (
            (qe[:, kv:].to(torch.bfloat16) @ self.side.T).float()
            + (qsk.half() @ self.csk.T).float()
        ) * sc_
        cert = (qres[:, None] * self.rho[None, :]) * sc_ / (kv - self.r) ** 0.5
        fired = ((idxs + self.zp * cert) > (max1 - self.margin)[:, None]).any(0)
        allsc = torch.cat([skept, strue], dim=-1)
        m = allsc.max(-1, keepdim=True).values
        w = (allsc - m).exp()
        Z = w.sum(-1)
        att = torch.cat(
            [torch.ones_like(skept, dtype=torch.bool), fired.expand_as(strue)], -1
        )
        pr = w / Z[:, None]
        ent = -(pr * pr.clamp_min(1e-30).log()).sum(-1)
        # WHY a missed row was missed: where the true argmax ranks under the
        # certified score the scan actually orders by, and how far its
        # certified score fell short of the kept maximum. A row ranked 3rd but
        # below threshold is a threshold problem; one ranked 40000th is an
        # ordering problem, and they have different fixes.
        top1 = allsc.argmax(-1)
        cert_sc = idxs + self.zp * cert
        a_top1 = (top1 - skept.shape[1]).clamp_min(0)
        arch_is_top = top1 >= skept.shape[1]
        rank = (cert_sc > cert_sc.gather(1, a_top1[:, None])).sum(1)
        marg = cert_sc.gather(1, a_top1[:, None]).squeeze(1) - max1
        sel = arch_is_top & (~fired.gather(0, a_top1.clamp_max(fired.numel() - 1)))
        # Leakage, both the observable and the truth, so the probe can ask
        # whether one tracks the other. The scan can know only the first: the
        # certified scores of the rows it did NOT fire, against the kept
        # log-sum-exp. By the online-softmax merge identity the weight those
        # rows carry in the answer is sigma(L_excluded - L_included), which is
        # why neither is normalised by max1 -- a single row overstates that
        # weight whenever the kept set is diffuse, which is most steps.
        cert_all = idxs + self.zp * cert
        neg = torch.finfo(torch.float32).min
        lse_kept_h = torch.logsumexp(skept, -1)
        out_cert = torch.where(fired[None, :], torch.full_like(cert_all, neg), cert_all)
        leak_cert = torch.sigmoid(torch.logsumexp(out_cert, -1) - lse_kept_h)
        # and what was actually left out, by the exact scores
        out_true = torch.where(fired[None, :], torch.full_like(strue, neg), strue)
        in_true = torch.cat(
            [skept, torch.where(fired[None, :], strue, torch.full_like(strue, neg))], -1
        )
        leak_true = torch.sigmoid(
            torch.logsumexp(out_true, -1) - torch.logsumexp(in_true, -1)
        )
        return {
            "leak_cert_p50": float(leak_cert.median()),
            "leak_cert_max": float(leak_cert.max()),
            "leak_true_p50": float(leak_true.median()),
            "leak_true_max": float(leak_true.max()),
            "top1_rank_p50": float(rank[sel].median()) if bool(sel.any()) else -1.0,
            "top1_margin_p50": float(marg[sel].median()) if bool(sel.any()) else 0.0,
            "qperp_rel": float((qres / qe[:, :kv].norm(dim=-1)).median()),
            "coverage": float(((w * att).sum(-1) / Z).median()),
            "coverage_min": float(((w * att).sum(-1) / Z).min()),
            "top1_attended": float(att.gather(1, top1[:, None]).float().mean()),
            "entropy": float(ent.median()),
            "n_beat_max1": int((strue > max1[:, None]).sum(1).max()),
            "n_fired": int(fired.sum()),
            "n_kept": int(self.kept_slots.shape[0]),
            "n_arch": int(rows.numel()),
            # The calibration's own regime reading, carried per step so it can
            # be crossed against n_beat_max1 PER LAYER. Constant across a
            # request's steps for a given layer, which is the point: the
            # question is whether the build's hard_rate predicts what the
            # answer steps of that same layer actually need. Two skip-tier2
            # ablations came back vacuous for want of this one number --
            # _log_zp throttles at every 200th build, so its hard_rate reading
            # is one sample per rank, never the distribution a threshold has to
            # be chosen against.
            "hard_rate": float(self.hard_rate),
            "archive_bound": bool(self.archive_bound),
        }

    @torch.inference_mode()
    def omitted_mass_and_mean(self, qe: torch.Tensor):
        """The softmax mass this step's scan does NOT attend, and its centroid.

        Returns (logM [H], mu [H, kv_lora_rank]). The scan attends kept +
        fired, and fired is the UNION over heads -- a row one head fires is
        read by all of them -- so a head's omitted set is the archive minus
        that union. Their true scores are unknown; what the scan has is the
        certified upper bound idxs + zp*cert, the same quantity it compares
        against the threshold, so the mass is overestimated rather than
        guessed.

        Why the centroid is mass-weighted and not the plain archive mean:
        blending the output toward a synthetic row is one exact step of the
        online-softmax recurrence

            O_n = lerp(O_{n-1}, v_n, sigmoid(s_n - lse_{n-1}))

        and that step is exact only when the synthetic row carries the
        omitted set's mass-weighted centroid. Measured offline on the Kimi
        dumps, the arithmetic mean leaves the attention-output error at 0.367
        against 0.195 for the weighted one (0.471 uncompensated).

        Chunked over the archive: the unchunked form materialises an [H, A]
        score matrix and an [A, 512] value selection, 123 MB of fp32 at 64k,
        against the peak footprint that decides whether a long request fits.
        Shifted by max1 per head, which is what the bound is compared against,
        so the exponentials cannot overflow.

        A research arm (SGLANG_DEBUG_VESTIGEKV_OMITTED_BLEND), not the serving
        path: it touches every archived row's value, which costs about six
        times the attention over the kept set. What it buys is the one term
        every other measurement here conditioned away.
        """
        sc_ = self.scale
        qe = qe.float()
        H = qe.shape[0]
        kv = D.KV_LORA_RANK
        ninf = qe.new_full((H,), float("-inf"))
        if self.arch is None or self.arch.numel() == 0:
            return ninf, qe.new_zeros((H, kv))
        if self.kept_slots.shape[0] == 0:
            max1 = ninf
            gate = qe.new_ones(H, dtype=torch.bool)
            emarg = 0.0
        else:
            skept = (qe.to(torch.bfloat16) @ self.kept_rows.T).float() * sc_
            max1 = self._thr_base(skept, skept.max(-1).values)
            emarg = self._ent_margin(skept)
            p1 = torch.softmax(skept, -1)
            ent = -(p1 * p1.clamp_min(D.ENTROPY_EPS).log()).sum(-1)
            gate = ent > self.thr_g
        if not torch.isfinite(max1).all():
            return ninf, qe.new_zeros((H, kv))
        thr = max1 - self.margin - emarg
        qsk = qe[:, :kv] @ self.V.T
        qres = (qe[:, :kv] - qsk @ self.V).norm(dim=-1)
        qside = qe[:, kv:].to(torch.bfloat16)
        qskh = qsk.half()
        denom = (kv - self.r) ** 0.5
        rows = self.arch.to(torch.int64)
        side_a, csk_a, rho_a = self.side, self.csk, self.rho
        acc_w = qe.new_zeros((H,))
        acc_v = qe.new_zeros((H, kv))
        for i in range(0, rows.numel(), 8192):
            sl = slice(i, i + 8192)
            idxs = ((qside @ side_a[sl].T).float() + (qskh @ csk_a[sl].T).float()) * sc_
            cert = (qres[:, None] * rho_a[sl][None, :]) * sc_ / denom
            score = idxs + self.zp * cert
            fired = (score > thr[:, None]) & gate[:, None]
            om = ~fired.any(0)  # the union is what the fetch buffer holds
            if not bool(om.any()):
                continue
            w = (score[:, om] - max1[:, None]).exp()
            acc_w += w.sum(-1)
            vals = self._kbuf.index_select(0, rows[sl][om])[:, :kv].float()
            acc_v += w @ vals
        safe = acc_w.clamp_min(D.ENTROPY_EPS)
        logm = torch.where(acc_w > 0, acc_w.log() + max1, ninf)
        return logm, acc_v / safe[:, None]

    @torch.inference_mode()
    @ieee_fp32
    def query(self, qe: torch.Tensor) -> torch.Tensor:
        """qe: [H, 576] one decode step's expanded queries (float).
        Returns absolute pool indices of rows to fetch (possibly empty)."""
        sc_ = self.scale
        qe = qe.float()
        if self.kept_slots.shape[0] == 0:
            H = qe.shape[0]
            max1 = qe.new_full((H,), float("-inf"))  # no baseline: all eligible
            gate = qe.new_ones(H, dtype=torch.bool)
            emarg = 0.0
        else:
            skept = (qe.to(torch.bfloat16) @ self.kept_rows.T).float() * sc_
            max1 = self._thr_base(skept, skept.max(-1).values)  # [H]
            emarg = self._ent_margin(skept)
            p1 = torch.softmax(skept, -1)
            ent = -(p1 * p1.clamp_min(D.ENTROPY_EPS).log()).sum(-1)
            gate = ent > self.thr_g  # [H]
            if not bool(gate.any()):
                return self.arch[:0]
        qsk = qe[:, : D.KV_LORA_RANK] @ self.V.T
        qres = (qe[:, : D.KV_LORA_RANK] - qsk @ self.V).norm(dim=-1)
        idxs = (
            (qe[:, D.KV_LORA_RANK :].to(torch.bfloat16) @ self.side.T).float()
            + (qsk.half() @ self.csk.T).float()
        ) * sc_
        cert = (
            (qres[:, None] * self.rho[None, :]) * sc_ / (D.KV_LORA_RANK - self.r) ** 0.5
        )
        score = idxs + self.zp * cert
        fire = (score > (max1 - self.margin - emarg)[:, None]) & gate[:, None]
        return self.arch[fire.any(0)]

    @ieee_fp32
    @torch.inference_mode()
    def extend_closed(self, new_rows: torch.Tensor, new_slots: torch.Tensor) -> None:
        """Append a newly CLOSED block to the projection caches.
        new_rows: [B, 576] pool rows of the block; new_slots: [B] pool row ids.
        """
        if self._csk_all is None:
            dev = new_rows.device
            self._pos_all = torch.zeros(0, dtype=torch.int64, device=dev)
            self._csk_all = torch.zeros(0, self.r, device=dev, dtype=torch.float16)
            self._rho_all = torch.zeros(0, device=dev)
        Cf = new_rows.float()
        csk = Cf[:, : D.KV_LORA_RANK] @ self.V.T
        rho = (Cf[:, : D.KV_LORA_RANK] - csk @ self.V).norm(dim=-1)
        # No _side_all. The sidecar is the pool row's own tail at KV_LORA_RANK
        # and _pos_all already names every closed row, so carrying a [closed,64]
        # bf16 copy alongside is storing what the pool still holds -- 8 MiB per
        # (layer, request) at 64k, and it grows with the context.
        self._csk_all = torch.cat([self._csk_all, csk.half()])
        self._rho_all = torch.cat([self._rho_all, rho])
        self._pos_all = torch.cat([self._pos_all, new_slots])

    @ieee_fp32
    @torch.inference_mode()
    def refresh_membership(
        self,
        keep: torch.Tensor,
        kbuf: torch.Tensor,
    ) -> None:
        """Re-derive the archive from a NEW keep mask over all closed rows.
        keep: [closed] bool (True = tier-1 keeps it); kbuf the layer's pool
        buffer, which both the sidecars and the kept rows are re-read from by
        row id -- the caller used to gather the kept rows and hand them in,
        which is the gather this now does on demand and only if asked.
        Thresholds (zp, gate) are untouched: the conformal certificate is
        sound for any archive."""
        # Membership changes are an index change, not a data movement: the
        # closed-prefix caches already hold every row's projection, and the
        # archive is a selection over them.
        idx = (~keep).nonzero().flatten()
        self._arch_idx = idx.to(torch.int32)
        self.arch = self._pos_all.index_select(0, idx).to(torch.int32)
        self.kept_slots = self._pos_all[keep].to(torch.int32)
        self._kbuf = kbuf
        self._side_mat = self._kept_mat = None  # re-read from the pool on use
        self._csk_mat = self._rho_mat = None  # re-selected from the caches
        self._qside_t = self._qsk_t = self._hit_buf = None  # re-size lazily
        self.version += 1
