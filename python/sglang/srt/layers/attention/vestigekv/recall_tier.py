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

import math

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.archive_pool import (
    expand_groups,
    pool_operands,
)
from sglang.srt.layers.attention.vestigekv.defaults import ieee_fp32
from sglang.srt.layers.attention.vestigekv.geometry import KIMI_LINEAR, Geometry
from sglang.srt.layers.attention.vestigekv.scan_kernel import vestige_scan


def owned_calibration_queries(local_best, other_bests, rank, ranks):
    """Which calibration queries this rank fits zp on when the archive is split.

    zp is calibrated on the best ARCHIVED row per query, and that best is a max
    over the archive. Each rank holds a shard, so its own max is at or below
    the union's: left alone, every rank calibrates against a worse row than the
    one that exists, zp comes out too small and the certificate fires too
    little -- rows that should be recalled are not. A recall failure, not a
    tightness one.

    So a query belongs to whichever rank actually holds its best archived row.
    Ties go to the lowest rank, which matters only when two shards hold rows of
    exactly equal score and costs nothing to make deterministic.
    """
    import torch

    best = local_best
    for other in other_bests:
        best = torch.maximum(best, other)
    mine = local_best >= best
    for lower in other_bests[:rank]:
        mine &= local_best > lower
    return mine


def zp_from_pooled(z_parts, n_cal_q, recall_target):
    """The conformal quantile over the union of the ranks' z samples.

    The guarantee is marginal over the exchangeable calibration queries, so the
    order statistic has to be taken on all of them together; a per-rank
    quantile of a per-rank sample guarantees nothing about the union. Each rank
    contributes only the queries it owns, so concatenating is the union and
    nothing is counted twice.
    """
    import torch

    pooled = torch.cat([z for z in z_parts if z.numel()])
    if pooled.numel() != n_cal_q:
        # The count and the sample are computed separately -- one from the
        # ownership masks, one from the values they select -- so a sharded
        # caller that double-counts a query or drops a rank shows up here and
        # not as a quantile taken at the wrong k.
        raise ValueError(
            f"pooled z sample is {pooled.numel()} but n_cal_q says {n_cal_q}; "
            "the ownership masks and the values disagree"
        )
    k = D.conformal_k(n_cal_q, recall_target)
    return min(float(pooled.kthvalue(k).values), D.Z_MAX)


class RecallTier:
    def __init__(
        self,
        r: int = D.INDEX_RANK,
        recall_target: float = D.RECALL_TARGET,
        scale: float | None = None,
        margin: float = 0.0,
        threshold: str = "max",
        geom: Geometry = KIMI_LINEAR,
    ):
        self.r = r
        self.recall_target = recall_target
        self.geom = geom
        self.scale = geom.attn_scale if scale is None else scale
        self.margin = margin  # scan threshold = base - margin (see VestigeKVConfig)
        self.threshold = threshold  # base: "max" kept score or "lse" of kept scores
        self.built = False
        # live-archive projection caches: None until the first decode-time
        # close backfills them (extend_closed); _pos_all doubles as the fill
        # watermark read by the close path.
        self._pos_all = self._csk_all = self._rho_all = None
        # Pooled view of the closed-prefix caches, the scan's unit when
        # pool_size > 1 (vestigekv/archive_pool.py). Kept BESIDE the per-row
        # caches rather than replacing them: the close path backfills by row,
        # the pack addresses pool rows, and tier 1 re-decides membership per
        # row, so the row form stays the store and pooling is a second view.
        self._csk_pool = self._rho_pool = self._spread_pool = None
        self.pool_size = envs.SGLANG_VESTIGEKV_ARCHIVE_POOL.get()
        # Sketch storage precision. fp8 halves the scan's dominant load; the
        # dot is fp8 x fp8 because Triton has no mixed fp8 dot on SM120
        # (verified 2026-09-23), so the query is rounded the same way. One
        # power-of-two scale per tier keeps the dequantisation a scalar the
        # kernel folds into the attention scale instead of a per-row load.
        self._csk_fp8 = envs.SGLANG_VESTIGEKV_CSK_FP8.get()
        self.csk_scale = 1.0
        self._side_mat = None  # lazily materialised; see the `side` property
        self._kept_mat = None  # lazily materialised; see `kept_rows`
        self._csk_mat = self._rho_mat = None  # selections over the _all caches
        self._arch_idx = None  # positions of the archive in the closed prefix
        self._arch_mat = None  # lazily materialised; see `arch`
        # Index tables. These are what the tier STORES; the row tables
        # (side, kept_rows) and the archive selections (csk, rho, arch) are
        # properties over them.
        self.kept_slots = None  # pool row ids tier-1 keeps
        self._kbuf = None  # the layer's pool buffer, to re-read from
        self.version = 0  # bumped on in-place membership refresh (pack sync key)
        self._scatter_buf = None  # reused static-shape scatter target (query_fixed)
        # fixed-address staging for the fused scan (capturable)
        self._qside_t = self._qsk_t = self._hit_buf = self._inf = None

    @property
    def csk_dtype(self):
        return torch.float8_e4m3fn if self._csk_fp8 else torch.float16

    def _set_csk_scale(self, c: torch.Tensor) -> None:
        """Fix the tier's sketch scale from its first built chunk.

        A power of two so the quantisation is a pure exponent shift and the
        scale round-trips exactly. It is fixed for the tier's life because
        extend_closed appends later blocks into the same cache, and a rescale
        would leave the rows already stored on a different footing.
        """
        if not self._csk_fp8:
            self.csk_scale = 1.0
            return
        amax = float(c.abs().amax())
        self.csk_scale = 2.0 ** math.ceil(math.log2(max(amax, 1e-9) / 448.0))

    def _q_csk(self, c: torch.Tensor) -> torch.Tensor:
        if not self._csk_fp8:
            return c.half()
        q = c / self.csk_scale
        torch._assert_async((q.abs().amax() < 448.0).to(torch.bool))
        return q.to(torch.float8_e4m3fn)

    def _deq_csk(self, c: torch.Tensor) -> torch.Tensor:
        """Stored sketch as fp32 in the ORIGINAL units, for the torch paths
        that fit and check the certificate. The kernel never calls this: it
        folds the scale into the attention scale instead."""
        return c.float() * self.csk_scale if self._csk_fp8 else c.float()

    def _q_query(self, qsk: torch.Tensor):
        """(rounded query, scale). Rounded to the sketch's storage dtype so zp
        is calibrated on exactly the operands the kernel multiplies."""
        if not self._csk_fp8:
            return qsk.half(), 1.0
        s = 2.0 ** math.ceil(math.log2(max(float(qsk.abs().amax()), 1e-9) / 448.0))
        return (qsk / s).to(torch.float8_e4m3fn), s

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

    def drop_operands(self):
        """Release the materialised archive selections; the properties
        re-derive them from the closed-prefix caches."""
        self._csk_mat = self._rho_mat = None

    @property
    def arch(self):
        """Pool row ids of the archive, [A] int32 -- a selection over _pos_all,
        on the same terms as `csk`: the in-graph pack holds its own copy in
        the arena, so the tier keeps only the index it is derived from."""
        if self._arch_mat is None:
            if self._pos_all is None or self._arch_idx is None:
                raise RuntimeError(
                    "arch is not populated; build()/refresh_membership() must "
                    "record the closed prefix before the archive can be selected"
                )
            self._arch_mat = self._pos_all.index_select(
                0, self._arch_idx.to(torch.int64)
            ).to(torch.int32)
        return self._arch_mat

    @arch.setter
    def arch(self, v):
        self._arch_mat = v

    # ---- the scan's unit: a row when pool_size == 1, a group when it is not ----

    def refresh_pool(self) -> None:
        """(Re)build the pooled view from the closed-prefix caches.

        Pools the WHOLE closed prefix, not the archive selection. Grouping has
        to be the same grouping DSA uses -- group g is closed-prefix rows
        g*P .. g*P+P-1 -- or a fired group cannot be mapped back to rows; a
        selection compacted first would group rows that are not adjacent and
        the mapping would be to the wrong tokens. With rho = 1/32 nearly every
        group holds an evicted row anyway, so scanning all of them costs
        almost nothing over scanning only those that do.
        """
        if self.pool_size <= 1 or self._csk_all is None:
            self._csk_pool = self._rho_pool = self._spread_pool = None
            return
        self._csk_pool, self._rho_pool, self._spread_pool = pool_operands(
            self._csk_all, self._rho_all, self.pool_size
        )

    @property
    def scan_operands(self):
        """(csk, rho, spread, n) the scan runs over.

        Unpooled this is the archive selection and spread is None, which is
        the current path bit for bit. Pooled it is every group of the closed
        prefix, and spread is what the certificate must inflate by -- a scan
        that ignores it is calibrated and wrong.
        """
        if self.pool_size <= 1:
            return self.csk, self.rho, None, self.n_arch
        if self._csk_pool is None:
            self.refresh_pool()
        return (
            self._csk_pool,
            self._rho_pool,
            self._spread_pool,
            int(self._csk_pool.shape[0]),
        )

    def scan_rows(self, fired) -> torch.Tensor:
        """Pool row ids for the entries the scan fired, [N] int32.

        Unpooled, `fired` indexes the archive directly. Pooled, it indexes
        groups, and each one expands to its P closed-prefix positions, which
        _pos_all turns into pool row ids. Rows tier 1 already keeps come back
        here too -- they are attended regardless, so dropping them is the
        caller's dedupe, not a correctness matter.
        """
        if self.pool_size <= 1:
            return self.arch.index_select(0, fired.to(torch.int64))
        pos = expand_groups(fired.to(torch.int64), self.pool_size)
        pos = pos[pos < self._pos_all.shape[0]]
        return self._pos_all.index_select(0, pos).to(torch.int32)

    @property
    def n_arch(self) -> int:
        """Archive size without materialising the selection."""
        if self._arch_idx is not None:
            return int(self._arch_idx.shape[0])
        return int(self.arch.shape[0])

    def drop_arch(self):
        """Release the materialised archive row ids; the property re-derives them."""
        self._arch_mat = None

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
        g = self.geom
        self._side_mat = self._kbuf[self.arch][
            :, g.kv_lora_rank : g.kv_lora_rank + g.side_dim
        ].contiguous()
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
        """rows: [T,latent_dim] pool rows (fp32). keep: [T] bool tier-1 mask.
        q_cal: [n,H,latent_dim] expanded calibration queries; q_pos: [n] positions.
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
        qe = q_cal.reshape(-1, self.geom.latent_dim).float()  # [n*H, 576]

        qcal_c = qe[:, : self.geom.kv_lora_rank] - qe[:, : self.geom.kv_lora_rank].mean(0, keepdim=True)
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
            V = torch.zeros(self.r, self.geom.kv_lora_rank, device=dev, dtype=qe.dtype)
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
            assert torch.equal(donor_pos, row_slots.to(donor_pos.dtype)), (
                "operand reuse from a tier holding a different prefix row-set"
            )
            self.V = V = operands_from.V
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
            from sglang.srt.layers.attention.vestigekv.operand_fused import (
                build_operands_fused,
            )

            if kbuf.is_cuda and kbuf.dtype == torch.bfloat16:
                # Fused single-kernel operand build: gather + project +
                # residual + casts (operand_fused.py). Same fp32-ieee
                # arithmetic; ~1 ulp reduction-order difference vs cuBLAS,
                # gated by fire-set stability + retrieval. Per row, so the
                # values on the archived subset do not depend on the row set.
                # side_dim=0: the kernel's sidecar store is gated by that
                # constexpr, and the tier never keeps the sidecar (it is the
                # pool row's own tail); writing it was a [closed, 64] bf16
                # transient at every build.
                _fused_csk, self._rho_all, _ = build_operands_fused(
                    kbuf, row_slots, V, kv=self.geom.kv_lora_rank, side_dim=0
                )
                # The fused kernel emits fp16. Requantising here rather than
                # teaching it the dtype keeps one quantisation rule for both
                # build paths; an fp16 lhs against the fp8 query is a dot
                # Triton refuses outright, which is how the split was found.
                self._set_csk_scale(_fused_csk)
                self._csk_all = self._q_csk(_fused_csk.float())
                del _fused_csk
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
                self._csk_all = torch.empty(T, self.r, device=dev, dtype=self.csk_dtype)
                self._rho_all = torch.empty(T, device=dev, dtype=torch.float32)
                for a0 in range(0, T, D.BUILD_ROW_CHUNK):
                    a1 = min(a0 + D.BUILD_ROW_CHUNK, T)
                    blk = kbuf[row_slots[a0:a1]].float()
                    content = blk[:, : self.geom.kv_lora_rank]
                    c = content @ V.T
                    torch._assert_async(
                        (c.abs().amax() < 6e4).to(torch.bool)
                    )  # fp16 range guard: a violation here is a model-scale anomaly
                    if a0 == 0:
                        self._set_csk_scale(c)
                    self._csk_all[a0:a1] = self._q_csk(c)
                    self._rho_all[a0:a1] = (content - c @ V).norm(dim=-1)
                    del blk, content, c
        # Index tables in place, and the closed-prefix caches now cover the
        # built prefix, so _pos_all is an honest backfill watermark for the
        # close path. Everything below (calibration included) reads the row
        # tables and the archive selections through these.
        self._kbuf = kbuf
        # int32 throughout: a pool row id indexes kbuf, so it is below
        # kbuf.shape[0]; the check is host-side and free, and it is what keeps
        # the narrowing from wrapping silently. _pos_all is the one per-row
        # table the tier keeps for the request's life besides csk/rho; arch is
        # a selection over it, and no kernel reads either (the pack copies
        # into its own dtype-checked int32 arenas).
        assert kbuf.shape[0] < 2**31, "pool row ids do not fit int32"
        self._pos_all = row_slots.to(torch.int32)
        self._arch_idx = arch_idx.to(torch.int32)
        self._arch_mat = None
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
        if diag:
            # Debug telemetry: how many ARCHIVED rows each calibration query
            # truly prefers to its best kept row (the recall need), against
            # what the certificate fires below.
            max1_d = (qe.to(torch.bfloat16) @ self.kept_rows.T).float().mul_(sc_).max(-1).values
            need = torch.zeros(nq, dtype=torch.int64, device=dev)
        for k0 in range(0, T, KB):
            k1 = min(k0 + KB, T)
            # In place throughout: `* sc_` and `masked_fill` each copied the
            # whole [n*H, KB] block, 67 MB per chunk that the allocator then had
            # to find under a serving mem-fraction.
            cblk = kbuf[row_slots[k0:k1]].float()
            blk = qe @ cblk.T
            del cblk
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
        self.need_more_hard = n_cal_q < D.min_hard(self.recall_target)
        if self.need_more_hard:
            # Not enough evidence for the guarantee yet: serve with the safety
            # clamp (over-fetches, never under-recalls) and tell the caller to
            # keep collecting.
            self.zp = D.Z_MAX
        else:
            qh = qe[has_arch]
            qskh = qh[:, : self.geom.kv_lora_rank] @ V.T
            qresh = (qh[:, : self.geom.kv_lora_rank] - qskh @ V).norm(dim=-1)
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
                :, self.geom.kv_lora_rank : self.geom.latent_dim
            ]  # [n, side_dim]
            src = self._arch_idx.to(torch.int64).index_select(0, pa64)
            tgt_csk = self._csk_all.index_select(0, src)  # [n, r]
            tgt_rho = self._rho_all.index_select(0, src)  # [n]
            # Same rounding as the serve-time scoring path: query operands
            # rounded to the storage dtypes before the (exact-in-fp32)
            # products, so zp is calibrated on exactly what the kernel scores.
            idxs_t = (
                qh[:, self.geom.kv_lora_rank :].to(torch.bfloat16).float() * tgt_side.float()
            ).sum(-1) + (self._q_query(qskh)[0].float() * self._deq_csk(tgt_csk)).sum(-1)
            idxs_t = idxs_t * sc_
            cert_t = (
                qresh * tgt_rho * sc_ / (self.geom.kv_lora_rank - self.r) ** 0.5
            ).clamp_min(D.ENTROPY_EPS)
            z_req = (abest_val[has_arch] - idxs_t) / cert_t
            k = D.conformal_k(n_cal_q, self.recall_target)
            self.zp = min(float(z_req.kthvalue(k).values), D.Z_MAX)
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
        idxs = (self._q_query(qsk)[0].float() @ self._deq_csk(self.csk).T) * sc_
        if D.SIDECAR_DIM:
            idxs = idxs + (qe[:, D.KV_LORA_RANK :].to(torch.bfloat16).float() @ self.side.float().T) * sc_
        cert = (qres[:, None] * self.rho[None, :]) * sc_ / (D.KV_LORA_RANK - self.r) ** 0.5
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
        bound_g = (qsk @ cbar.T) * sc_ + (qsk.norm(dim=-1)[:, None] * delta[None, :]) * sc_
        bound_g = bound_g + (qres[:, None] * rho_g[None, :]) * sc_ * zp / (D.KV_LORA_RANK - self.r) ** 0.5
        gfire = (bound_g > (max1 - self.margin)[:, None]).sum(-1) * 4
        return {
            "gfire_p50": q(gfire, 0.5), "gfire_p90": q(gfire, 0.9),
            "delta_over_c_p50": q(delta / cbar.norm(dim=-1).clamp_min(1e-6), 0.5),
            "A": A,
            "need_p50": q(need, 0.5), "need_p90": q(need, 0.9), "need_max": int(need.max()),
            "fire_p50": q(fire, 0.5), "fire_p90": q(fire, 0.9),
            "fire_sketch_only_p50": q(fire0, 0.5),
            "max1_p50": q(max1, 0.5),
            "cert_p50": q(cert.median(dim=-1).values * zp, 0.5),
            "rho_p50": q(self.rho, 0.5), "qres_p50": q(qres, 0.5),
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
        else:
            skept = (qe.to(torch.bfloat16) @ self.kept_rows.T).float() * sc_
            max1 = self._thr_base(skept, skept.max(-1).values)
            p1 = torch.softmax(skept, -1)
            ent = -(p1 * p1.clamp_min(D.ENTROPY_EPS).log()).sum(-1)
            gate = ent > self.thr_g
        qsk = qe[:, : self.geom.kv_lora_rank] @ self.V.T
        qres = (qe[:, : self.geom.kv_lora_rank] - qsk @ self.V).norm(dim=-1)
        if self._qside_t is None:
            H = qe.shape[0]
            # Storage dtypes, matching the kernel's native-dtype dots: bf16 for
            # the sidecar branch (exact -- qbuf is bf16), fp16 for the sketch
            # projection (rounded here AND at calibration, so zp covers it).
            self._qside_t = qe.new_empty(self.geom.side_dim, H, dtype=torch.bfloat16)
            self._qsk_t = qe.new_empty(self.r, H, dtype=self.csk_dtype)
            # Sized off the index table: self.side.shape[0] would gather the
            # whole [A, 64] sidecar to read one integer.
            self._hit_buf = torch.empty(self.n_arch, dtype=torch.int32, device=qe.device)
            self._inf = qe.new_full((), float("inf"))
        self._qside_t.copy_(qe[:, self.geom.kv_lora_rank :].T)
        qsk_q, q_scale = self._q_query(qsk)
        self._qsk_t.copy_(qsk_q.T)
        # The dot runs on the STORED operands, so its product carries both
        # quantisation scales; folding them into the attention scale keeps the
        # kernel's arithmetic unchanged and costs no per-row load. The
        # certificate's own term is in original units and uses sc_ untouched.
        sc_dot = sc_ * self.csk_scale * q_scale
        # Fold the gate into the threshold: a closed head can never fire, so
        # +inf makes it lose every comparison and the kernel needs no second
        # predicate. Scores stay in registers -- see scan_kernel for why
        # that, not arithmetic, is what the scan costs.
        max1g = torch.where(gate, max1 - self.margin, self._inf)
        hit = (
            vestige_scan(
                self._qside_t,
                self._qsk_t,
                qres,
                max1g,
                self.side,
                self.csk,
                self.rho,
                sc_dot,
                self.zp * sc_ / (self.geom.kv_lora_rank - self.r) ** 0.5,
                out=self._hit_buf,
            )
            != 0
        )
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
        out_ovf[slot] = total > W

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
        else:
            skept = (qe.to(torch.bfloat16) @ self.kept_rows.T).float() * sc_
            max1 = self._thr_base(skept, skept.max(-1).values)  # [H]
            p1 = torch.softmax(skept, -1)
            ent = -(p1 * p1.clamp_min(D.ENTROPY_EPS).log()).sum(-1)
            gate = ent > self.thr_g  # [H]
            if not bool(gate.any()):
                return self.arch[:0]
        qsk = qe[:, : self.geom.kv_lora_rank] @ self.V.T
        qres = (qe[:, : self.geom.kv_lora_rank] - qsk @ self.V).norm(dim=-1)
        idxs = (
            (qe[:, self.geom.kv_lora_rank :].to(torch.bfloat16) @ self.side.T).float()
            + (self._q_query(qsk)[0].float() @ self._deq_csk(self.csk).T)
        ) * sc_
        cert = (
            (qres[:, None] * self.rho[None, :]) * sc_ / (self.geom.kv_lora_rank - self.r) ** 0.5
        )
        score = idxs + self.zp * cert
        fire = (score > (max1 - self.margin)[:, None]) & gate[:, None]
        return self.arch[fire.any(0)]

    @ieee_fp32
    @torch.inference_mode()
    def extend_closed(self, new_rows: torch.Tensor, new_slots: torch.Tensor) -> None:
        """Append a newly CLOSED block to the projection caches.
        new_rows: [B, 576] pool rows of the block; new_slots: [B] pool row ids.
        """
        if self._csk_all is None:
            dev = new_rows.device
            self._pos_all = torch.zeros(0, dtype=torch.int32, device=dev)
            self._csk_all = torch.zeros(0, self.r, device=dev, dtype=torch.float16)
            self._rho_all = torch.zeros(0, device=dev)
        Cf = new_rows.float()
        csk = Cf[:, : self.geom.kv_lora_rank] @ self.V.T
        rho = (Cf[:, : self.geom.kv_lora_rank] - csk @ self.V).norm(dim=-1)
        # No _side_all. The sidecar is the pool row's own tail at KV_LORA_RANK
        # and _pos_all already names every closed row, so carrying a [closed,64]
        # bf16 copy alongside is storing what the pool still holds -- 8 MiB per
        # (layer, request) at 64k, and it grows with the context.
        self._csk_all = torch.cat([self._csk_all, self._q_csk(csk)])
        self._rho_all = torch.cat([self._rho_all, rho])
        self._pos_all = torch.cat([self._pos_all, new_slots.to(torch.int32)])

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
        self.kept_slots = self._pos_all[keep].to(torch.int32)
        self._kbuf = kbuf
        self._side_mat = self._kept_mat = None  # re-read from the pool on use
        self._csk_mat = self._rho_mat = self._arch_mat = None  # re-selected from the caches
        self._qside_t = self._qsk_t = self._hit_buf = None  # re-size lazily
        self.version += 1
