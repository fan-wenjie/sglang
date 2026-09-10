"""GPU-resident recall tier for VestigeKV (PREREG19/20).

Vendored VERBATIM from the validated mini-sglang stack
(minisgl/kimi/tier2.py). It is pure-tensor (operates on [T, 576] latent rows and
[H, 576] expanded queries), so the same code runs unchanged over sglang's
MLATokenToKVPool rows. Kept bit-identical to the reference so the two
implementations can be asserted equal (see test/manual/test_vestige_equiv.py); fix
record-invalidating bugs in lockstep with the reference, never one-sided.

Index per MLA slot, built once at a compression event: exact 64-dim sidecar
summand over archived rows, rank-r sketch of the 512-dim content (PCA basis of
prefix queries), residual-norm certificate self-calibrated (z) on prefix queries
with full-cache labels, entropy gate threshold from the same calibration
(auto-off when it cannot separate, PREREG24). Per decode query:
score = sidecar + sketch + z*certificate; fire where score beats the tier-1 max
and the gate is open; fetch top-j fired archived rows.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.defaults import ieee_fp32
from sglang.srt.layers.attention.vestigekv.scan_kernel import vestige_scan


class RecallTier:
    def __init__(
        self,
        r: int = D.INDEX_RANK,
        topj: int = -1,
        recall_target: float = D.RECALL_TARGET,
        scale: float = D.ATTN_SCALE,
    ):
        # topj: per-head fetch cap. DEFAULT -1 = uncapped (fetch the full fired
        # set; the fool-proof default does the WHOLE thing, never a fraction
        # nobody typed). Set topj > 0 explicitly to enable the bounded-fetch
        # guarantee (fetch <= topj*num_heads rows/step/layer); recommended cap
        # is 16. Measured (PREREG33/34): the cap is nearly free at
        # batch=1/<=32k and changes generation not at all (bit-identical text),
        # trading a sub-0.001-bpb likelihood sliver for the worst-case bound
        # (uncapped fire max 3983 rows vs capped 296).
        self.r = r
        self.topj = topj
        self.recall_target = recall_target
        self.scale = scale
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
            # A sketch basis carried over from this layer's previous calibrated
            # build. The certificate is a Cauchy-Schwarz bound on the truncation
            # error and stays sound for ANY orthonormal basis, so reusing one is
            # a cost saving, not an approximation -- and a basis fitted on real
            # queries beats a PCA of cache-row proxies, which is all the
            # provisional build could compute for itself.
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
            assert torch.equal(donor_pos, row_slots), (
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
                self._csk_all, self._rho_all, _side = build_operands_fused(
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
        KB = D.BUILD_KEY_CHUNK
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
            bval, bidx = blk.max(-1)
            take = bval > best_val
            best_val = torch.where(take, bval, best_val)
            tgt = torch.where(take, bidx + k0, tgt)
            del blk
        del best_val
        hard = ~keep[tgt]

        skept = (qe.to(torch.bfloat16) @ self.kept_rows.T).float() * sc_
        p1 = torch.softmax(skept, -1)
        ent = -(p1 * p1.clamp_min(D.ENTROPY_EPS).log()).sum(-1)
        max1 = skept.max(-1).values
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

        # zp in closed form: for hard sample q the archived target wins iff
        # idxs_t + z*cert_t > max1, i.e. z > z_q := (max1 - idxs_t)/cert_t.
        # Over n exchangeable hard samples the k-th order statistic of z_q at
        # k = ceil((n+1)*scan_target) is the conformal quantile: it carries the
        # distribution-free marginal guarantee P(recovered) >= scan_target.
        # This replaces the former 11-rung ladder search exactly (the ladder
        # was a discretization of this quantile) and needs no fallback rung.
        # Positional, not pool ids: tgt is an index into the closed prefix
        # and _arch_idx is the archive's positions in it, ascending.
        pos_in_arch = torch.searchsorted(self._arch_idx.to(torch.int64), tgt)
        self.need_more_hard = n_hard < D.min_hard(self.recall_target)
        if n_hard == 0 or self.need_more_hard:
            # Not enough evidence for the guarantee yet: serve with the safety
            # clamp (over-fetches, never under-recalls) and tell the caller to
            # keep collecting.
            self.zp = D.Z_MAX
        else:
            qh = qe[hard]
            qskh = qh[:, : D.KV_LORA_RANK] @ V.T
            qresh = (qh[:, : D.KV_LORA_RANK] - qskh @ V).norm(dim=-1)
            pa = pos_in_arch[hard]
            # Gather the n_hard rows out of the caches directly. Going through
            # self.side / self.csk / self.rho would materialise the WHOLE
            # archive's selection to read a few dozen rows of it, and those
            # three views are the entire difference between the steady-state
            # footprint and the peak -- which is the figure that decides
            # whether a long request fits.
            pa64 = pa.to(torch.int64)
            arch_rows = self.arch.to(torch.int64).index_select(0, pa64)
            tgt_side = self._kbuf.index_select(0, arch_rows)[
                :, D.KV_LORA_RANK :
            ]  # [n_hard, 64]
            src = self._arch_idx.to(torch.int64).index_select(0, pa64)
            tgt_csk = self._csk_all.index_select(0, src)  # [n_hard, r]
            tgt_rho = self._rho_all.index_select(0, src)  # [n_hard]
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
            z_req = (max1[hard] - idxs_t) / cert_t
            k = D.conformal_k(n_hard, self.recall_target)
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
        return {
            "zp": zp,
            "gate_off": thr_g == float("-inf"),
            "n_hard": n_hard,
            "need_more_hard": self.need_more_hard,
            "arch": int(arch_idx.numel()),
        }

    @torch.inference_mode()
    @ieee_fp32
    def query_fixed(
        self, qe: torch.Tensor, out: torch.Tensor, out_len: torch.Tensor, slot: int
    ) -> None:
        """Device-only variant of query(): writes the fired pool rows into
        out[slot, :W] and their count into out_len[slot] with NO host sync
        (no .numel(), no bool() early-exit). Serving decode uses this; query()
        stays as the reference/equivalence-tested form. Overflow past W is
        truncated -- W is sized above the observed worst fire."""
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
            max1 = skept.max(-1).values
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
        if self.topj is not None and self.topj > 0:
            # Capped fetch needs the per-head ranking, so it keeps the eager
            # form: the fused kernel reduces over heads and never materializes
            # the score matrix the top-j selection ranks. The cap is explicit
            # opt-in; the default (uncapped) path is the fused one.
            idxs = (
                (qe[:, D.KV_LORA_RANK :].to(torch.bfloat16) @ self.side.T).float()
                + (qsk.half() @ self.csk.T).float()
            ) * sc_
            cert = (
                (qres[:, None] * self.rho[None, :])
                * sc_
                / (D.KV_LORA_RANK - self.r) ** 0.5
            )
            score = idxs + self.zp * cert
            fire = (score > max1[:, None]) & gate[:, None]
            topj = score.topk(min(self.topj, score.shape[1]), dim=-1).indices
            fetch = torch.zeros_like(fire)
            fetch.scatter_(1, topj, True)
            fetch &= fire
            hit = fetch.any(0)  # [A] bool, stays on device
        else:
            # Fold the gate into the threshold: a closed head can never fire, so
            # +inf makes it lose every comparison and the kernel needs no second
            # predicate. Scores stay in registers -- see scan_kernel for why
            # that, not arithmetic, is what the scan costs.
            max1g = torch.where(gate, max1, self._inf)
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
        W = out.shape[1]
        n = hit.sum().clamp(max=W)  # [] int64 on device
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
            max1 = skept.max(-1).values  # [H]
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
        fire = (score > max1[:, None]) & gate[:, None]
        if self.topj is not None and self.topj > 0:
            topj = score.topk(min(self.topj, score.shape[1]), dim=-1).indices
            fetch = torch.zeros_like(fire)
            fetch.scatter_(1, topj, True)
            fetch &= fire
        else:  # topj <= 0 (-1): uncapped, full fired set
            fetch = fire
        return self.arch[fetch.any(0)]

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
