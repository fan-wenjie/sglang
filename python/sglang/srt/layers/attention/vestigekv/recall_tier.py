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
        self._pos_all = self._side_all = self._csk_all = self._rho_all = None
        self.version = 0  # bumped on in-place membership refresh (pack sync key)
        self._scatter_buf = None  # reused static-shape scatter target (query_fixed)
        # fixed-address staging for the fused scan (capturable)
        self._qside_t = self._qsk_t = self._hit_buf = self._inf = None

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
    ) -> dict:
        """rows: [T,576] pool rows (fp32). keep: [T] bool tier-1 mask.
        q_cal: [n,H,576] expanded calibration queries; q_pos: [n] positions.
        Returns stats. All thresholds derive from the prefix itself."""
        sc_ = self.scale
        T = row_slots.numel()
        dev = row_slots.device
        H = q_cal.shape[1]
        self.keep = keep
        self.arch = (~keep).nonzero().flatten()
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
        else:
            evals, evecs = torch.linalg.eigh(qcal_c.T @ qcal_c)
            V = evecs[:, -self.r :].T.flip(0)  # descending singular value order
        self.V = V
        # Chunked, and the pool rows are never materialized in fp32 as a whole.
        # The unchunked form allocated the full [T, 576] fp32 copy plus TWO
        # [A, 512] content copies (csk and rho each indexed it), ~429 MB per
        # build at S=64k. Ten builds per request of that, against a serving
        # mem-fraction, is why an in-server build cost 3.8x its isolated time.
        A = int(self.arch.numel())
        arch_slots = row_slots[self.arch]
        self.csk = torch.empty(A, self.r, device=dev, dtype=torch.float32)
        self.rho = torch.empty(A, device=dev, dtype=torch.float32)
        self.side = torch.empty(A, D.SIDECAR_DIM, device=dev, dtype=torch.float32)
        for a0 in range(0, A, D.BUILD_ROW_CHUNK):
            a1 = min(a0 + D.BUILD_ROW_CHUNK, A)
            blk = kbuf[arch_slots[a0:a1]].float()
            content = blk[:, : D.KV_LORA_RANK]
            c = content @ V.T
            self.csk[a0:a1] = c
            self.rho[a0:a1] = (content - c @ V).norm(dim=-1)
            self.side[a0:a1] = blk[:, D.KV_LORA_RANK :]
            del blk, content, c
        self.kept_rows = kbuf[row_slots[keep]].float()

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
                "arch": int(self.arch.numel()),
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

        skept = (qe @ self.kept_rows.T) * sc_
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
        pos_in_arch = torch.searchsorted(self.arch.contiguous(), tgt)
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
            tgt_side = self.side[pa]  # [n_hard, 64]
            tgt_csk = self.csk[pa]  # [n_hard, r]
            tgt_rho = self.rho[pa]  # [n_hard]
            idxs_t = (qh[:, D.KV_LORA_RANK :] * tgt_side).sum(-1) + (
                qskh * tgt_csk
            ).sum(-1)
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
        # Caches stay None here BY CONTRACT: the close path reads
        # `0 if _pos_all is None else _pos_all.shape[0]` as its backfill
        # watermark, so the first decode-time close projects rows 0..c1 in
        # one batch. An eager _pos_all with lazy side/csk desynchronized the
        # watermark from the cache contents (first serving close: refresh
        # indexed a 4096-row cache with full-prefix indices).
        self._pos_all = self._side_all = self._csk_all = self._rho_all = None
        self.built = True
        return {
            "zp": zp,
            "gate_off": thr_g == float("-inf"),
            "n_hard": n_hard,
            "need_more_hard": self.need_more_hard,
            "arch": int(self.arch.numel()),
        }

    @torch.inference_mode()
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
        skept = (qe @ self.kept_rows.T) * sc_
        max1 = skept.max(-1).values
        p1 = torch.softmax(skept, -1)
        ent = -(p1 * p1.clamp_min(D.ENTROPY_EPS).log()).sum(-1)
        gate = ent > self.thr_g
        qsk = qe[:, : D.KV_LORA_RANK] @ self.V.T
        qres = (qe[:, : D.KV_LORA_RANK] - qsk @ self.V).norm(dim=-1)
        if self._qside_t is None:
            H = qe.shape[0]
            self._qside_t = qe.new_empty(D.SIDECAR_DIM, H)
            self._qsk_t = qe.new_empty(self.r, H)
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
            idxs = (qe[:, D.KV_LORA_RANK :] @ self.side.T + qsk @ self.csk.T) * sc_
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
    def query(self, qe: torch.Tensor) -> torch.Tensor:
        """qe: [H, 576] one decode step's expanded queries (float).
        Returns absolute pool indices of rows to fetch (possibly empty)."""
        sc_ = self.scale
        qe = qe.float()
        skept = (qe @ self.kept_rows.T) * sc_
        max1 = skept.max(-1).values  # [H]
        p1 = torch.softmax(skept, -1)
        ent = -(p1 * p1.clamp_min(D.ENTROPY_EPS).log()).sum(-1)
        gate = ent > self.thr_g  # [H]
        if not bool(gate.any()):
            return self.arch[:0]
        qsk = qe[:, : D.KV_LORA_RANK] @ self.V.T
        qres = (qe[:, : D.KV_LORA_RANK] - qsk @ self.V).norm(dim=-1)
        idxs = (qe[:, D.KV_LORA_RANK :] @ self.side.T + qsk @ self.csk.T) * sc_
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


    @torch.inference_mode()
    def extend_closed(self, new_rows: torch.Tensor, new_slots: torch.Tensor) -> None:
        """Append a newly CLOSED block to the projection caches.
        new_rows: [B, 576] pool rows of the block; new_slots: [B] pool row ids.
        """
        if self._side_all is None:
            dev = new_rows.device
            self._pos_all = torch.zeros(0, dtype=torch.int64, device=dev)
            self._side_all = torch.zeros(0, D.SIDECAR_DIM, device=dev)
            self._csk_all = torch.zeros(0, self.r, device=dev)
            self._rho_all = torch.zeros(0, device=dev)
        Cf = new_rows.float()
        csk = Cf[:, : D.KV_LORA_RANK] @ self.V.T
        side = Cf[:, D.KV_LORA_RANK :]
        rho = (Cf[:, : D.KV_LORA_RANK] - csk @ self.V).norm(dim=-1)
        self._side_all = torch.cat([self._side_all, side])
        self._csk_all = torch.cat([self._csk_all, csk])
        self._rho_all = torch.cat([self._rho_all, rho])
        self._pos_all = torch.cat([self._pos_all, new_slots])

    @torch.inference_mode()
    def refresh_membership(self, keep: torch.Tensor, kept_rows: torch.Tensor) -> None:
        """Re-derive the archive from a NEW keep mask over all closed rows.
        keep: [closed] bool (True = tier-1 keeps it); kept_rows: [n_keep, 576]
        the live kept rows for the max1 competition. Thresholds (zp, gate)
        are untouched: the conformal certificate is sound for any archive."""
        arch_idx = (~keep).nonzero().flatten()
        self.side = self._side_all[arch_idx].contiguous()
        self.csk = self._csk_all[arch_idx].contiguous()
        self.rho = self._rho_all[arch_idx].contiguous()
        self.arch = self._pos_all[arch_idx].contiguous()
        self.kept_rows = kept_rows.float()
        self._qside_t = self._qsk_t = self._hit_buf = None  # re-size lazily
        self.version += 1
