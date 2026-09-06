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


class RecallTier:
    def __init__(
        self,
        r: int = 64,
        topj: int = -1,
        recall_target: float = 0.90,
        scale: float = 192**-0.5,
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

    @torch.inference_mode()
    def build(
        self,
        rows: torch.Tensor,
        keep: torch.Tensor,
        q_cal: torch.Tensor,
        q_pos: torch.Tensor,
    ) -> dict:
        """rows: [T,576] pool rows (fp32). keep: [T] bool tier-1 mask.
        q_cal: [n,H,576] expanded calibration queries; q_pos: [n] positions.
        Returns stats. All thresholds derive from the prefix itself."""
        sc_ = self.scale
        Cf = rows.float()
        T = Cf.shape[0]
        H = q_cal.shape[1]
        self.keep = keep
        self.arch = (~keep).nonzero().flatten()
        qe = q_cal.reshape(-1, 576).float()  # [n*H, 576]

        qcal_c = qe[:, :512] - qe[:, :512].mean(0, keepdim=True)
        V = torch.linalg.svd(qcal_c, full_matrices=False)[2][: self.r]
        self.V = V
        self.csk = Cf[self.arch, :512] @ V.T  # [A, r]
        self.rho = (Cf[self.arch, :512] - self.csk @ V).norm(dim=-1)
        self.side = Cf[self.arch, 512:]  # [A, 64]
        self.kept_rows = Cf[keep]

        # calibration: full-cache causal labels (pool is complete by invariant).
        # CHUNKED over keys to avoid a [n*H, T] materialization (OOMs at 512k).
        qpos_r = q_pos.repeat_interleave(H)
        nq = qe.shape[0]
        best_val = torch.full((nq,), torch.finfo(torch.float32).min, device=Cf.device)
        tgt = torch.zeros(nq, dtype=torch.long, device=Cf.device)
        KB = 16384
        for k0 in range(0, T, KB):
            k1 = min(k0 + KB, T)
            blk = (qe @ Cf[k0:k1].T) * sc_
            colk = torch.arange(k0, k1, device=Cf.device)[None, :]
            blk = blk.masked_fill(
                colk > qpos_r[:, None], torch.finfo(torch.float32).min
            )
            bval, bidx = blk.max(-1)
            take = bval > best_val
            best_val = torch.where(take, bval, best_val)
            tgt = torch.where(take, bidx + k0, tgt)
            del blk
        del best_val
        hard = ~keep[tgt]

        skept = (qe @ self.kept_rows.T) * sc_
        p1 = torch.softmax(skept, -1)
        ent = -(p1 * p1.clamp_min(1e-12).log()).sum(-1)
        max1 = skept.max(-1).values
        del skept, p1

        n_hard = int(hard.sum())
        thr_g = float(ent[hard].quantile(0.03)) if n_hard > 5 else float("-inf")
        if float((ent > thr_g).float().mean()) > 0.60:
            thr_g = float("-inf")
        self.thr_g = thr_g

        # z-calibration on hard rows only: target-row score (+ z*cert) vs max1.
        pos_in_arch = torch.searchsorted(self.arch.contiguous(), tgt)
        if n_hard == 0:
            self.zp = 2.0
        else:
            qh = qe[hard]
            qskh = qh[:, :512] @ V.T
            qresh = (qh[:, :512] - qskh @ V).norm(dim=-1)
            pa = pos_in_arch[hard]
            tgt_side = self.side[pa]  # [n_hard, 64]
            tgt_csk = self.csk[pa]  # [n_hard, r]
            tgt_rho = self.rho[pa]  # [n_hard]
            idxs_t = (qh[:, 512:] * tgt_side).sum(-1) + (qskh * tgt_csk).sum(-1)
            idxs_t = idxs_t * sc_
            cert_t = qresh * tgt_rho * sc_ / (512 - self.r) ** 0.5
            m1h = max1[hard]
            self.zp = 8.0
            for z in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
                rec = float(((idxs_t + z * cert_t) > m1h).float().mean())
                if rec >= self.recall_target:
                    self.zp = z
                    break
        zp = self.zp
        self.built = True
        return {
            "zp": zp,
            "gate_off": thr_g == float("-inf"),
            "n_hard": n_hard,
            "arch": int(self.arch.numel()),
        }

    @torch.inference_mode()
    def query(self, qe: torch.Tensor) -> torch.Tensor:
        """qe: [H, 576] one decode step's expanded queries (float).
        Returns absolute pool indices of rows to fetch (possibly empty)."""
        sc_ = self.scale
        qe = qe.float()
        skept = (qe @ self.kept_rows.T) * sc_
        max1 = skept.max(-1).values  # [H]
        p1 = torch.softmax(skept, -1)
        ent = -(p1 * p1.clamp_min(1e-12).log()).sum(-1)
        gate = ent > self.thr_g  # [H]
        if not bool(gate.any()):
            return self.arch[:0]
        qsk = qe[:, :512] @ self.V.T
        qres = (qe[:, :512] - qsk @ self.V).norm(dim=-1)
        idxs = (qe[:, 512:] @ self.side.T + qsk @ self.csk.T) * sc_
        cert = (qres[:, None] * self.rho[None, :]) * sc_ / (512 - self.r) ** 0.5
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
