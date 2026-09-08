"""Fused scan-prologue kernel: correctness against an upcast-fp32 reference.

The kernel replaces the pack's eager chain (skept/softmax/entropy/gate/qsk/
qres/max1g and the transpose-casts). Contract mirrors the scan kernel's:
exact bf16/fp16 products with fp32 accumulation, so only rows within float
rounding of a threshold may disagree with the reference.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, suite="base-b-test-1-gpu-small")

H, R, KV, DD = 32, 64, 512, 64


def _reference(q, kr, v, nk_len, thr, sc):
    P = kr.shape[0]
    max1g = q.new_empty(P, H)
    qres = q.new_empty(P, H)
    qsk_all = q.new_empty(P, H, R)
    for p in range(P):
        nk = int(nk_len[p])
        qe = q[p]
        qsk = qe[:, :KV] @ v[p].T
        qsk_all[p] = qsk
        qres[p] = (qe[:, :KV] - qsk @ v[p]).norm(dim=-1)
        if nk == 0:
            max1g[p] = -float("inf")
            continue
        skept = (qe.to(torch.bfloat16).float() @ kr[p, :nk].float().T) * sc
        m1 = skept.max(-1).values
        p1 = torch.softmax(skept, -1)
        ent = -(p1 * p1.clamp_min(1e-9).log()).sum(-1)
        gate = ent > float(thr[p])
        max1g[p] = torch.where(gate, m1, torch.tensor(float("inf"), device=q.device))
    qside_t = q[:, :, KV:].transpose(1, 2).to(torch.bfloat16)
    qsk_t = qsk_all.transpose(1, 2).half()
    return max1g, qside_t, qsk_t, qres


def _case(P=4, NKm=512, seed=0, empty=()):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(P, H, 576, device=dev)
    kr = torch.randn(P, NKm, 576, device=dev, dtype=torch.bfloat16)
    v = torch.stack(
        [torch.linalg.qr(torch.randn(KV, R, device=dev))[0].T.contiguous() for _ in range(P)]
    )
    nk_len = torch.randint(NKm // 2, NKm, (P,), device=dev, dtype=torch.int64)
    for i in empty:
        nk_len[i] = 0
    thr = torch.rand(P, device=dev) * 3
    return q, kr, v, nk_len, thr


class TestFusedPrologue(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_matches_reference(self):
        from sglang.srt.layers.attention.vestigekv.fused_prologue import fused_prologue

        q, kr, v, nk_len, thr = _case(P=6, NKm=768, seed=1)
        sc = 1 / 24.0
        got = fused_prologue(q, kr, v, nk_len, thr, sc)
        ref = _reference(q, kr, v, nk_len, thr, sc)
        g1, gs, gk, gr = got
        r1, rs, rk, rr = ref
        # query transposes: exact casts
        self.assertTrue(torch.equal(gs, rs))
        # qsk: same ieee GEMM, allow fp32-order noise then fp16 cast
        self.assertTrue(torch.allclose(gk.float(), rk.float(), atol=2e-3, rtol=1e-3))
        # qres via Pythagoras vs explicit residual: small relative tolerance
        self.assertTrue(torch.allclose(gr, rr, atol=1e-3, rtol=1e-3))
        # max1g: finite entries close; +/-inf pattern may differ only where the
        # entropy sits within rounding of the gate threshold
        both_finite = torch.isfinite(g1) & torch.isfinite(r1)
        self.assertTrue(
            torch.allclose(g1[both_finite], r1[both_finite], atol=1e-4, rtol=1e-4)
        )
        disagree = torch.isfinite(g1) != torch.isfinite(r1)
        self.assertLess(int(disagree.sum()), max(4, g1.numel() // 32))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_empty_kept_opens_gate(self):
        from sglang.srt.layers.attention.vestigekv.fused_prologue import fused_prologue

        q, kr, v, nk_len, thr = _case(P=3, NKm=256, seed=2, empty=(0, 2))
        g1, _, _, _ = fused_prologue(q, kr, v, nk_len, thr, 1 / 24.0)
        self.assertTrue(bool((g1[0] == -float("inf")).all()))
        self.assertTrue(bool((g1[2] == -float("inf")).all()))
        self.assertTrue(bool(torch.isfinite(g1[1]).any() | (g1[1] == float("inf")).any()))


if __name__ == "__main__":
    unittest.main(verbosity=3)
