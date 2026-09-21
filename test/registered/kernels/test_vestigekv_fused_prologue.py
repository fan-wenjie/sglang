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


def _case(P=4, NKm=512, seed=0, empty=(), width=576):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(P, H, width, device=dev)
    kr = torch.randn(P, NKm, width, device=dev, dtype=torch.bfloat16)
    v = torch.stack(
        [
            torch.linalg.qr(torch.randn(KV, R, device=dev))[0].T.contiguous()
            for _ in range(P)
        ]
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
        self.assertTrue(
            bool(torch.isfinite(g1[1]).any() | (g1[1] == float("inf")).any())
        )



    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_rope_less_rows_match_reference(self):
        # width == KV: no sidecar columns, so the kernel's D-loop and the
        # transposed sidecar output must both handle DD == 0.
        from sglang.srt.layers.attention.vestigekv.fused_prologue import fused_prologue

        q, kr, v, nk_len, thr = _case(P=5, NKm=640, seed=3, width=KV)
        got = fused_prologue(q, kr, v, nk_len, thr, 1 / 16.0, kv=KV)
        ref = _reference(q, kr, v, nk_len, thr, 1 / 16.0)
        g1, gs, gk, gr = got
        r1, rs, rk, rr = ref
        self.assertEqual(tuple(gs.shape), (5, 0, H))
        self.assertTrue(torch.equal(gs, rs))
        self.assertTrue(torch.allclose(gk.float(), rk.float(), atol=2e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(gr, rr, atol=1e-3, rtol=1e-3))
        both_finite = torch.isfinite(g1) & torch.isfinite(r1)
        self.assertTrue(
            torch.allclose(g1[both_finite], r1[both_finite], atol=1e-4, rtol=1e-4)
        )



# ---------------------------------------------------------------------------
# The split form: a kept set held by more than one rank.
#
# max1g leaves the kernel with the margin subtracted, the entropy gain applied
# and a closed gate encoded as +inf, and the entropy is a property of the WHOLE
# kept set. So finished thresholds do not merge and the online (max, sum, t)
# accumulators do (docs/context-parallel.md). The last case below is the point:
# merging the finished values with a max produces a number that looks like a
# threshold and is not one.
# ---------------------------------------------------------------------------


class TestKeptStatsCombine(CustomTestCase):
    def _split(self, q, kr, v, nk_len, thr, sc, cut):
        """(mst per holder, fused max1g over the union)."""
        from sglang.srt.layers.attention.vestigekv.fused_prologue import fused_prologue

        P = kr.shape[0]
        parts, locals_ = [], []
        for lo, hi in ((0, cut), (cut, kr.shape[1])):
            n = (nk_len - lo).clamp(0, hi - lo)
            mst = q.new_zeros(P, H, 3)
            m1, _, _, _ = fused_prologue(
                q, kr[:, lo:hi].contiguous(), v, n, thr, sc, mst=mst
            )
            parts.append(mst)
            locals_.append(m1.clone())
        whole, _, _, _ = fused_prologue(q, kr, v, nk_len, thr, sc)
        return parts, locals_, whole

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_one_holder_round_trips(self):
        from sglang.srt.layers.attention.vestigekv.fused_prologue import (
            combine_kept_stats,
            fused_prologue,
        )

        q, kr, v, nk_len, thr = _case(P=5, NKm=512, seed=7)
        sc = 1 / 24.0
        mst = q.new_zeros(5, H, 3)
        m1, _, _, _ = fused_prologue(q, kr, v, nk_len, thr, sc, mst=mst)
        merged = combine_kept_stats([mst], thr[:, None], 0.0, False)
        self.assertTrue(torch.equal(torch.isfinite(merged), torch.isfinite(m1)))
        f = torch.isfinite(m1)
        self.assertTrue(torch.allclose(merged[f], m1[f], atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_two_holders_recover_the_whole_set(self):
        from sglang.srt.layers.attention.vestigekv.fused_prologue import (
            combine_kept_stats,
        )

        q, kr, v, nk_len, thr = _case(P=5, NKm=512, seed=8)
        sc = 1 / 24.0
        parts, _, whole = self._split(q, kr, v, nk_len, thr, sc, 256)
        merged = combine_kept_stats(parts, thr[:, None], 0.0, False)
        f = torch.isfinite(whole) & torch.isfinite(merged)
        self.assertTrue(torch.allclose(merged[f], whole[f], atol=1e-4, rtol=1e-4))
        gate_differs = torch.isfinite(merged) != torch.isfinite(whole)
        self.assertLess(int(gate_differs.sum()), max(4, whole.numel() // 32))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_maxing_the_finished_thresholds_closes_a_gate_that_should_be_open(self):
        """The mistake the triple exists to prevent, in the regime where it shows.

        With every gate open a max over per-holder max1g happens to be right:
        the max component does merge by max. What does not is the gate, and it
        fails in one direction -- half a kept set is a less flat distribution
        than the whole, so a holder's entropy sits below the union's. Choose a
        threshold inside that gap and both holders close while the union stays
        open; the max of two +inf is +inf, so tier 2 fires nothing where it
        should fire normally. Nothing downstream can tell that from a step
        where nothing deserved to fire.
        """
        from sglang.srt.layers.attention.vestigekv.fused_prologue import (
            combine_kept_stats,
        )

        q, kr, v, nk_len, _ = _case(P=5, NKm=512, seed=9)
        sc = 1 / 24.0
        nk_len = torch.full_like(nk_len, 512)
        # measured for this fixture: halves top out at 5.21 nats, the union
        # bottoms out at 5.56, and the gate is `entropy > thr`
        thr = torch.full((5,), 5.35, device=q.device)
        parts, locals_, whole = self._split(q, kr, v, nk_len, thr, sc, 256)
        merged = combine_kept_stats(parts, thr[:, None], 0.0, False)
        naive = torch.maximum(locals_[0], locals_[1])

        self.assertTrue(
            bool(torch.isfinite(whole).all()), "fixture drifted: union gates closed"
        )
        self.assertTrue(
            bool((naive == float("inf")).all()),
            "fixture drifted: the halves' gates did not close",
        )
        self.assertTrue(
            bool(torch.isfinite(merged).all()),
            "the merged triple must reproduce the union's open gate",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
