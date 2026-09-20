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

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-small")

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


class TestEntropyMargin(CustomTestCase):
    """The margin may grow with how flat the kept distribution is.

    A query with no dominant kept row is the one whose kept maximum says least,
    and it is the one that needs several archived rows rather than one, so
    --vestigekv-entropy-margin-gain lowers its threshold and leaves a confident
    query's alone. Gain 0 must reproduce the old threshold exactly, including
    on an empty kept set where lse and the maximum are both -inf and their
    difference is a NaN that multiplying by zero does not clear.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_gain_zero_is_the_old_threshold_and_gain_lowers_it(self):
        from sglang.srt.layers.attention.vestigekv.fused_prologue import fused_prologue

        q, kr, v, nk_len, thr = _case(P=3, NKm=256, seed=5)
        base, _, _, _ = fused_prologue(q, kr, v, nk_len, thr, 1 / 24.0)
        same, _, _, _ = fused_prologue(q, kr, v, nk_len, thr, 1 / 24.0, ent_gain=0.0)
        lower, _, _, _ = fused_prologue(q, kr, v, nk_len, thr, 1 / 24.0, ent_gain=0.5)
        self.assertTrue(torch.equal(base, same), "gain 0 must change nothing")
        fin = torch.isfinite(base) & torch.isfinite(lower)
        self.assertTrue(bool(fin.any()), "no open gate to compare")
        self.assertTrue(
            bool((lower[fin] <= base[fin] + 1e-6).all()),
            "a positive gain may only lower the threshold",
        )
        self.assertTrue(
            bool((lower[fin] < base[fin] - 1e-6).any()),
            "a positive gain must lower some threshold",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_empty_kept_stays_minus_inf_under_a_gain(self):
        from sglang.srt.layers.attention.vestigekv.fused_prologue import fused_prologue

        q, kr, v, nk_len, thr = _case(P=3, NKm=256, seed=2, empty=(0, 2))
        g, _, _, _ = fused_prologue(q, kr, v, nk_len, thr, 1 / 24.0, ent_gain=0.5)
        for p in (0, 2):
            self.assertTrue(
                bool((g[p] == -float("inf")).all()),
                f"empty kept pair {p} produced {g[p][:4]}",
            )


class TestSpreadTruncation(CustomTestCase):
    """An overflowing pair keeps the same number of rows, spread not prefixed.

    With the fallback off a pair that fires more than the buffer holds keeps
    its first W rows in position order, and the multi-key tasks lose 0.70 of a
    cell to that. SGLANG_DEBUG_VESTIGEKV_SPREAD_TRUNCATE keeps W rows spread
    over the archive instead, which is the falsifiable half of "the keys are
    spread and a prefix cuts the later ones". This pins that the switch keeps
    the count it reports, writes inside the buffer, and reaches the tail of the
    archive that the prefix never sees.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_spread_reaches_the_tail_and_reports_its_count(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.vestigekv.fused_prologue import compact_fired

        A, W = 4096, 64  # fires every row, so the buffer overflows 64-fold
        dev = "cuda"
        hit = torch.ones(A, dtype=torch.int32, device=dev)
        arch = torch.arange(A, dtype=torch.int32, device=dev)
        a_len = torch.tensor([A], dtype=torch.int32, device=dev)
        a_off = torch.zeros(1, dtype=torch.int64, device=dev)
        li = torch.zeros(1, dtype=torch.int32, device=dev)
        slot = torch.zeros(1, dtype=torch.int32, device=dev)
        got = {}
        for spread in (False, True):
            buf = torch.full((1, 1, W), -1, dtype=torch.int32, device=dev)
            ln = torch.zeros(1, 1, dtype=torch.int32, device=dev)
            ovf = torch.zeros(1, 1, dtype=torch.int32, device=dev)
            cnt = torch.zeros(1, dtype=torch.int32, device=dev)
            # counts arrive pre-filled by the scan; every row fires here
            nb = (A + 1023) // 1024
            scratch = (
                torch.full((1, nb), 1024, dtype=torch.int32, device=dev),
                torch.zeros(1, nb, dtype=torch.int32, device=dev),
                torch.zeros(1, dtype=torch.int32, device=dev),
            )
            with envs.SGLANG_DEBUG_VESTIGEKV_SPREAD_TRUNCATE.override(spread):
                compact_fired(
                    hit, arch, a_len, a_off, li, slot, buf, ln, ovf, cnt, scratch, A
                )
            torch.cuda.synchronize()
            n = int(ln[0, 0])
            got[spread] = (n, buf[0, 0, :n].clone())
            self.assertEqual(
                int(ovf[0, 0]), 1, "the pair must be flagged as overflowing"
            )
            self.assertLessEqual(n, W, f"reported count {n} exceeds the buffer")
            self.assertTrue(
                (buf[0, 0, :n] >= 0).all(), "reported count covers unwritten cells"
            )
        pre_max = int(got[False][1].max())
        spr_max = int(got[True][1].max())
        self.assertLess(
            pre_max, A // 2, "the prefix should not reach the archive's tail"
        )
        self.assertGreater(spr_max, A // 2, "the spread selection must reach the tail")


if __name__ == "__main__":
    unittest.main(verbosity=3)
