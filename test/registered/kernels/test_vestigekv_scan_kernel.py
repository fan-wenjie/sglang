"""The fused tier-2 archive scan must select exactly the rows the eager
formulation selects.

The kernel decides which archive rows the model attends to, and it reduces over
heads inside the kernel, so a disagreement never shows up as a wrong number --
only as a different set of attended rows, which reads downstream as a slightly
worse answer. These tests pin the fire set against the eager reference on the
serving shapes, including the two ways a row can be excluded (a losing score and
a closed gate).
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, suite="base-b-test-1-gpu-small")

H, D, R = 32, 64, 64
SCALE = 192**-0.5


def _eager_fire(qside_t, qsk_t, qres, max1g, side, csk, rho, sc, cc):
    # Reference at full fp32: upcast BEFORE the matmul. A native fp16 matmul
    # here would use torch's reduced-precision fp16 reduction (~1e-3 error)
    # and the REFERENCE, not the kernel, becomes the inaccurate side --
    # measured: two rows flipped by the eager path that fp64 says must not
    # fire, while the kernel agreed with fp64. Products of bf16/fp16 inputs
    # are exact in fp32, so this reference differs from the kernel only by
    # fp32 accumulation order (~1e-6), covered by the boundary tolerance.
    idxs = qside_t.T.float() @ side.T.float() + qsk_t.T.float() @ csk.T.float()
    score = idxs * sc + cc * (qres[:, None] * rho[None, :])
    return (score > max1g[:, None]).any(0).to(torch.int32)


class TestVestigeScanKernel(CustomTestCase):
    def _case(self, A, seed, closed_heads=(), quantile=0.999, want_extras=False):
        from sglang.srt.layers.attention.vestigekv.scan_kernel import vestige_scan

        dev = "cuda"
        g = torch.Generator(device=dev).manual_seed(seed)
        rnd = lambda *s: torch.randn(*s, device=dev, generator=g)  # noqa: E731
        qside_t = rnd(D, H).contiguous().to(torch.bfloat16)
        qsk_t = rnd(R, H).contiguous().half()
        qres = torch.rand(H, device=dev, generator=g) * 2
        side, csk = rnd(A, D).to(torch.bfloat16), rnd(A, R).half()
        rho = torch.rand(A, device=dev, generator=g) * 2
        cc = 2.0 * SCALE / (512 - R) ** 0.5
        # place the threshold where a realistic fraction of rows fires
        score = (
            qside_t.T.float() @ side.T.float() + qsk_t.T.float() @ csk.T.float()
        ) * SCALE + cc * (qres[:, None] * rho[None, :])
        max1g = torch.full(
            (H,), torch.quantile(score.flatten().float(), quantile).item(), device=dev
        )
        for h in closed_heads:
            max1g[h] = float("inf")  # a closed gate folded into the threshold
        ref = _eager_fire(qside_t, qsk_t, qres, max1g, side, csk, rho, SCALE, cc)
        got = vestige_scan(qside_t, qsk_t, qres, max1g, side, csk, rho, SCALE, cc)
        if want_extras:
            full = (
                qside_t.T.float() @ side.T.float() + qsk_t.T.float() @ csk.T.float()
            ) * SCALE + cc * (qres[:, None] * rho[None, :])
            return ref, got, (full, max1g)
        return ref, got

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_matches_eager_across_archive_sizes(self):
        # Native tensor-core dots have exact products but a different fp32
        # accumulation ORDER than torch's eager matmul, so single rows whose
        # score sits within float rounding of the threshold may flip. The
        # contract is therefore: any disagreement must be a boundary row
        # (score within eps of its head threshold), and there may be only a
        # handful. Systematic errors (tf32-style truncation) would flip rows
        # far from the boundary and fail this.
        from sglang.srt.layers.attention.vestigekv.scan_kernel import (  # noqa: F401
            vestige_scan,
        )

        for A in (4096, 16384, 58900):
            ref, got, boundary = self._case_with_boundary(A, seed=A)
            dis = (ref != got).nonzero().flatten()
            self.assertLessEqual(int(dis.numel()), max(4, A // 4096), f"A={A}")
            for i in dis.tolist():
                self.assertTrue(
                    bool(boundary[i]), f"A={A} non-boundary row {i} flipped"
                )
            self.assertGreater(int(ref.sum()), 0, "threshold left nothing firing")

    def _case_with_boundary(self, A, seed, closed_heads=(), quantile=0.999):
        ref, got, extras = self._case(A, seed, closed_heads, quantile, want_extras=True)
        score, max1g = extras
        margin = (score - max1g[:, None]).abs().min(0).values
        eps = 1e-5 * (1 + score.abs().max())
        return ref, got, margin < eps

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_closed_gates_never_fire(self):
        ref, got = self._case(16384, seed=1, closed_heads=range(0, H, 3))
        self.assertEqual(int((ref != got).sum()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_all_gates_closed_fires_nothing(self):
        ref, got = self._case(8192, seed=2, closed_heads=range(H))
        self.assertEqual(int(got.sum()), 0)
        self.assertEqual(int(ref.sum()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_a_permissive_threshold_fires_everything(self):
        # the guard-fires case: if the kernel silently dropped rows, a
        # threshold below every score would still show it
        ref, got = self._case(8192, seed=3, quantile=0.0)
        self.assertEqual(int((ref != got).sum()), 0)
        self.assertGreater(int(got.sum()), 8192 * 0.99)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_ragged_archive_size_is_masked(self):
        # A not a multiple of the block: the tail block must not write past A
        for A in (64 * 7 + 1, 64 * 7 + 63):
            ref, got = self._case(A, seed=A)
            self.assertEqual(int((ref != got).sum()), 0, f"A={A}")


if __name__ == "__main__":
    unittest.main()


class TestOperandReuse(CustomTestCase):
    """Calibrated rebuilds adopt the previous tier's scan operands: they are
    pure functions of (prefix rows, basis, keep mask) and independent of the
    calibration queries, so adoption must be BIT-exact -- operands and the
    calibration scalars alike. Audit rule: the changed-row-set guard is also
    driven with a known-bad input and shown to refuse."""

    def _mk(self, T=4096, H=8, n_cal=16, seed=0):
        import torch

        from sglang.srt.layers.attention.vestigekv import defaults as D

        torch.manual_seed(seed)
        dev = "cuda"
        kbuf = torch.randn(T + 64, D.LATENT_DIM, device=dev, dtype=torch.bfloat16)
        row_slots = torch.arange(T, device=dev, dtype=torch.int64)
        keep = torch.zeros(T, dtype=torch.bool, device=dev)
        keep[::32] = True
        keep[-256:] = True
        q_cal = torch.randn(n_cal, H, D.LATENT_DIM, device=dev, dtype=torch.float32)
        q_pos = torch.arange(T - n_cal, T, device=dev, dtype=torch.long)
        return kbuf, row_slots, keep, q_cal, q_pos

    def test_adopted_build_is_bit_identical(self):
        import torch

        from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

        kbuf, row_slots, keep, q_cal, q_pos = self._mk()
        prov = RecallTier(r=64, topj=-1)
        prov.build(kbuf, row_slots, keep, q_cal[:2], q_pos[:2], conservative=True)
        fresh = RecallTier(r=64, topj=-1)
        s_f = fresh.build(
            kbuf, row_slots, keep, q_cal, q_pos, conservative=False, v_init=prov.V
        )
        adopt = RecallTier(r=64, topj=-1)
        s_a = adopt.build(
            kbuf,
            row_slots,
            keep,
            q_cal,
            q_pos,
            conservative=False,
            v_init=prov.V,
            operands_from=prov,
        )
        for name in ("csk", "rho", "side", "V", "kept_rows"):
            self.assertTrue(
                torch.equal(getattr(fresh, name), getattr(adopt, name)), name
            )
        self.assertEqual(s_f["zp"], s_a["zp"])
        self.assertEqual(fresh.thr_g, adopt.thr_g)
        self.assertEqual(fresh.zp, adopt.zp)

    def test_changed_row_set_is_refused(self):
        from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

        kbuf, row_slots, keep, q_cal, q_pos = self._mk()
        prov = RecallTier(r=64, topj=-1)
        prov.build(kbuf, row_slots, keep, q_cal[:2], q_pos[:2], conservative=True)
        keep2 = keep.clone()
        keep2[1] = ~keep2[1]  # one row moves tier -> different archive
        bad = RecallTier(r=64, topj=-1)
        with self.assertRaises(AssertionError):
            bad.build(
                kbuf,
                row_slots,
                keep2,
                q_cal,
                q_pos,
                conservative=False,
                v_init=prov.V,
                operands_from=prov,
            )
