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
    idxs = (qside_t.T @ side.T + qsk_t.T @ csk.T) * sc
    score = idxs + cc * (qres[:, None] * rho[None, :])
    return (score > max1g[:, None]).any(0).to(torch.int32)


class TestVestigeScanKernel(CustomTestCase):
    def _case(self, A, seed, closed_heads=(), quantile=0.999):
        from sglang.srt.layers.attention.vestigekv.scan_kernel import vestige_scan

        dev = "cuda"
        g = torch.Generator(device=dev).manual_seed(seed)
        rnd = lambda *s: torch.randn(*s, device=dev, generator=g)  # noqa: E731
        qside_t = rnd(D, H).contiguous()
        qsk_t = rnd(R, H).contiguous()
        qres = torch.rand(H, device=dev, generator=g) * 2
        side, csk = rnd(A, D), rnd(A, R)
        rho = torch.rand(A, device=dev, generator=g) * 2
        cc = 2.0 * SCALE / (512 - R) ** 0.5
        # place the threshold where a realistic fraction of rows fires
        score = (qside_t.T @ side.T + qsk_t.T @ csk.T) * SCALE + cc * (
            qres[:, None] * rho[None, :]
        )
        max1g = torch.full(
            (H,), torch.quantile(score.flatten().float(), quantile).item(), device=dev
        )
        for h in closed_heads:
            max1g[h] = float("inf")  # a closed gate folded into the threshold
        ref = _eager_fire(qside_t, qsk_t, qres, max1g, side, csk, rho, SCALE, cc)
        got = vestige_scan(qside_t, qsk_t, qres, max1g, side, csk, rho, SCALE, cc)
        return ref, got

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_matches_eager_across_archive_sizes(self):
        for A in (4096, 16384, 58900):
            ref, got = self._case(A, seed=A)
            self.assertEqual(int((ref != got).sum()), 0, f"A={A}")
            self.assertGreater(int(ref.sum()), 0, "threshold left nothing firing")

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
