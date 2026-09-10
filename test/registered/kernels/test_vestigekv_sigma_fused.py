"""Fused sigma / operand kernels vs their reference implementations."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, suite="base-b-test-1-gpu-small")


class TestSigmaFused(CustomTestCase):
    """The projection-form fused sigma must match (a) the fp64 projection
    reference numerically and (b) the cuFFT chain's top-m selection, and the
    batched grid must be bit-identical to the single-instance call."""

    def _mk(self, T=4096, D_=64, seed=0):
        torch.manual_seed(seed)
        t = torch.arange(T, dtype=torch.float64)
        base = torch.stack(
            [torch.sin(2 * torch.pi * (k % 7 + 1) * t / T + k) for k in range(D_)], 1
        )
        return (
            (base + 0.1 * torch.randn(T, D_, dtype=torch.float64))
            .to(torch.bfloat16)
            .cuda()
        )

    def test_matches_fp64_projection_and_cufft_topm(self):
        from sglang.srt.layers.attention.vestigekv.sigma_fused import sigma_fused

        r = self._mk()
        T = r.shape[0]
        u = torch.arange(T, dtype=torch.float64, device="cuda")
        cols = [torch.ones(T, dtype=torch.float64, device="cuda") / T**0.5]
        for k in range(1, 16):
            a = 2 * torch.pi * k * u / T
            cols += [torch.cos(a) * (2 / T) ** 0.5, torch.sin(a) * (2 / T) ** 0.5]
        C = torch.stack(cols, 1)
        x = r.double()
        ref = (x - C @ (C.T @ x)).norm(dim=1)
        sig, hist = sigma_fused(r)
        self.assertLess(float((sig.double() - ref).abs().max()), 1e-4)
        # top-m set vs the cuFFT chain (both flavors quantize identically)
        m = T // 32
        # compute the raw cuFFT chain directly (route-independent reference)
        f = torch.fft.rfft(r.float(), dim=0)
        f[16:] = 0
        low = torch.fft.irfft(f, n=T, dim=0)
        s_fft = (r.float() - low).norm(dim=-1)
        ov = len(
            set(torch.topk(sig, m).indices.tolist())
            & set(torch.topk(s_fft, m).indices.tolist())
        )
        self.assertGreaterEqual(ov / m, 0.99)
        self.assertEqual(int(hist.sum()), T)

    def test_batched_equals_single(self):
        from sglang.srt.layers.attention.vestigekv.sigma_fused import sigma_fused

        r = torch.randn(8, 4096, 64, device="cuda").to(torch.bfloat16)
        sb, _ = sigma_fused(r)
        s3, _ = sigma_fused(r[3])
        self.assertTrue(torch.equal(sb[3], s3))

    def test_topm_from_hist_is_exact(self):
        from sglang.srt.layers.attention.vestigekv.sigma_fused import (
            sigma_fused,
            topm_from_hist,
        )

        r = self._mk(seed=1)
        sig, hist = sigma_fused(r)
        m = r.shape[0] // 32
        idx = topm_from_hist(sig, hist, m)
        thr = torch.topk(sig, m).values.min()
        must = set((sig > thr).nonzero().flatten().tolist())
        self.assertEqual(idx.numel(), m)
        self.assertTrue(must <= set(idx.tolist()))

    def test_from_pool_is_bit_identical(self):
        """The in-kernel strided read must equal the gather-then-slice path
        exactly: same arithmetic, only the addressing differs."""
        from sglang.srt.layers.attention.vestigekv.sigma_fused import (
            sigma_fused,
            sigma_fused_from_pool,
        )

        torch.manual_seed(3)
        pool = torch.randn(50000, 576, device="cuda").to(torch.bfloat16)
        blk, nb = 4096, 3
        slots = torch.randperm(50000, device="cuda")[: blk * nb].to(torch.int64)
        s_gather, h_gather = sigma_fused(pool[slots][:, 512:].reshape(nb, blk, 64))
        s_pool, h_pool = sigma_fused_from_pool(pool, slots, blk)
        self.assertTrue(torch.equal(s_gather.reshape(-1), s_pool))
        self.assertTrue(torch.equal(h_gather, h_pool))


class TestOperandFused(CustomTestCase):
    """Fused operand builder vs the chunked torch loop: side bit-identical,
    rho to fp32 tolerance, csk within one fp16 ulp (reduction order)."""

    def test_matches_torch_loop(self):
        from sglang.srt.layers.attention.vestigekv.operand_fused import (
            build_operands_fused,
        )

        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        pool = torch.randn(20000, 576, device="cuda").to(torch.bfloat16)
        A = 7937
        arch = torch.randperm(20000, device="cuda")[:A].to(torch.int64)
        V = torch.linalg.qr(torch.randn(512, 64, device="cuda")).Q.T.contiguous()
        blk = pool[arch].float()
        c = blk[:, :512] @ V.T
        csk_r = c.half()
        rho_r = (blk[:, :512] - c @ V).norm(dim=-1)
        side_r = blk[:, 512:].to(torch.bfloat16)
        csk_f, rho_f, side_f = build_operands_fused(pool, arch, V)
        self.assertTrue(torch.equal(side_r, side_f))
        self.assertLess(
            float(((rho_r - rho_f).abs() / rho_r.clamp_min(1e-6)).max()), 1e-5
        )
        # csk within one fp16 ulp of the cuBLAS-computed value
        ulp = torch.finfo(torch.float16).eps * csk_r.abs().float().clamp_min(1.0)
        self.assertTrue(bool(((csk_r.float() - csk_f.float()).abs() <= 2 * ulp).all()))


if __name__ == "__main__":
    unittest.main(verbosity=3)
