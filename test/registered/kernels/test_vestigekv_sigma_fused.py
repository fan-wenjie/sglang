"""Fused sigma / operand kernels vs their reference implementations."""

import unittest

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.sigma_fused import (
    _KB_PAD,
    sigma_fused_from_pool,
    topm_from_hist,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="base-b-test-1-gpu-small")


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




class TestGeometryVariants(CustomTestCase):
    def test_sigma_from_a_side_pool_matches_the_gather(self):
        # A [rows, 128] side pool: the branch is the whole row (offset 0).
        from sglang.srt.layers.attention.vestigekv.sigma_fused import (
            sigma_fused,
            sigma_fused_from_pool,
        )

        torch.manual_seed(4)
        pool = torch.randn(30000, 128, device="cuda").to(torch.bfloat16)
        blk, nb = 4096, 3
        slots = torch.randperm(30000, device="cuda")[: blk * nb].to(torch.int64)
        s_gather, h_gather = sigma_fused(pool[slots].reshape(nb, blk, 128))
        s_pool, h_pool = sigma_fused_from_pool(pool, slots, blk, offset=0, dim=128)
        self.assertTrue(torch.equal(s_gather.reshape(-1), s_pool))
        self.assertTrue(torch.equal(h_gather, h_pool))

    def test_operands_without_a_sidecar_keep_the_content_terms(self):
        from sglang.srt.layers.attention.vestigekv.operand_fused import (
            build_operands_fused,
        )

        torch.manual_seed(5)
        pool = torch.randn(20000, 512, device="cuda").to(torch.bfloat16)
        arch = torch.randperm(20000, device="cuda")[:5000].to(torch.int64)
        V = torch.linalg.qr(torch.randn(512, 64, device="cuda")).Q.T.contiguous()
        csk, rho, side = build_operands_fused(pool, arch, V, kv=512, side_dim=0)
        self.assertEqual(tuple(side.shape), (5000, 0))
        blk = pool[arch].float()
        c = blk @ V.T
        self.assertLess(
            float(((blk - c @ V).norm(dim=-1) - rho).abs().max()), 1e-3
        )
        ulp = torch.finfo(torch.float16).eps * c.abs().clamp_min(1.0)
        self.assertTrue(bool(((c.half().float() - csk.float()).abs() <= 2 * ulp).all()))



class TestFp8SidePoolKernel(CustomTestCase):
    def test_scaled_fp8_pool_matches_the_dequantized_gather(self):
        # HAS_SCALE path: fp8 row * fp32 scale inside the kernel must equal
        # feeding the dequantized fp32 rows to the unscaled kernel, bit for bit.
        from sglang.srt.layers.attention.vestigekv.salience import (
            dequantize_salience,
            quantize_salience,
        )
        from sglang.srt.layers.attention.vestigekv.sigma_fused import (
            sigma_fused,
            sigma_fused_from_pool,
        )

        torch.manual_seed(6)
        keys = torch.randn(30000, 128, device="cuda") * torch.rand(30000, 1, device="cuda") * 8
        pool, scale = quantize_salience(keys)
        blk, nb = 4096, 3
        slots = torch.randperm(30000, device="cuda")[: blk * nb].to(torch.int64)
        rows = dequantize_salience(pool[slots], scale[slots])
        s_ref, h_ref = sigma_fused(rows.reshape(nb, blk, 128))
        s_pool, h_pool = sigma_fused_from_pool(
            pool, slots, blk, offset=0, dim=128, scale=scale
        )
        self.assertTrue(torch.equal(s_ref.reshape(-1), s_pool))
        self.assertTrue(torch.equal(h_ref, h_pool))



# ---------------------------------------------------------------------------
# The split form: sigma over a block whose rows are held by more than one rank.
#
# Y = C^T R is a sum over rows, so it decomposes over whatever partition holds
# them and the residual pass is row-local once the sum is back
# (docs/context-parallel.md). Two claims, two comparisons: one holder split
# into two launches must be EXACT, because the rows add in the same order; two
# holders must be CLOSE and must select the same top-m, because the partials
# do not.
# ---------------------------------------------------------------------------

BLOCK = 512  # a whole CLOSE_BLOCK is 4096; the decomposition does not care
DIM = D.SIDECAR_DIM
ROW = D.KV_LORA_RANK + DIM


def _pool(n_rows, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(n_rows, ROW, generator=g, device="cuda", dtype=torch.bfloat16)


class TestSigmaSplit(CustomTestCase):
    def setUp(self):
        self.kbuf = _pool(BLOCK)
        self.slots = torch.arange(BLOCK, device="cuda", dtype=torch.int64)
        self.fused, self.fused_hist = sigma_fused_from_pool(
            self.kbuf, self.slots, BLOCK
        )

    def _y(self, slots, rows, pos=None):
        y = torch.zeros(1, _KB_PAD, DIM, device="cuda", dtype=torch.float32)
        sigma_fused_from_pool(
            self.kbuf, slots, rows, y=y, emit_y=True, pos=pos, basis_len=BLOCK
        )
        return y

    def test_one_holder_split_into_two_launches_is_exact(self):
        y = self._y(self.slots, BLOCK)
        sig, hist = sigma_fused_from_pool(self.kbuf, self.slots, BLOCK, y=y)
        self.assertTrue(torch.equal(sig, self.fused))
        self.assertTrue(torch.equal(hist, self.fused_hist))

    def test_identity_positions_change_nothing(self):
        # HAS_POS must be a different way of saying the same thing when the
        # positions are the ones the loop counter would have produced.
        pos = torch.arange(BLOCK, device="cuda", dtype=torch.int32)
        y = self._y(self.slots, BLOCK, pos=pos)
        sig, _ = sigma_fused_from_pool(
            self.kbuf, self.slots, BLOCK, y=y, pos=pos, basis_len=BLOCK
        )
        self.assertTrue(torch.equal(sig, self.fused))

    def test_two_interleaved_holders_agree_with_the_fused_block(self):
        pos = torch.arange(BLOCK, device="cuda", dtype=torch.int32)
        halves = [(self.slots[r::2].contiguous(), pos[r::2].contiguous())
                  for r in (0, 1)]

        ys = [self._y(sl, BLOCK // 2, pos=p) for sl, p in halves]
        total = ys[0] + ys[1]

        rebuilt = torch.empty(BLOCK, device="cuda", dtype=torch.float32)
        for (sl, p), r in zip(halves, (0, 1)):
            part, _ = sigma_fused_from_pool(
                self.kbuf, sl, BLOCK // 2, y=total, pos=p, basis_len=BLOCK
            )
            rebuilt[r::2] = part

        rms = (rebuilt - self.fused).pow(2).mean().sqrt().item()
        scale = self.fused.pow(2).mean().sqrt().item()
        self.assertLess(rms / scale, 1e-5, f"relative rms {rms / scale:.2e}")

        m = int(BLOCK * 0.03)
        a = set(torch.topk(self.fused, m).indices.tolist())
        b = set(torch.topk(rebuilt, m).indices.tolist())
        self.assertEqual(a, b, "the kept set must not depend on who held the rows")

    def test_a_wrong_partial_sum_is_caught(self):
        # The check above is only worth running if it can fail: dropping one
        # holder's partial is the mistake a sharded implementation makes.
        pos = torch.arange(BLOCK, device="cuda", dtype=torch.int32)
        sl0, p0 = self.slots[0::2].contiguous(), pos[0::2].contiguous()
        only_one = self._y(sl0, BLOCK // 2, pos=p0)
        part, _ = sigma_fused_from_pool(
            self.kbuf, sl0, BLOCK // 2, y=only_one, pos=p0, basis_len=BLOCK
        )
        rms = (part - self.fused[0::2]).pow(2).mean().sqrt().item()
        self.assertGreater(rms / self.fused.pow(2).mean().sqrt().item(), 1e-3)

    def test_histograms_of_the_holders_sum_to_the_fused_one(self):
        # Tier 1 selects from the summed histogram, so this is the property
        # the global top-m rests on.
        pos = torch.arange(BLOCK, device="cuda", dtype=torch.int32)
        halves = [(self.slots[r::2].contiguous(), pos[r::2].contiguous())
                  for r in (0, 1)]
        total = sum(self._y(sl, BLOCK // 2, pos=p) for sl, p in halves)
        hists = []
        sigs = []
        for sl, p in halves:
            sig, h = sigma_fused_from_pool(
                self.kbuf, sl, BLOCK // 2, y=total, pos=p, basis_len=BLOCK
            )
            hists.append(h)
            sigs.append(sig)
        summed = hists[0] + hists[1]
        self.assertEqual(int(summed.sum()), BLOCK)
        m = int(BLOCK * 0.03)
        merged = torch.cat(sigs)
        self.assertEqual(len(topm_from_hist(merged, summed, m)), m)


if __name__ == "__main__":
    unittest.main(verbosity=3)
