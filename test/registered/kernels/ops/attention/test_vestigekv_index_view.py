"""The index-k cache is paged with a split key and scale block, not interleaved.

`index_buf_accessor` is the layout's definition: GetK slices
``[: page_size * index_head_dim]`` and GetS slices
``[page_size * index_head_dim :]``, so within a page the keys come first and
the scales follow. A flat ``[n_slots, index_head_dim + 4]`` view does not give
that order, and this module took one for a while.

What it cost: tier 1 ranks rows by sigma over these keys and its top-k IS the
keep decision, so on a rope-less MLA the kept set was chosen from key bytes
read as scales. A written page's dequantised key norms came back inf where the
correct read gives about 16, and the "scale" read 6.2e7 against the true
constant 0.015625. Nothing in a fetch count or a latency shows that.

Size cannot catch it: a page of 8448 bytes is both 64 x 132 and
64 x 128 + 64 x 4, and the aiter accessor really does use the interleaved
view. So these build a buffer in the canonical layout and check the values,
not the shapes.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

PAGE, DIM = 64, 128


def _canonical_buffer(num_pages, keys_f32, scales):
    """[num_pages, PAGE*DIM + PAGE*4] uint8, keys block then scale block."""
    buf = torch.zeros(num_pages, PAGE * DIM + PAGE * 4, dtype=torch.uint8,
                      device="cuda")
    k8 = keys_f32.to(torch.float8_e4m3fn).view(torch.uint8)      # [P, PAGE, DIM]
    buf[:, : PAGE * DIM] = k8.reshape(num_pages, PAGE * DIM)
    buf[:, PAGE * DIM :] = scales.contiguous().view(torch.uint8).reshape(
        num_pages, PAGE * 4
    )
    return buf


class TestIndexViewLayout(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.P = 3
        # fp8-representable values so the round trip is exact
        self.keys = (
            torch.randint(-8, 9, (self.P, PAGE, DIM), device="cuda").float() / 8.0
        )
        self.scales = torch.full((self.P, PAGE), 0.015625, device="cuda")
        self.buf = _canonical_buffer(self.P, self.keys, self.scales)

    def test_rows_match_the_canonical_accessor_arithmetic(self):
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import index_rows

        slots = torch.tensor([0, 1, 63, 64, 65, 127, 128, 191], device="cuda")
        got = index_rows(
            self.buf, slots, index_head_dim=DIM, quant_block_size=DIM,
            slots_per_page=PAGE,
        )
        p, t = slots // PAGE, slots % PAGE
        want = self.keys[p, t] * self.scales[p, t][:, None]
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_the_interleaved_read_disagrees(self):
        """The bug, as a measurement: the old view is not merely different.

        If this ever passes, the two layouts have converged and the fix is
        moot -- which would be worth knowing.
        """
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import index_rows

        slots = torch.arange(PAGE * self.P, device="cuda")
        good = index_rows(
            self.buf, slots, index_head_dim=DIM, quant_block_size=DIM,
            slots_per_page=PAGE,
        )
        flat = self.buf.reshape(-1, DIM + 4)
        bad_k = flat.view(torch.float8_e4m3fn)[:, :DIM].float()
        bad_s = flat.view(torch.float32)[:, DIM // 4]
        bad = bad_k * bad_s[:, None]
        self.assertFalse(
            torch.allclose(good[: bad.shape[0]], bad[: good.shape[0]]),
            "the interleaved read agrees with the paged one; the layouts have "
            "converged and this guard is obsolete",
        )
        # The interleaved read's "scale" is whatever key bytes sit at that
        # offset. On the real pool those decode to 6.2e7; on a synthetic
        # buffer of small fp8 values they decode small, so the magnitude is
        # not the invariant -- that it is NOT the stored scale is.
        self.assertFalse(
            torch.allclose(bad_s, self.scales.reshape(-1)[: bad_s.numel()]),
            "the interleaved column happened to read the stored scale; this "
            "buffer cannot distinguish the layouts",
        )

    def test_a_wrong_page_size_is_refused_not_silently_read(self):
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import (
            index_page_views,
        )

        with self.assertRaises(AssertionError):
            index_page_views(
                self.buf, index_head_dim=DIM, quant_block_size=DIM, slots_per_page=32
            )

    def test_sigma_is_finite_on_a_canonical_buffer(self):
        """The symptom that was visible and unexamined: inf key norms."""
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import index_sigma

        slots = torch.arange(PAGE * self.P, device="cuda")
        sig = index_sigma(
            buf=self.buf, slots=slots, index_head_dim=DIM, quant_block_size=DIM,
            slots_per_page=PAGE, block=PAGE,
        )
        self.assertTrue(bool(torch.isfinite(sig).all()), f"sigma not finite: {sig}")
        # One sigma per TOKEN, not per block: "blockwise" is the transform
        # window. _advance_sigma concatenates these and takes topk over them
        # as the keep decision, which only type-checks per token.
        self.assertEqual(int(sig.numel()), PAGE * self.P)


if __name__ == "__main__":
    unittest.main(verbosity=2)
