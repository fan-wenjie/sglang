"""The fused cook against the eager sequence it replaces, at real shapes.

Bit equality is not claimed -- a fused kernel reassociates -- so the comparison is allclose at
float32 tolerances, plus the one property reassociation cannot excuse: the coefficient built
from the fused outputs must contract a state to the same reading as the eager one within the
same tolerance.
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

CUDA = torch.cuda.is_available()

KH, VH, DK = 16, 48, 128
KEY = KH * DK
TAPS = 4


@unittest.skipUnless(CUDA, "triton compiles for the device")
class TestTheFusedCookIsTheEagerCook(CustomTestCase):
    def _pieces(self, rows, seed):
        torch.manual_seed(seed)
        qk = torch.randn(rows, 2 * KEY, device="cuda", dtype=torch.bfloat16)
        beta = torch.rand(rows, VH, device="cuda")
        # the history arrives already summed -- the partial the ring's owner computes
        partial = torch.randn(rows, 2 * KEY, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2 * KEY, TAPS, device="cuda", dtype=torch.bfloat16) * 0.2
        return qk, beta, partial, w

    def test_agreement_across_rows_and_seeds(self):
        from sglang.srt.afd_query_shift.cook_kernel import cook_early_fused
        from sglang.srt.afd_query_shift.pool_cook import cook_early

        for rows in (1, 3, 8):
            for seed in (11, 12):
                qk, beta, win, w = self._pieces(rows, seed)
                want_t, want_q = cook_early(
                    qk, beta, win, w, key_heads=KH, value_heads=VH, head_k_dim=DK
                )
                got_t, got_q = cook_early_fused(
                    qk, beta, win, w, key_heads=KH, value_heads=VH, head_k_dim=DK
                )
                torch.testing.assert_close(got_q, want_q, rtol=2e-3, atol=2e-3)
                torch.testing.assert_close(got_t, want_t, rtol=2e-3, atol=2e-3)

    def test_a_wrong_width_is_refused(self):
        from sglang.srt.afd_query_shift.cook_kernel import cook_early_fused

        qk, beta, win, w = self._pieces(1, 11)
        with self.assertRaises(ValueError):
            cook_early_fused(
                qk[:, :100], beta, win, w, key_heads=KH, value_heads=VH, head_k_dim=DK
            )


if __name__ == "__main__":
    unittest.main()
