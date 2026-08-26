"""The ratio multiply IS the own norm -- held to the MODEL'S norm, not to a formula.

Two RMSNorms of the same residual share the same rms -- the only expensive part -- so this
layer's own reading is the reused reading times a per-channel ratio computed once at install.
The first version of this file checked that identity against its own hand-rolled rmsnorm and
stayed green while the deployment spoke garbage: the layers being served are GemmaRMSNorm,
which scales by `1 + weight`, and the wiring had passed the stored weights -- near -1 on the
real checkpoint -- where the effective scales belonged. So the anchor here is the actual
`GemmaRMSNorm` module the model runs, weights drawn to straddle -1 the way the checkpoint's
do, and the ratio is fed `1 + weight` exactly as the install-time wiring now does. The
refusal case pins the one condition under which the shortcut is not taken, because a ratio
through a near-zero scale amplifies whatever noise lives there.
"""

import unittest

import torch

from sglang.srt.afd_query_shift.pool_cook import norm_ratio, renorm_with_ratio
from sglang.test.test_utils import CustomTestCase


def _gemma_norm(dim, weight, eps=1e-6):
    from sglang.srt.layers.layernorm import GemmaRMSNorm

    norm = GemmaRMSNorm(dim, eps=eps)
    with torch.no_grad():
        norm.weight.copy_(weight)
    # forward_native is the module's reference arithmetic and runs anywhere; the CUDA
    # kernel is held to it by sglang's own tests
    return norm.forward_native


class TestTheRatioIsTheOwnNorm(CustomTestCase):
    def test_identity_against_the_models_own_norm(self):
        for rows, dim, seed in ((1, 5120, 11), (4, 5120, 12), (2, 256, 13)):
            torch.manual_seed(seed)
            x = torch.randn(rows, dim)
            # the checkpoint's regime: stored weights near -1, effective scales near zero
            w_reused = (
                torch.rand(dim) * 1.5 - 0.9
            )  # scales in (0.1, 1.6), off the floor
            w_own = torch.randn(dim) - 1.0
            reused = _gemma_norm(dim, w_reused)(x.clone())
            want = _gemma_norm(dim, w_own)(x.clone())
            ratio = norm_ratio(1.0 + w_own.float(), 1.0 + w_reused.float())
            got = renorm_with_ratio(reused, ratio)
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)

    def test_the_stored_weights_are_the_wrong_space(self):
        # the regression this file exists for: feeding the ratio the stored Gemma weights
        # (rather than 1 + weight) must NOT reproduce the own norm
        torch.manual_seed(11)
        dim = 512
        x = torch.randn(3, dim)
        w_reused = torch.rand(dim) * 1.5 - 0.9
        w_own = torch.randn(dim) - 1.0
        reused = _gemma_norm(dim, w_reused)(x.clone())
        want = _gemma_norm(dim, w_own)(x.clone())
        wrong = renorm_with_ratio(reused, norm_ratio(w_own, w_reused, floor=0.0))
        self.assertFalse(
            torch.allclose(wrong, want, rtol=1e-2, atol=1e-2),
            "the stored-weight ratio agreed with the own norm; this test no longer "
            "distinguishes the convention it was written to pin",
        )

    def test_the_identity_survives_any_shared_eps(self):
        # the ratio never touches the normaliser, so eps placement cannot break it
        torch.manual_seed(11)
        dim = 512
        x = torch.randn(3, dim)
        w_reused = torch.rand(dim) * 1.5 - 0.9
        w_own = torch.randn(dim) - 1.0
        for eps in (1e-6, 1e-2):
            reused = _gemma_norm(dim, w_reused, eps)(x.clone())
            want = _gemma_norm(dim, w_own, eps)(x.clone())
            ratio = norm_ratio(1.0 + w_own.float(), 1.0 + w_reused.float())
            got = renorm_with_ratio(reused, ratio)
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)

    def test_a_near_zero_scale_is_refused(self):
        scale_reused = torch.rand(64) + 0.5
        scale_reused[7] = 1e-5
        with self.assertRaisesRegex(ValueError, "channel"):
            norm_ratio(torch.randn(64) + 1.0, scale_reused)


if __name__ == "__main__":
    unittest.main()
