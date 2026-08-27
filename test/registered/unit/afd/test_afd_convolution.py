"""The arrangement's short convolution against sglang's own, with a control that can fail.

`benchmark/afd/gdn_split.py` verifies the recurrence against the fused kernel, but it starts from
`mixed` -- the projection AFTER the convolution -- so the convolution itself had no coverage. It
is now `linear_history.convolve_with_ring`, one implementation serving whichever side of the wire
holds the ring, and with one token and an empty ring its output is decided entirely by the LAST
tap. Taking the wrong one would be wrong in the simplest case there is.

The control reverses the taps on ONE side. The first version of this comparison reversed them on
both, which changes the same tap in each and agrees again -- a control that cannot fail, and the
second one written in this line of work.

An earlier version of this file called `LinearRunner._convolve`, the pool-side reimplementation
that died with the ring-on-pool path; the file kept passing locally against a stale editable
install and failed everywhere else. It now imports the function it tests.
"""

import unittest

import torch

from sglang.srt.afd.linear_history import convolve_with_ring
from sglang.test.test_utils import CustomTestCase

CUDA = torch.cuda.is_available()

CHANNELS = 12
TAPS = 4


def _convolve(x, weight, ring):
    # the ring advances as a side effect; every caller passes a clone, so each case reads the
    # history it constructed rather than a neighbour's leftovers
    return convolve_with_ring(
        ring,
        x,
        weight,
        slots=[0],
        runs=[(7, 0, 1)],
    )


@unittest.skipUnless(CUDA, "compared against sglang's CUDA-only conv kernel")
class TestTheArrangementsConvolutionIsTheModels(CustomTestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.x = torch.randn(1, CHANNELS, device="cuda", dtype=torch.bfloat16)
        self.weight = torch.randn(CHANNELS, TAPS, device="cuda", dtype=torch.bfloat16)
        self.ring = torch.randn(1, CHANNELS, TAPS, device="cuda", dtype=torch.bfloat16)

    def sglang_one(self, x, weight, ring):
        from sglang.srt.layers.attention.mamba.causal_conv1d import (
            causal_conv1d_update,
        )

        # the ring holds TAPS entries and the window is [ring[1:], x]; the kernel's state is the
        # TAPS-1 entries of history, which is exactly ring[..., 1:]
        state = ring[..., 1:].clone()
        return causal_conv1d_update(
            x.clone(), state, weight, bias=None, activation="silu"
        )

    def test_one_token_against_an_empty_ring_is_the_last_tap(self):
        empty = torch.zeros_like(self.ring)
        out = _convolve(self.x, self.weight, empty.clone())
        want = torch.nn.functional.silu(self.weight[:, -1] * self.x)
        torch.testing.assert_close(out, want, rtol=2e-2, atol=2e-2)

    def test_a_step_with_history_matches_sglang(self):
        out = _convolve(self.x, self.weight, self.ring.clone())
        want = self.sglang_one(self.x, self.weight, self.ring)
        torch.testing.assert_close(out, want, rtol=2e-2, atol=2e-2)

    def test_the_control_reversing_one_sides_taps_disagrees(self):
        out = _convolve(self.x, self.weight.flip(-1), self.ring.clone())
        want = self.sglang_one(self.x, self.weight, self.ring)
        self.assertFalse(torch.allclose(out, want, rtol=1e-3, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
