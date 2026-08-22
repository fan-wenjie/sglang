"""The span's short convolution against sglang's own, with a control that can fail.

`benchmark/afd/gdn_split.py` verifies the recurrence against the fused kernel, but it starts from
`mixed` -- the projection AFTER the convolution -- so `_convolve` had no coverage at all. It is
the span's own reimplementation of a causal depthwise filter, and with one token and an empty ring
its output is decided entirely by the LAST tap. Taking the wrong one would be wrong in the
simplest case there is, which is the case the arrangement is still failing.

The control reverses the taps on ONE side. The first version of this comparison reversed them on
both, which changes the same tap in each and agrees again -- a control that cannot fail, and the
second one written in this line of work.
"""

import unittest

import torch
from sglang.test.test_utils import CustomTestCase

from afd_tiny_stack import build_tiny_stack

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "the fixture's norms have no CPU kernel")
class TestTheSpansConvolutionIsTheModels(CustomTestCase):
    def setUp(self):
        from sglang.srt.afd.linear_state import LinearStates
        from sglang.srt.afd_query_shift.span import SpanRunner

        self.stack, self.config, kinds = build_tiny_stack(device="cuda")
        self.attn = self.stack.layers[0].linear_attn
        self.weight = self.attn.conv1d.weight.contiguous()
        self.channels, self.taps = self.weight.shape[0], self.weight.shape[-1]
        states = LinearStates(
            slots=4, num_v_heads=self.config.linear_num_value_heads,
            head_k_dim=self.config.linear_key_head_dim,
            head_v_dim=self.config.linear_value_head_dim, device=torch.device("cuda"))
        self.states = states
        self.runner = SpanRunner(self.stack, states, layer_types=kinds, query_shift=1)

    def span(self, x, weight, ids):
        self.states.conv_buffer(0, width=self.channels, taps=self.taps, dtype=x.dtype).zero_()
        saved = self.attn.conv1d.weight
        self.attn.conv1d.weight = torch.nn.Parameter(weight)
        try:
            return self.runner._convolve(self.attn, x.clone(), ids, 0).float()
        finally:
            self.attn.conv1d.weight = saved

    def sglang_one(self, x, weight):
        from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_update

        state = torch.zeros(1, self.channels, self.taps - 1, device="cuda", dtype=x.dtype)
        return causal_conv1d_update(
            x.clone(), state, weight.view(weight.shape[0], weight.shape[2]),
            self.attn.conv1d.bias, self.attn.activation).float()

    def sglang_chunk(self, x, weight):
        from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn

        out = causal_conv1d_fn(
            x.t().unsqueeze(0).contiguous(), weight.view(weight.shape[0], weight.shape[2]),
            self.attn.conv1d.bias, activation=self.attn.activation)
        return out[0].t().float()

    def relative(self, a, b):
        return float((a - b).norm() / (b.norm() + 1e-9))

    def test_one_token_against_an_empty_ring(self):
        x = torch.randn(1, self.channels, device="cuda", dtype=torch.bfloat16)
        self.assertLess(
            self.relative(self.span(x, self.weight, [3]), self.sglang_one(x, self.weight)), 0.02)

    def test_a_chunk_where_every_tap_participates(self):
        n = self.taps
        x = torch.randn(n, self.channels, device="cuda", dtype=torch.bfloat16)
        self.assertLess(
            self.relative(self.span(x, self.weight, [3] * n), self.sglang_chunk(x, self.weight)),
            0.02)

    def test_the_control_reverses_the_taps_on_one_side_only(self):
        """Without this the thresholds above are decoration: any tolerance passes a comparison
        that cannot tell a wrong filter from a right one."""
        x = torch.randn(1, self.channels, device="cuda", dtype=torch.bfloat16)
        wrong = self.weight.flip(-1).contiguous()
        self.assertGreater(
            self.relative(self.span(x, self.weight, [3]), self.sglang_one(x, wrong)), 0.5)

        n = self.taps
        y = torch.randn(n, self.channels, device="cuda", dtype=torch.bfloat16)
        self.assertGreater(
            self.relative(self.span(y, self.weight, [3] * n), self.sglang_chunk(y, wrong)), 0.5)

    def test_this_checkpoint_family_has_no_convolution_bias(self):
        """The span adds none. sglang passes `layer.bias` through, so if a sibling checkpoint ever
        carried one the span would silently drop it -- and the deployed Qwen3.8-27B has 48 conv1d
        tensors and not one bias among them."""
        self.assertIsNone(self.attn.conv1d.bias)


if __name__ == "__main__":
    unittest.main()
