"""Slicing the fused projection must compute the same q and k, to the bit.

`in_proj_qkvz` produces `[q | k | v | z]` in one matmul. The early view wants q and k -- 25% of
that output on this model -- and takes them by slicing the weight's rows rather than by computing
all four and discarding two. Measured, one row, this device: 112 us fused against 16 us sliced,
because the shape is bound by writing the output rather than by launch overhead.

An optimisation that changes the arithmetic is not an optimisation, and this one changes an input
to the query coefficient -- where being slightly wrong produces a model that still writes fluent
text. So the equality is asserted rather than assumed, against the fused call the model itself
would have made.

The convolution slices the same way, and the case for it is the same: depthwise means each channel
carries its own taps, so a prefix of the channels is the same arithmetic on those channels. The
control takes a prefix of the input while leaving the weight whole, which is the mistake the slice
is one typo away from.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.test.test_utils import CustomTestCase


class TestSlicingTheProjectionChangesNothing(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.hidden = 512
        self.key_dim, self.value_dim = 64, 192
        out = 2 * self.key_dim + 2 * self.value_dim
        self.weight = torch.randn(out, self.hidden)
        self.x = torch.randn(1, self.hidden)

    def test_the_slice_is_the_fused_calls_q_and_k(self):
        fused = torch.nn.functional.linear(self.x, self.weight)
        q_f, k_f, _v, _z = fused.split(
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim], dim=-1
        )
        sliced = torch.nn.functional.linear(self.x, self.weight[: 2 * self.key_dim])
        q_s, k_s = sliced.split([self.key_dim, self.key_dim], dim=-1)
        self.assertTrue(torch.equal(q_s, q_f), "the sliced q differs from the fused q")
        self.assertTrue(torch.equal(k_s, k_f), "the sliced k differs from the fused k")

    def test_the_slice_is_the_fused_calls_b(self):
        """`in_proj_ba` lays out `[b | a]` and the early path wants b alone: `a` feeds the decay,
        and the decay is taken from the current projection rather than this one. Same shape of
        slice as q|k, and same reason it is safe -- b is a PREFIX of the weight's rows.
        """
        heads = 8
        weight = torch.randn(2 * heads, self.hidden)
        fused = torch.nn.functional.linear(self.x, weight)
        b_f, _a = fused.split([heads, heads], dim=-1)
        b_s = torch.nn.functional.linear(self.x, weight[:heads])
        self.assertTrue(torch.equal(b_s, b_f), "the sliced b differs from the fused b")

    def test_the_write_strength_alone_is_what_gates_returns_for_it(self):
        """Two expressions for beta would drift. `write_strength` exists because the decay is
        five kernels this path discards, not because beta is computed differently here.
        """
        from sglang.srt.afd.linear_history import gates, write_strength

        b = torch.randn(3, 8)
        a = torch.randn(3, 8)
        A_log = torch.randn(8)
        dt_bias = torch.randn(8)
        _alpha, beta = gates(a, b, A_log, dt_bias)
        self.assertTrue(
            torch.equal(write_strength(b), beta),
            "write_strength and gates disagree about beta, so one of them is a second model",
        )

    def test_a_quantised_projection_is_refused_rather_than_sliced(self):
        """A quantised linear is not `weight @ x`. Slicing its rows computes something else and
        says nothing, and on this path that is a query coefficient nobody can see is wrong.
        """
        import types

        from sglang.srt.afd.span import _project_prefix

        class _AWQ:
            pass

        linear = types.SimpleNamespace(
            quant_method=_AWQ(), weight=torch.randn(16, self.hidden)
        )
        with self.assertRaises(NotImplementedError) as caught:
            _project_prefix(linear, self.x, 8, "the rest")
        self.assertIn("_AWQ", str(caught.exception))

    def test_a_view_needs_no_copy(self):
        """`weight[:n]` and `weight[:n].contiguous()` measured the same, so the copy would be
        bytes spent for nothing. Asserted so a later 'tidy-up' that adds one has to argue.
        """
        view = torch.nn.functional.linear(self.x, self.weight[: 2 * self.key_dim])
        copy = torch.nn.functional.linear(
            self.x, self.weight[: 2 * self.key_dim].contiguous()
        )
        self.assertTrue(torch.equal(view, copy))


class TestSlicingTheConvolutionChangesNothing(CustomTestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.channels, self.taps = 30, 4
        self.prefix = 12
        self.weight = torch.randn(self.channels, self.taps)
        self.ring = torch.randn(2, self.channels, self.taps)
        self.x = torch.randn(1, self.channels)

    def _convolve(self, ring, weight, x):
        window = torch.cat([ring[0][:, 1:], x.reshape(-1, 1)], dim=1)
        return torch.nn.functional.silu((weight * window).sum(dim=1))

    def test_a_channel_prefix_is_the_same_on_those_channels(self):
        whole = self._convolve(self.ring, self.weight, self.x)
        part = self._convolve(
            self.ring[:, : self.prefix],
            self.weight[: self.prefix],
            self.x[:, : self.prefix],
        )
        torch.testing.assert_close(part, whole[: self.prefix], rtol=0, atol=0)

    def test_slicing_the_input_but_not_the_weight_disagrees(self):
        """The control. Three prefixes of the same length keep a depthwise filter's channels
        lined up; two out of three filter one channel with another's taps."""
        whole = self._convolve(self.ring, self.weight, self.x)
        with self.assertRaises((RuntimeError, AssertionError)):
            wrong = self._convolve(
                self.ring[:, : self.prefix],
                self.weight,  # NOT sliced
                self.x[:, : self.prefix],
            )
            torch.testing.assert_close(wrong, whole[: self.prefix], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
