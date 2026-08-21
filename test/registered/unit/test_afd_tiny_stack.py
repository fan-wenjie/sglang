"""The fixture the span will be checked against has the deployed model's shape.

This guards the fixture, not the span. It is worth its own file because the fixture is the thing
that failed last time: a fake decoder layer with a different shape than the real one let 234 cases
pass while the code they covered could not run at all. A reference that has drifted from the model
does not report that it has drifted -- it reports agreement.

Every assertion here is about a property the span's arithmetic depends on. If one of them changes
because the model changed, the split-exactness check built on this fixture is measuring something
else and should fail here first, loudly, rather than there, quietly.
"""

import unittest

import torch
from sglang.test.test_utils import CustomTestCase

from afd_tiny_stack import TINY, build_tiny_stack


class TestTheFixtureIsTheModelsOwnClasses(CustomTestCase):
    def setUp(self):
        self.stack, self.config, self.kinds = build_tiny_stack(device="cuda")

    def test_the_layers_are_the_models_own_decoder_classes(self):
        """Not a fake shaped like them. The class names are asserted because a fake that subclassed
        or duck-typed them is exactly what this fixture exists to stop being used."""
        names = [type(l).__name__ for l in self.stack.layers]
        self.assertEqual(
            names,
            ["Qwen3_5LinearDecoderLayer"] * 3
            + ["Qwen3_5AttentionDecoderLayer"]
            + ["Qwen3_5LinearDecoderLayer"] * 3
            + ["Qwen3_5AttentionDecoderLayer"],
        )

    def test_the_softmax_layer_is_the_last_of_its_group(self):
        """What makes a group a span. If the softmax layer were first, the span would run from an
        attention output to the SAME attention's next input and the cut would not be the one that
        was measured at 2046 us."""
        self.assertEqual(self.kinds,
                         ["linear_attention"] * 3 + ["full_attention"]
                         + ["linear_attention"] * 3 + ["full_attention"])
        self.assertEqual(self.config.full_attention_interval, 4)

    def test_the_projections_live_where_the_span_reaches_for_them(self):
        """`layer.self_attention` is a METHOD on these classes and not a submodule. The span takes
        `qkv_proj`, `o_proj` and `attn` off the decoder layer itself, and a rename upstream would
        make it reach through a method and get something that is not a module."""
        softmax = self.stack.layers[3]
        for name in ("qkv_proj", "o_proj", "attn", "mlp"):
            self.assertTrue(hasattr(softmax, name), f"softmax layer has no {name}")
        self.assertTrue(callable(getattr(softmax, "self_attention", None)))
        self.assertNotIsInstance(getattr(softmax, "self_attention", None), torch.nn.Module)

        linear = self.stack.layers[0]
        for name in ("linear_attn", "mlp"):
            self.assertTrue(hasattr(linear, name), f"linear layer has no {name}")
        self.assertFalse(hasattr(linear, "o_proj"))

    def test_the_linear_layer_fuses_its_projections(self):
        """One `in_proj_qkvz` splitting to [key, key, value, value], not four projections. The
        transformers implementation of the same model has four, and a port that assumed four here
        reads a slice of the wrong tensor and stays fluent."""
        linear = self.stack.layers[0].linear_attn
        self.assertTrue(hasattr(linear, "in_proj_qkvz"))
        self.assertFalse(hasattr(linear, "q_proj"))

    def test_the_convolution_is_depthwise_and_its_weight_is_taps_last(self):
        """Splicing a contiguous channel range at the projection's output is only safe because the
        convolution is grouped.

        The weight is (channels, 1, taps) -- THREE dimensions, with a singleton in the middle,
        which is what the `.squeeze(1)` at two call sites is for. This case was first written
        asserting 2-D, from a note that said the weight was "already (channels, taps)", and the
        fixture said otherwise on the first run. That is the fixture doing the job a fake could
        not: a hand-written conv would have had whatever shape the note claimed.
        """
        weight = self.stack.layers[0].linear_attn.conv1d.weight
        self.assertEqual(weight.dim(), 3, f"conv weight is {tuple(weight.shape)}")
        self.assertEqual(weight.shape[1], 1, "the middle axis is the one squeezed out")
        self.assertEqual(weight.shape[-1], TINY["linear_conv_kernel_dim"], "taps are last")

    def test_no_parameter_is_still_zero(self):
        """The constructor leaves them zero, and zero is not noise.

        sglang allocates with `torch.empty` and expects a loader; on a fresh CUDA allocation that
        is all-zero. 63 of this stack's 84 parameters came back zero, every projection among them,
        and a span run against that returns its input UNCHANGED -- which is the exact signature of
        the deployment bug this fixture exists to hunt. That reading was taken and nearly reported
        as a reproduction. With values drawn, the same span moves its input by 51%.

        So this is not a tautology about `copy_`. It is the guard on the difference between a
        fixture that can fail and one that agrees with anything.
        """
        zero = [n for n, p in self.stack.named_parameters() if float(p.float().abs().max()) == 0.0]
        self.assertEqual(zero, [], f"{len(zero)} parameter(s) never filled")

    def test_the_norms_are_not_scaled_away(self):
        """A norm weight near 0.02 like the projections would make every residual dwarf every
        contribution -- the identity again, by a different route and just as quiet."""
        for name, parameter in self.stack.named_parameters():
            if "norm" in name and name.endswith("weight"):
                self.assertGreater(float(parameter.float().abs().mean()), 0.5, name)

    def test_the_rope_is_mrope_with_three_axes(self):
        """positions is (3, tokens) on this model -- the rows are AXES, not tokens. Flattening it
        to a column gave 366 values where 122 were wanted, and the pool joins along the LAST axis
        because of it."""
        self.assertEqual(sum(TINY["rope_parameters"]["mrope_section"]), 4)
        self.assertTrue(TINY["rope_parameters"]["mrope_interleaved"])


if __name__ == "__main__":
    unittest.main()
