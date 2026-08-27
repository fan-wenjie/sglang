"""Which kind each layer is, on a family that names it and on one that does not.

`layer_kinds` decides from the BUILT model, never from a config, and it used to decide from the
decoder layer's class name alone. That is enough for a family which splits the two kinds into two
classes -- Qwen3.5 does -- and says nothing at all about one that builds a single decoder class
and decides by what it hangs on `self_attn`. `kimi_linear` is the second kind: 27 layers, all
`KimiDecoderLayer`, twenty of them holding `KimiDeltaAttention` and seven `DeepseekV2AttentionMLA`,
and neither of those names contains the word the old rule looked for.

Built from stand-ins rather than from a checkpoint: what the rule reads is the shape of the module
tree -- a short convolution, a fused projection, a `RadixAttention` inside -- and a stand-in
carries that shape exactly. A test that needed the real thing would need 91.5 GiB of weights and
would not run anywhere.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import torch.nn as nn

from sglang.test.test_utils import CustomTestCase

from sglang.srt.afd.layer_kinds import (
    attention_module,
    is_full_attention,
    layer_types_of,
    linear_widths,
)


def _bare(cls):
    """An instance of `cls` with no arguments: the tree's shape is what is under test."""
    made = cls.__new__(cls)
    nn.Module.__init__(made)
    return made


class _KimiDeltaAttention(nn.Module):
    """sglang's shape: the projections, the fused convolution, and the recurrent kernel."""

    def __init__(self):
        super().__init__()
        from sglang.srt.layers.radix_linear_attention import RadixLinearAttention

        self.qkv_proj = nn.Identity()
        self.qkv_conv1d = nn.Identity()
        self.attn = _bare(RadixLinearAttention)


class _KimiMLAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Identity()
        self.kv_b_proj = nn.Identity()
        from sglang.srt.layers.radix_attention import RadixAttention

        self.attn_mqa = _bare(RadixAttention)


class _KimiDecoderLayer(nn.Module):
    """One class for both kinds, which is the shape the old rule could not read."""

    def __init__(self, attn):
        super().__init__()
        self.self_attn = attn


class _SplitLinearDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = nn.Identity()


class _SplitAttentionDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Identity()


class _Stack(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layers)


class TestAFamilyThatNamesItsKinds(CustomTestCase):
    """The behaviour that was already there, pinned so the new rule cannot cost it."""

    def test_the_class_name_still_decides_where_it_says(self):
        stack = _Stack([_SplitLinearDecoderLayer(), _SplitAttentionDecoderLayer()])
        self.assertEqual(
            layer_types_of(stack), ["linear_attention", "full_attention"]
        )


class TestAFamilyThatDoesNot(CustomTestCase):
    def test_one_decoder_class_is_sorted_by_what_it_holds(self):
        stack = _Stack(
            [
                _KimiDecoderLayer(_KimiDeltaAttention()),
                _KimiDecoderLayer(_KimiDeltaAttention()),
                _KimiDecoderLayer(_KimiDeltaAttention()),
                _KimiDecoderLayer(_KimiMLAAttention()),
            ]
        )
        self.assertEqual(
            layer_types_of(stack),
            ["linear_attention"] * 3 + ["full_attention"],
            "the three-linear-one-attention pattern this checkpoint repeats",
        )

    def test_a_delta_block_is_not_sorted_by_its_name(self):
        # "KimiDeltaAttention" contains "Attention" and no "Linear". A rule reading the
        # attention's NAME would put a recurrent state where a cache sweep was expected, and
        # nothing downstream would say so. Nor by an attribute name: sglang calls the
        # convolution `qkv_conv1d` where the checkpoint's own file calls it `q_conv1d`.
        self.assertIn("Attention", _KimiDeltaAttention.__name__)
        self.assertNotIn("Linear", _KimiDeltaAttention.__name__)
        self.assertFalse(is_full_attention(_KimiDecoderLayer(_KimiDeltaAttention())))

    def test_the_accessor_answers_for_both_spellings(self):
        self.assertIsNotNone(attention_module(_SplitLinearDecoderLayer()))
        self.assertIsNotNone(attention_module(_SplitAttentionDecoderLayer()))
        self.assertIsInstance(
            attention_module(_KimiDecoderLayer(_KimiMLAAttention())), _KimiMLAAttention
        )

    def test_a_layer_of_neither_kind_is_refused_by_index(self):
        class _Unknown(nn.Module):
            pass

        stack = _Stack([_SplitAttentionDecoderLayer(), _Unknown()])
        with self.assertRaises(RuntimeError) as caught:
            layer_types_of(stack)
        self.assertIn("layer 1", str(caught.exception))


class TestTheWidthsAConvolutionIsSizedFrom(CustomTestCase):
    """`linear_widths` is what `installer._conv_width` measures the ring with, and a ring built
    to the wrong width is a host that reads garbage rather than a host that refuses."""

    def test_the_flat_fields_are_returned_unchanged(self):
        config = SimpleNamespace(
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        self.assertEqual(
            linear_widths(config),
            {"k_heads": 16, "v_heads": 32, "dk": 128, "dv": 128, "conv_taps": 4},
        )

    def test_the_normalised_accessor_answers_where_the_flat_fields_are_absent(self):
        config = SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(
                shape=SimpleNamespace(
                    num_k_heads=16,
                    num_heads=32,
                    head_k_dim=72,
                    head_dim=128,
                    conv_kernel=4,
                )
            )
        )
        self.assertEqual(
            linear_widths(config),
            {"k_heads": 16, "v_heads": 32, "dk": 72, "dv": 128, "conv_taps": 4},
        )

    def test_a_shape_that_does_not_separate_the_key_side_says_so_by_omission(self):
        # A shape carrying only `num_heads`/`head_dim` is stating that the two sides are the
        # same width, not that the key side is unknown.
        config = SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(
                shape=SimpleNamespace(num_heads=32, head_dim=128, conv_kernel=4)
            )
        )
        self.assertEqual(
            linear_widths(config),
            {"k_heads": 32, "v_heads": 32, "dk": 128, "dv": 128, "conv_taps": 4},
        )

    def test_a_config_that_states_neither_is_refused_rather_than_guessed(self):
        with self.assertRaises(AttributeError) as caught:
            linear_widths(SimpleNamespace())
        self.assertIn("mamba2_cache_params", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
