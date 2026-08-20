"""What happens on a model this wiring was not written for.

It installs by wrapping methods a model file happens to have. On another family those methods are
spelled differently or mean something else, and without a check the failure is the bad kind: an
AttributeError from three calls inside a wrapper, during a forward pass, naming one requirement
out of a contract nobody wrote down. Someone bringing a new model would discover the list one
exception at a time, in production.

These cases pin that the refusal comes first, comes whole, and names the model's own classes.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

from sglang.srt.afd.supported import ModelNotSupported, check_supported
from sglang.test.test_utils import CustomTestCase


def _attn():
    return types.SimpleNamespace(tp_q_head_num=24, tp_k_head_num=4, qk_head_dim=256,
                                 v_head_dim=256, scaling=0.0625)


def _named(name):
    """A layer whose CLASS NAME is what the classifier reads, as the real ones are."""
    return type(name, (object,), {})()


def _softmax_layer(name="FooAttentionDecoderLayer"):
    layer = _named(name)
    layer.input_layernorm = object()
    layer.layer_communicator = types.SimpleNamespace(prepare_mlp=lambda *a: None)
    layer.attn = _attn()
    for m in ("forward_prepare_cuda_fused", "forward_prepare_fused_gate",
              "forward_prepare_native", "forward_prepare_npu"):
        setattr(layer, m, lambda **k: None)
    return layer


def _linear_layer(name="FooLinearDecoderLayer", with_proj=True):
    layer = _named(name)
    layer.input_layernorm = object()
    layer.layer_communicator = types.SimpleNamespace(prepare_mlp=lambda *a: None)
    if with_proj:
        layer.linear_attn = types.SimpleNamespace(_forward_input_proj=lambda *a: None)
    return layer


def _model(layers):
    return types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))


class TestAStackThatProvidesEverythingPasses(CustomTestCase):
    def test_a_hybrid_stack_of_the_right_shape(self):
        model = _model([_linear_layer(), _linear_layer(), _linear_layer(), _softmax_layer()] * 2)
        summary = check_supported(model)
        self.assertEqual(summary, {"layers": 8, "full_attention": 2, "linear_attention": 6})

    def test_softmax_coverage_does_not_require_the_linear_projection(self):
        """That arm never touches those layers, so a stack whose linear layers are shaped
        differently can still serve it."""
        model = _model([_linear_layer(with_proj=False), _softmax_layer()])
        with self.assertRaises(ModelNotSupported):
            check_supported(model, coverage="all")
        check_supported(model, coverage="softmax")


class TestARefusalComesFirstAndComesWhole(CustomTestCase):
    def test_an_empty_stack_is_refused(self):
        """Converting nothing costs nothing, and a cost of zero reads as tolerance."""
        with self.assertRaises(ModelNotSupported) as caught:
            check_supported(_model([]))
        self.assertIn("nothing here to wrap", str(caught.exception))

    def test_a_layer_that_classifies_as_neither_is_named(self):
        model = _model([_softmax_layer(name="MambaBlock")])
        with self.assertRaises(ModelNotSupported) as caught:
            check_supported(model)
        self.assertIn("MambaBlock", str(caught.exception))

    def test_a_missing_prepare_variant_is_named_with_the_reason(self):
        """The query is projected by whichever variant the layer's own dispatch picks, and a
        missing one means that dispatch is not the one mirrored here."""
        layer = _softmax_layer()
        delattr(layer, "forward_prepare_npu")
        with self.assertRaises(ModelNotSupported) as caught:
            check_supported(_model([layer]))
        self.assertIn("forward_prepare_npu", str(caught.exception))

    def test_a_missing_geometry_field_is_named(self):
        layer = _softmax_layer()
        del layer.attn.scaling
        with self.assertRaises(ModelNotSupported) as caught:
            check_supported(_model([layer]))
        self.assertIn("scaling", str(caught.exception))

    def test_every_gap_is_reported_at_once_not_the_first(self):
        """Reporting one would be the same discovery process with extra steps."""
        a = _softmax_layer()
        delattr(a, "input_layernorm")
        b = _softmax_layer()
        delattr(b, "layer_communicator")
        c = _linear_layer(with_proj=False)
        with self.assertRaises(ModelNotSupported) as caught:
            check_supported(_model([a, b, c]))
        message = str(caught.exception)
        self.assertIn("input_layernorm", message)
        self.assertIn("prepare_mlp", message)
        self.assertIn("_forward_input_proj", message)

    def test_the_message_says_the_names_are_the_thing_to_map(self):
        """A family that spells these differently needs the names mapped, not the check relaxed,
        and the message has to say so or somebody will relax it."""
        with self.assertRaises(ModelNotSupported) as caught:
            check_supported(_model([_linear_layer(with_proj=False)]))
        self.assertIn("mapped, not this check relaxed", str(caught.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
