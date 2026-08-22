"""What the host releases once the group cut is installed, and what it must keep.

The measurement that made this necessary: a group-cut host held 19.18 GB of weights, the same as a
per-layer host, because the loader makes only the feed-forward absent. Everything the pool computes
was still allocated here -- a full set of attention projections nobody multiplies by anything.

Releasing them is easy to get one layer too far. The host still runs the softmax attention CORE
against its own KV cache, and it still applies the norms the span's bookkeeping calls on this side.
A meta tensor reaching either raises "expected all tensors to be on the same device" from inside a
kernel, which names neither the cut nor the module -- so what is kept is asserted here beside what
goes.
"""

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd_query_shift.absent_projections import strip_routed_weights
from sglang.test.test_utils import CustomTestCase


def a_layer():
    layer = torch.nn.Module()
    layer.qkv_proj = torch.nn.Linear(8, 24, bias=False)
    layer.o_proj = torch.nn.Linear(8, 8, bias=False)
    layer.mlp = torch.nn.Linear(8, 32, bias=False)
    layer.post_attention_layernorm = torch.nn.LayerNorm(8)
    layer.attn = torch.nn.Module()                 # the core: no parameters of its own
    return layer


def a_model(n=8):
    model = types.SimpleNamespace()
    model.model = types.SimpleNamespace(layers=[a_layer() for _ in range(n)])
    return model


def is_meta(module) -> bool:
    return all(p.is_meta for p in module.parameters())


class TestOnlyWhatTheRoutingSpeaksForIsReleased(CustomTestCase):
    def setUp(self):
        self.model = a_model()
        # one span: three passengers and the head that closes them, as the deployed shape is
        self.routing = types.SimpleNamespace(passengers={0, 1, 2, 4, 5, 6}, heads={3, 7})
        self.report = strip_routed_weights(self.model, self.routing)

    def test_a_passenger_layer_goes_whole(self):
        """Its forward is a pass-through -- the pool runs all of it, so none of it is needed."""
        for index in (0, 1, 2, 4, 5, 6):
            self.assertTrue(is_meta(self.model.model.layers[index]), f"layer {index} still held")

    def test_a_head_layer_loses_its_projections_and_its_feed_forward(self):
        """The query, key and value arrive already projected in the span's reply, and o_proj is
        applied on the pool by `run_epilogue`. None of the three is used here."""
        head = self.model.model.layers[3]
        for name in ("qkv_proj", "o_proj", "mlp"):
            self.assertTrue(is_meta(getattr(head, name)), f"head kept its {name}")

    def test_a_head_layer_keeps_its_norms(self):
        """The span's bookkeeping calls post_attention_layernorm on THIS side. A norm is a vector
        -- releasing it saves nothing and breaks the first token."""
        head = self.model.model.layers[3]
        self.assertFalse(is_meta(head.post_attention_layernorm))

    def test_an_unrouted_layer_is_untouched(self):
        """A half-installed cut must release exactly the half it routed. The list comes from the
        routing rather than from a layer count, so a layer nobody speaks for keeps everything."""
        model, routing = a_model(), types.SimpleNamespace(passengers={0}, heads=set())
        strip_routed_weights(model, routing)
        self.assertTrue(is_meta(model.model.layers[0]))
        for index in range(1, 8):
            self.assertFalse(is_meta(model.model.layers[index]), f"layer {index} was not routed")

    def test_the_report_counts_what_went(self):
        """Quoted in the host's log beside the arrangement, so a run that released nothing cannot
        read as one that released everything."""
        self.assertEqual(self.report["passenger_layers"], 6)
        self.assertEqual(self.report["head_layers"], 2)
        self.assertGreater(self.report["gib_freed"], 0.0)


if __name__ == "__main__":
    unittest.main()
