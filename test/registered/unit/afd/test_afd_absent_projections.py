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

from sglang.srt.afd.absent_projections import strip_routed_weights
from sglang.test.test_utils import CustomTestCase


def a_layer(with_convolution=False):
    layer = torch.nn.Module()
    if with_convolution:
        # depthwise, as the model's is: `_convolution_of` names it by the path `release` resolves
        layer.linear_attn = torch.nn.Module()
        layer.linear_attn.conv1d = torch.nn.Conv1d(8, 8, 4, groups=8)
    layer.qkv_proj = torch.nn.Linear(8, 24, bias=False)
    layer.o_proj = torch.nn.Linear(8, 8, bias=False)
    layer.mlp = torch.nn.Linear(8, 32, bias=False)
    layer.post_attention_layernorm = torch.nn.LayerNorm(8)
    layer.attn = torch.nn.Module()  # the core: no parameters of its own
    return layer


def a_model(n=8, with_convolution=False):
    model = types.SimpleNamespace()
    model.model = types.SimpleNamespace(
        layers=[a_layer(with_convolution) for _ in range(n)]
    )
    return model


def a_routing(passengers, heads, history=None):
    """The routing as `strip_routed_weights` reads it.

    `history` is what decides whether the convolution ring is on this host, and therefore whether
    the weight that filters it may be released. It is read rather than a flag because the ring's
    presence is the fact that matters -- and the fixture must carry it, or the read raises an
    AttributeError that says nothing about any of this.
    """
    return types.SimpleNamespace(passengers=passengers, heads=heads, history=history)


def is_meta(module) -> bool:
    return all(p.is_meta for p in module.parameters())


class TestOnlyWhatTheRoutingSpeaksForIsReleased(CustomTestCase):
    def setUp(self):
        self.model = a_model()
        # one span: three passengers and the head that closes them, as the deployed shape is
        self.routing = types.SimpleNamespace(
            history=None,  # the real routing always has it; None means the ring is not here
            passengers={0, 1, 2, 4, 5, 6},
            heads={3, 7},
        )
        self.report = strip_routed_weights(
            self.model, self.routing, remote_embedding=False
        )

    def test_a_passenger_layer_goes_whole(self):
        """Its forward is a pass-through -- the pool runs all of it, so none of it is needed."""
        for index in (0, 1, 2, 4, 5, 6):
            self.assertTrue(
                is_meta(self.model.model.layers[index]), f"layer {index} still held"
            )

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
        routing rather than from a layer count, so a layer nobody speaks for keeps everything.
        """
        model, routing = a_model(), types.SimpleNamespace(
            history=None,  # the real routing always has it; None means the ring is not here
            passengers={0},
            heads=set(),
        )
        strip_routed_weights(model, routing, remote_embedding=False)
        self.assertTrue(is_meta(model.model.layers[0]))
        for index in range(1, 8):
            self.assertFalse(
                is_meta(model.model.layers[index]), f"layer {index} was not routed"
            )

    def test_the_report_counts_what_went(self):
        """Quoted in the host's log beside the arrangement, so a run that released nothing cannot
        read as one that released everything."""
        self.assertEqual(self.report["passenger_layers"], 6)
        self.assertEqual(self.report["head_layers"], 2)
        self.assertGreater(self.report["gib_freed"], 0.0)


if __name__ == "__main__":
    unittest.main()


class TestTheFilterStaysWhenTheRingIsOnThisHost(CustomTestCase):
    """A passenger layer runs entirely on the pool -- except its convolution, when the ring is here.

    Releasing the whole layer then takes the one weight this end still needs, and the failure is
    not an error: the convolution is applied against a freed tensor or skipped, and the answer is
    still fluent. 80 KiB a layer, 3.75 MiB for all 48, which is what makes holding it the design
    rather than a compromise.

    Both branches are pinned here because neither was. Every fixture in this file carried no
    history at all, so when the read moved from a flag to the history the five tests above went red
    with an `AttributeError` and this decision stayed unexercised in either direction.
    """

    def _strip_one_passenger(self, conv_weight):
        model = a_model(with_convolution=True)
        history = types.SimpleNamespace(conv_weight=conv_weight)
        strip_routed_weights(
            model, a_routing({0}, set(), history=history), remote_embedding=False
        )
        return model.model.layers[0]

    def test_a_ring_on_this_host_keeps_the_filter(self):
        layer = self._strip_one_passenger(
            conv_weight=lambda index: torch.zeros(8, 1, 4)
        )
        self.assertFalse(layer.linear_attn.conv1d.weight.is_meta)
        # and the rest of the layer still goes -- keeping one weight is not keeping the layer
        self.assertTrue(layer.mlp.weight.is_meta)

    def test_a_ring_on_the_pool_releases_it(self):
        layer = self._strip_one_passenger(conv_weight=None)
        self.assertTrue(layer.linear_attn.conv1d.weight.is_meta)

    def test_no_history_at_all_releases_it(self):
        """A host with no history service is a host with no ring."""
        model = a_model(with_convolution=True)
        strip_routed_weights(model, a_routing({0}, set()), remote_embedding=False)
        self.assertTrue(model.model.layers[0].linear_attn.conv1d.weight.is_meta)
