"""The host under the group cut: which layers still run, and what refuses when one that shouldn't.

Every case here guards a failure that produces fluent output. That is the whole difficulty of this
arrangement -- a layer running in the wrong place, a reply taken under the wrong key, a row sent
under the wrong request id -- none of them raise, and none of them show up in the text. They show
up in a throughput number that gets quoted as this arrangement's.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.protocol import OP_SPAN, OP_SPAN_ENTER, OP_SPAN_EXIT, OP_SPAN_Q
from sglang.srt.afd.span_routing import SpanClient, SpanRouting
from sglang.test.test_utils import CustomTestCase

TYPES = ["linear_attention"] * 3 + ["full_attention"] + \
        ["linear_attention"] * 3 + ["full_attention"]


class Recorder:
    """A pool client that records what was asked and answers with shaped noise."""

    def __init__(self):
        self.issued = []
        self.collected = []

    def issue_frame(self, request_id, layer, tensors, op):
        self.issued.append({"request_id": request_id, "layer": layer, "op": op,
                            "tensors": tensors})
        return SimpleNamespace(request_id=request_id, layer=layer, op=op,
                               _replace=lambda **k: SimpleNamespace(
                                   request_id=request_id, layer=layer, **k))

    def collect_frame(self, handle, device):
        self.collected.append((handle.layer, handle.op))
        return (torch.zeros(1, 4),)


def a_client():
    return SpanClient(Recorder(), reply_timeout_s=5.0)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, hidden_states, residual=None, *args, **kwargs):
        self.forward_calls += 1
        return hidden_states * 2, residual


class Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer() for _ in TYPES])


class TestThePassThroughs(CustomTestCase):
    """Three layers in four do not run here, and a layer that still does is caught.

    The install this mirrors went wrong once already in this tree, in the other direction: a
    construction hook patched one of sixteen loaders and nothing said so. What made it invisible
    was that a layer running where it should not still produces a number.
    """

    def setUp(self):
        self.stack = Stack()
        # only the pass-throughs are installed here; the heads need an attention this fake has not
        # got, and what is under test is which layers stop running
        self.routing = SpanRouting.__new__(SpanRouting)
        self.routing.model = self.stack
        self.routing._undo = []
        from sglang.srt.afd.span import group_layers

        self.routing.spans = group_layers(TYPES)
        for span in self.routing.spans:
            for layer_id in span[1:]:
                self.routing._make_pass_through(self.stack.model.layers[layer_id], layer_id)

    def test_a_passenger_layer_returns_its_input_untouched(self):
        x = torch.randn(2, 4)
        out, residual = self.stack.model.layers[0].forward(x, None)
        self.assertIs(out, x)
        self.assertIsNone(residual)
        self.assertEqual(self.stack.model.layers[0].forward_calls, 0)

    def test_the_head_layers_are_left_alone(self):
        x = torch.randn(2, 4)
        out, _ = self.stack.model.layers[3].forward(x, None)
        self.assertEqual(self.stack.model.layers[3].forward_calls, 1)
        torch.testing.assert_close(out, x * 2)

    def test_removing_the_install_puts_every_layer_back(self):
        self.routing.remove()
        x = torch.randn(2, 4)
        self.stack.model.layers[0].forward(x, None)
        self.assertEqual(self.stack.model.layers[0].forward_calls, 1)

    def test_a_layer_that_ran_in_between_is_refused(self):
        """The guard that says the install took, rather than assuming it."""
        self.routing._returned = torch.randn(1, 4)
        with self.assertRaises(RuntimeError) as caught:
            self.routing._check_untouched(7, torch.randn(1, 4))
        self.assertIn("pass-through install missed it", str(caught.exception))

    def test_the_same_tensor_passes_the_guard(self):
        held = torch.randn(1, 4)
        self.routing._returned = held
        self.routing._check_untouched(7, held)


class TestTheTwoHalvesAreTakenUnderDifferentKeys(CustomTestCase):
    """Both halves of a span's reply share a request and a group and are the same shape.

    Taken under one key the caller gets whichever landed first, and the query source and the span
    output are interchangeable as tensors -- so the attention would run against a hidden state
    from the wrong end of the span and nothing would say so. The opcode is what separates them.
    """

    def test_the_read_point_is_collected_under_its_own_opcode(self):
        client = a_client()
        handle = client.issue(3, torch.zeros(1, 4), torch.tensor([0]))
        client.collect_read_point(handle, "cpu")
        client.collect_output(handle, "cpu")
        self.assertEqual(client.client.collected, [(3, OP_SPAN_Q), (3, OP_SPAN)])

    def test_the_order_is_read_point_first(self):
        """It is the half that arrives early; collecting it second discards the head start."""
        client = a_client()
        handle = client.issue(3, torch.zeros(1, 4), torch.tensor([0]))
        client.collect(handle, "cpu")
        self.assertEqual(client.client.collected[0][1], OP_SPAN_Q)


class TestTheRowIdsTravel(CustomTestCase):
    """The pool keys both recurrent states by request; a row with no id advances the wrong one."""

    def test_a_row_count_mismatch_is_refused(self):
        client = a_client()
        with self.assertRaises(ValueError) as caught:
            client.issue(3, torch.zeros(4, 8), torch.tensor([0, 1]))
        self.assertIn("row id", str(caught.exception))

    def test_the_ids_ride_as_a_column_of_int64(self):
        client = a_client()
        client.issue(3, torch.zeros(2, 8), torch.tensor([5, 9]))
        ids = client.client.issued[0]["tensors"][1]
        self.assertEqual(tuple(ids.shape), (2, 1))
        self.assertEqual(ids.dtype, torch.int64)
        self.assertEqual(ids.reshape(-1).tolist(), [5, 9])

    def test_every_call_gets_a_fresh_request_id(self):
        """Frames are keyed by (request, layer, op); a reused id crosses two answers."""
        client = a_client()
        for _ in range(3):
            client.issue(3, torch.zeros(1, 4), torch.tensor([0]))
        ids = [c["request_id"] for c in client.client.issued]
        self.assertEqual(len(set(ids)), 3)


class TestTheOpcodesNameTheThreeShapes(CustomTestCase):
    """The ends of the stack are different arrangements, not special cases of the middle one."""

    def test_they_are_distinct(self):
        self.assertEqual(len({OP_SPAN, OP_SPAN_Q, OP_SPAN_ENTER, OP_SPAN_EXIT}), 4)

    def test_the_opcode_travels_with_the_frame(self):
        client = a_client()
        client.issue(0, torch.zeros(1, 4), torch.tensor([0]), OP_SPAN_ENTER)
        self.assertEqual(client.client.issued[0]["op"], OP_SPAN_ENTER)


if __name__ == "__main__":
    unittest.main()
