"""rung 2's two ends are constructed here, because a rung that cannot start measures nothing.

Both failures this guards were found by deploying: two processes on two machines, a model load
each, and a verdict script that then had nothing to read. Neither is subtle once seen and neither
was visible in 342 passing cases, because nothing had ever called these two functions.

    the pool end   `make_pool_runner` raised NameError on a helper whose import had been moved
                   into the other function. The pool died at startup, the host died at its first
                   token pointing at the pool, and the reading was "PoolClosed"
    the host end   the linear attention moved to the pool, but the CALLS BACK to its recurrent
                   state had no handler here. The pool waited thirty seconds and gave up with
                   "no state reading for request 4 layer 0" -- which reads as a wire problem

The rungs are an experiment rather than a feature, so these cases are about wiring and not about
arithmetic: whether both ends construct, whether the host answers what the pool will ask it, and
whether the rows it answers with are the rows of the layer in flight. What the rung COMPUTES is
checked against colocated by `benchmark/afd/rung_verdict.py`, which is the only comparison that
can see it.
"""

import types
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd_query_shift.rung2 import LinearOnPool, Rung2Arm
from sglang.test.test_utils import CustomTestCase

CONFIG = types.SimpleNamespace(
    linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
)


class FakeLinearDecoderLayer:
    """Named for the rule that decides a layer's kind -- `layer_kinds` reads the CLASS NAME."""

    def __init__(self):
        self.linear_attn = types.SimpleNamespace(forward=lambda h, **k: h * 3)
        self.mlp = types.SimpleNamespace(forward=lambda h: h)


class FakeAttentionDecoderLayer:
    def __init__(self):
        self.attn = object()
        self.mlp = types.SimpleNamespace(forward=lambda h: h)


def a_stack(groups=2):
    layers = []
    for _ in range(groups):
        layers += [FakeLinearDecoderLayer() for _ in range(3)]
        layers.append(FakeAttentionDecoderLayer())
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=layers), config=CONFIG,
        parameters=lambda: iter([torch.zeros(1)]),
    )
    return model


class StubClient:
    address = "stub://pool"

    def __init__(self):
        self.issued = []
        self.serve = None

    def require(self, *flags):
        pass

    def issue(self, request_id, layer, hidden):
        return (request_id, layer, hidden)

    def collect(self, handle, device):
        return handle[2]

    def issue_frame(self, request_id, layer, tensors, op):
        self.issued.append((request_id, layer, op))
        return types.SimpleNamespace(layer=layer, op=op)

    def collect_frame(self, handle, device):
        return (torch.zeros(2, 4),)


def _install(model, client):
    with mock.patch("sglang.srt.afd_query_shift.installer._span_slots", return_value=4):
        return Rung2Arm().install_on_host(model, client, sweep_ahead=None)


class TestBothEndsAreConstructible(CustomTestCase):
    def test_the_pool_end_builds_its_runner(self):
        """The NameError above. `make_pool_runner` is reached only in a spawned pool process, so
        the failure surfaced as a pool that would not bind and a host that could not connect."""
        with mock.patch("sglang.srt.afd_query_shift.installer._span_slots", return_value=2):
            runner = Rung2Arm().make_pool_runner(a_stack(), device="cpu")
        self.assertEqual(runner.query_shift, 0, "rung 2 does not move the read point")

    def test_the_host_installs_both_halves(self):
        """A rung is the one below it plus ONE change. Installing only the linear-attention move
        would leave the feed-forward on the host -- a third arrangement, neither rung 1 nor rung 2,
        measured under rung 2's name. The first version of this did exactly that."""
        model = a_stack()
        feed_forward, routing = _install(model, StubClient())
        self.assertIsNotNone(feed_forward, "the feed-forward offload is rung 0 and must be on")
        self.assertIsInstance(routing, LinearOnPool)


class TestTheHostAnswersWhatThePoolWillAsk(CustomTestCase):
    def test_a_state_reading_has_somewhere_to_land(self):
        """Moving the linear attention moves the calls to its recurrent state with it. With no
        history on this end the pool asks, waits its full timeout, and reports a wire problem."""
        from sglang.srt.afd.history_service import HistoryService

        client = StubClient()
        model = a_stack()
        _, routing = _install(model, client)
        self.assertIsNotNone(client.serve, "the pool has nobody to ask for the state")
        self.assertIsInstance(client.serve, HistoryService)

        # one state per layer of the stack, or a layer's reading would land in another's history
        self.assertEqual(client.serve.cache.state.shape[0], len(model.model.layers))

    def test_the_rows_answered_are_the_layer_in_flights(self):
        """Read through the routing, not captured when the first frame went out.

        A captured list would answer every later layer with the FIRST layer's rows. On a decode
        batch row, request and token coincide, so that is invisible there and wrong on a prompt --
        the same shape of mistake this tree has now made four times.
        """
        model = a_stack()
        _, routing = _install(model, StubClient())

        first = types.SimpleNamespace(req_pool_indices=[7], extend_seq_lens_cpu=[3])
        model.model.layers[0].linear_attn.forward(torch.zeros(3, 4), forward_batch=first)
        self.assertEqual(routing.current_rows(), [7, 7, 7])

        second = types.SimpleNamespace(req_pool_indices=[9], extend_seq_lens_cpu=[2])
        model.model.layers[1].linear_attn.forward(torch.zeros(2, 4), forward_batch=second)
        self.assertEqual(routing.current_rows(), [9, 9])


if __name__ == "__main__":
    unittest.main()
