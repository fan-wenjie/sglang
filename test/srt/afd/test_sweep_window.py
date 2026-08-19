"""The window's order, which is the whole schedule and is invisible in any output.

A sweep launched in the wrong place still produces the right answer. Every arrangement below is
numerically identical -- the same query over the same cache, merged with the same token -- and only
one of them overlaps anything:

    issue, sweep, collect     the window. The host sends, the GPU sweeps, the host waits
    sweep, issue, collect     nothing hidden. `issue` copies the hidden states to the host, which
                              synchronises the stream, so the send waits for the sweep it was
                              supposed to run beside
    issue, collect, sweep     nothing hidden, and this is what the synchronous port did

So the order is asserted directly. A benchmark cannot tell these apart from a wrong answer, only
from a smaller number, and a smaller number has many other explanations.
"""

import types
import unittest

import torch


class RecordingClient:
    """Only what PoolRouting calls, and a log of when."""

    address = "stub://pool"

    def __init__(self, order):
        self.order = order

    def issue(self, request_id, layer, hidden):
        self.order.append(("issue", layer))
        return (request_id, layer, hidden)

    def collect(self, handle, device):
        self.order.append(("collect", handle[1]))
        return handle[2]


class RecordingSchedule:
    """A stand-in for SweepAhead with the two members the router touches."""

    def __init__(self, order, sweep_at):
        self.order = order
        self.sweep_at = sweep_at
        self.routed_layers = set()

    def sweep_after_issue(self, layer):
        self.order.append(("sweep", layer))


def _model(n_layers):
    layers = []
    for _ in range(n_layers):
        mlp = types.SimpleNamespace()
        mlp.forward = lambda x: x
        layers.append(types.SimpleNamespace(mlp=mlp))
    return types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))


class TestTheWindowOpensBetweenIssueAndCollect(unittest.TestCase):
    def test_the_sweep_runs_after_the_send_and_before_the_wait(self):
        from sglang.srt.afd.roles import install_pool_routing

        order = []
        model = _model(4)
        schedule = RecordingSchedule(order, {0: 1, 2: 3})
        install_pool_routing(model, RecordingClient(order), (0, 2), sweep_ahead=schedule)

        hidden = torch.zeros(2, 4)
        model.model.layers[0].mlp.forward(hidden)
        model.model.layers[2].mlp.forward(hidden)
        self.assertEqual(
            order,
            [("issue", 0), ("sweep", 0), ("collect", 0),
             ("issue", 2), ("sweep", 2), ("collect", 2)],
        )

    def test_a_router_without_a_schedule_still_issues_and_collects(self):
        """The synchronous arm has to keep working; it is the thing being compared against."""
        from sglang.srt.afd.roles import install_pool_routing

        order = []
        model = _model(2)
        install_pool_routing(model, RecordingClient(order), (0,))
        model.model.layers[0].mlp.forward(torch.zeros(2, 4))
        self.assertEqual(order, [("issue", 0), ("collect", 0)])

    def test_the_router_takes_over_the_trigger_for_the_layers_it_routes(self):
        """Without the handover the sweep fires twice: once from the prepare_mlp wrapper, before
        the issue, and once from the router. The first one closes the window -- the send waits on
        it -- and the second finds the sweep already pending and does nothing, so the schedule
        reports a full count of windows while having overlapped none of them."""
        from sglang.srt.afd.roles import install_pool_routing

        order = []
        model = _model(4)
        schedule = RecordingSchedule(order, {0: 1, 2: 3})
        install_pool_routing(model, RecordingClient(order), (0, 2), sweep_ahead=schedule)
        self.assertEqual(schedule.routed_layers, {0, 2})

    def test_an_unrouted_source_keeps_its_own_trigger(self):
        """A layer the pool does not run has no window to wait in, and its sweep must still
        happen -- otherwise the split silently stops applying wherever routing does."""
        from sglang.srt.afd.sweep_ahead import SweepAhead

        model = _model(4)
        for layer in model.model.layers:
            layer.attn = object()
        ahead = SweepAhead.__new__(SweepAhead)
        ahead.layers = model.model.layers
        ahead.sweep_at = {0: 1, 2: 3}
        ahead.routed_layers = {0}
        ahead.pending = {}
        ahead._batch = object()
        ahead._forward_batch = None
        fired = []
        ahead.sweep_after_issue = fired.append

        ahead.on_prepare_mlp(0, ahead._batch)
        ahead.on_prepare_mlp(2, ahead._batch)
        self.assertEqual(fired, [2], "layer 0 is routed, so the router triggers it")


class TestNoQuerySurvivesItsPass(unittest.TestCase):
    """A stashed query outliving its pass is the silent failure this schedule can produce.

    `pending` is consumed by the join, which raises if the partner is missing. The precomputed
    query is consumed by the projection wrapper, which has no way to tell last pass's query from
    this one's -- it would attend this batch's cache with a query projected from another batch's
    residual stream, and the output would read fluently.
    """

    def _schedule(self, model):
        from sglang.srt.afd.sweep_ahead import SweepAhead

        ahead = SweepAhead.__new__(SweepAhead)
        ahead.layers = model.model.layers
        ahead.sweep_at = {0: 1, 2: 3}
        ahead.routed_layers = set()
        ahead.pending = {}
        ahead._batch = None
        ahead._forward_batch = None
        ahead.sweep_after_issue = lambda layer: None
        return ahead

    def test_a_new_batch_clears_the_queries_of_the_last_one(self):
        model = _model(4)
        ahead = self._schedule(model)
        first = object()
        ahead.on_prepare_mlp(0, first)
        model.model.layers[1]._afd_q_precomputed = ("stale q", "stale gate", "m")
        model.model.layers[3]._afd_q_precomputed = ("stale q", "stale gate", "m")

        ahead.on_prepare_mlp(0, object())
        self.assertIsNone(model.model.layers[1]._afd_q_precomputed)
        self.assertIsNone(model.model.layers[3]._afd_q_precomputed)

    def test_the_same_batch_keeps_them(self):
        """The clear is a pass boundary, not a per-layer reset: layer j+N's query is stashed at
        layer j and read several layers later, with other sources' prepare_mlp in between."""
        model = _model(4)
        ahead = self._schedule(model)
        batch = object()
        ahead.on_prepare_mlp(0, batch)
        model.model.layers[1]._afd_q_precomputed = ("q", "gate", "m")
        ahead.on_prepare_mlp(2, batch)
        self.assertEqual(model.model.layers[1]._afd_q_precomputed, ("q", "gate", "m"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
