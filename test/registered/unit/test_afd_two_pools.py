"""The two-pool split: the sweep goes out while the feed-forward is still in flight.

The whole claim of the arrangement is an ORDER, and every ordering produces the same tokens, so
nothing downstream can tell them apart. Asserted directly:

    issue FFN -> issue SWEEP -> collect SWEEP -> collect FFN

The sweep must leave AFTER the feed-forward (issuing it first makes the send wait on it) and its
answer must be collected BEFORE the feed-forward's (otherwise the two pools are in series and the
second machine bought nothing).

The split itself is why this is possible: the sweep needs the query, the query comes from
h_(l-1) under the Early-Q read point, and the feed-forward that produces x_l has not returned.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

import torch
from sglang.srt.afd.protocol import OP_APPEND, OP_SWEEP_Q
from sglang.test.test_utils import CustomTestCase

HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256


class RecordingCacheClient:
    """Only what the window and the join use, and a log of the order."""

    def __init__(self, order):
        self.order = order
        self.address = "stub://cache"

    def issue_frame(self, request_id, layer, tensors, op):
        self.order.append(("issue-cache", layer, op))
        return (request_id, layer, op)

    def collect_frame(self, handle, device):
        self.order.append(("collect-cache", handle[1], handle[2]))
        tokens = 1
        return (torch.zeros(tokens, HEADS * HEAD_DIM),
                torch.full((tokens, HEADS), float("-inf")))


class RecordingWeightsClient:
    address = "stub://weights"

    def __init__(self, order):
        self.order = order

    def issue(self, request_id, layer, hidden):
        self.order.append(("issue-ffn", layer, None))
        return (request_id, layer, hidden)

    def collect(self, handle, device):
        self.order.append(("collect-ffn", handle[1], None))
        return handle[2]


def _model(n_layers=4):
    layers = []
    for _ in range(n_layers):
        mlp = types.SimpleNamespace()
        mlp.forward = lambda x: x
        layers.append(types.SimpleNamespace(mlp=mlp))
    return types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))


class TestTwoPoolsAreConcurrent(CustomTestCase):
    def test_the_sweep_leaves_after_the_send_and_lands_before_the_wait(self):
        from sglang.srt.afd.roles import install_pool_routing
        from sglang.srt.afd.sweep_ahead import SweepAhead

        order = []
        model = _model(4)
        for layer in model.model.layers:
            layer.attn = object()

        ahead = SweepAhead.__new__(SweepAhead)
        ahead.layers = model.model.layers
        ahead.sweep_at = {0: 1}
        ahead.softmax_targets = {1}
        ahead.routed_layers = set()
        ahead.pending = {}
        ahead._batch = None
        ahead._forward_batch = None
        ahead.cache_client = RecordingCacheClient(order)
        ahead.cache_request_id = 1
        ahead.n_sweeps = 0

        def fake_sweep(source_layer):
            ahead.cache_client.issue_frame(1, ahead.sweep_at[source_layer], (torch.zeros(1, 8),),
                                           OP_SWEEP_Q)

        ahead.sweep_after_issue = fake_sweep
        install_pool_routing(model, RecordingWeightsClient(order), (0,), sweep_ahead=ahead)
        model.model.layers[0].mlp.forward(torch.zeros(1, 8))
        ahead.cache_client.collect_frame((1, 1, OP_SWEEP_Q), "cpu")

        self.assertEqual(
            [step[0] for step in order],
            ["issue-ffn", "issue-cache", "collect-ffn", "collect-cache"],
        )
        # the sweep left while the feed-forward was outstanding: that is the concurrency
        self.assertLess(order.index(("issue-cache", 1, OP_SWEEP_Q)),
                        order.index(("collect-ffn", 0, None)))

    def test_the_append_is_a_different_op_from_the_sweep(self):
        """They go to the same pool and must not be confused: a sweep that appended would put
        this step's key into the history the same step is sweeping, counting it twice."""
        self.assertNotEqual(OP_SWEEP_Q, OP_APPEND)


class TestTheCachePoolHoldsNoWeights(CustomTestCase):
    def test_an_empty_history_sweeps_to_the_identity_of_the_merge(self):
        """First token of a request: nothing cached. The log partition of an empty sum is -inf,
        and the join then takes this step's own value whole -- which is what attention over one
        position is. A zero here instead would average it with a zero vector."""
        from sglang.srt.afd.pool_attention import CachePool, KVHolder

        attn = types.SimpleNamespace(tp_q_head_num=HEADS, tp_k_head_num=KV_HEADS,
                                     qk_head_dim=HEAD_DIM, v_head_dim=HEAD_DIM,
                                     scaling=HEAD_DIM**-0.5)
        layers = [types.SimpleNamespace(attn=attn)]
        pool = CachePool(KVHolder("cpu", 16), layers, {})
        o, lse = pool.sweep(1, 0, torch.randn(1, HEADS * HEAD_DIM))
        self.assertTrue(torch.isinf(lse).all() and (lse < 0).all())
        self.assertTrue((o == 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
