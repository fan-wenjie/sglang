"""The stop merges what is safe to merge, relays the rest, and gives the pool one caller.

Why it exists at all is a measurement rather than a design preference: the pool does 2224 calls/s
reached by two connections and 1197 by eight, with every phase inflating tenfold and a matmul on an
idle GPU taking ten times longer. Threads waiting their turn, not work. So sixteen hosts must
arrive as one connection.

The dangerous half is the merge. Rows from two hosts may ride one upstream call only when they are
independent of each other, which is true of a feed-forward and false of anything whose departure
calls BACK to its caller -- a span or a layer carries per-request state on the return path, and
merging two of those would send one host's callback to another host. That failure would be silent:
both hosts get a tensor of the right shape.
"""

import threading
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.dispatcher import Stop
from sglang.srt.afd.protocol import OP_FFN, OP_SPAN, Frame
from sglang.test.test_utils import CustomTestCase


class Upstream:
    """A pool that records what it was asked, and answers rows for rows."""

    def __init__(self):
        self.calls = []

    def feed_forward(self, request_id, layer, joined):
        self.calls.append(("ffn", layer, int(joined.shape[0])))
        return joined + 1.0

    def round_trip(self, frame):
        self.calls.append(("relay", frame.layer, frame.op))
        return Frame.one(frame.request_id, frame.layer, frame.tensor)


class Sink:
    """A host's socket: keeps the replies it was sent."""

    def __init__(self):
        self.replies = []

    def sendmsg(self, buffers):
        self.replies.append(b"".join(bytes(b) for b in buffers))
        return sum(len(memoryview(b)) for b in buffers)


def a_frame(request_id, layer, rows, value, op=OP_FFN):
    return Frame(request_id, layer, (torch.full((rows, 4), float(value)),), op)


class TestTheMergeIsRowsForRows(CustomTestCase):
    def test_two_hosts_at_one_layer_become_one_upstream_call(self):
        """The whole point: the pool reads the layer's weights once for both."""
        upstream, stop = Upstream(), Stop(Upstream(), min_batch=2, max_wait_s=10.0)
        stop.upstream = upstream
        first, second = Sink(), Sink()
        stop.offer(a_frame(1, 3, rows=2, value=1.0), first)
        self.assertEqual(upstream.calls, [], "the first host departed alone")
        stop.offer(a_frame(2, 3, rows=5, value=2.0), second)

        self.assertEqual(upstream.calls, [("ffn", 3, 7)], "seven rows in one call")
        self.assertEqual(len(first.replies), 1)
        self.assertEqual(len(second.replies), 1)

    def test_a_lone_host_still_departs_when_the_bus_is_one_seat(self):
        upstream = Upstream()
        stop = Stop(upstream, min_batch=1, max_wait_s=10.0)
        stop.offer(a_frame(1, 0, rows=3, value=1.0), Sink())
        self.assertEqual(upstream.calls, [("ffn", 0, 3)])

    def test_a_layer_never_rides_with_another_layer(self):
        """A dense stack's layer weights differ: merging two layers would read one layer's weights
        for rows belonging to another."""
        upstream = Upstream()
        stop = Stop(upstream, min_batch=2, max_wait_s=10.0)
        stop.offer(a_frame(1, 0, rows=1, value=1.0), Sink())
        stop.offer(a_frame(2, 1, rows=1, value=2.0), Sink())
        self.assertEqual(upstream.calls, [], "two different layers completed a bus between them")


class TestOnlyIndependentRowsAreMerged(CustomTestCase):
    def test_an_op_that_calls_back_is_relayed_one_for_one(self):
        """A span carries per-request state on its return path. Merged, one host's callback would
        be answered to another -- and both would receive a tensor of the right shape, so nothing
        downstream would report it."""
        upstream = Upstream()
        stop = Stop(upstream, min_batch=2, max_wait_s=10.0)
        stop.offer(a_frame(1, 0, rows=1, value=1.0, op=OP_SPAN), Sink())
        stop.offer(a_frame(2, 0, rows=1, value=2.0, op=OP_SPAN), Sink())
        self.assertEqual(upstream.calls, [("relay", 0, OP_SPAN), ("relay", 0, OP_SPAN)])
        self.assertEqual(stop.report()["merged_calls"], 0)

    def test_a_pool_that_returns_the_wrong_row_count_is_refused(self):
        """Splitting a short reply would hand some host another host's rows, silently."""
        class Short(Upstream):
            def feed_forward(self, request_id, layer, joined):
                return joined[:1]

        stop = Stop(Short(), min_batch=2, max_wait_s=10.0)
        stop.offer(a_frame(1, 0, rows=2, value=1.0), Sink())
        with self.assertRaises(RuntimeError) as caught:
            stop.offer(a_frame(2, 0, rows=2, value=2.0), Sink())
        self.assertIn("another host's rows", str(caught.exception))


class TestTheTimeoutHalf(CustomTestCase):
    def test_a_partial_bus_waits_and_then_leaves(self):
        """Same rule as the pool's, from the same function: a strict minimum with no timeout hangs
        the last caller of a draining workload. Both halves in one case, because the interesting
        thing is the transition -- it must NOT have left on arrival and it must leave later."""
        import time as clock

        upstream = Upstream()
        stop = Stop(upstream, min_batch=4, max_wait_s=0.01)
        stop.offer(a_frame(1, 2, rows=1, value=1.0), Sink())
        self.assertEqual(upstream.calls, [], "a partial bus left on arrival")
        self.assertFalse(stop.depart_due(), "it left before its wait was up")
        clock.sleep(0.02)
        self.assertTrue(stop.depart_due(), "it never left")
        self.assertEqual(upstream.calls, [("ffn", 2, 1)])


if __name__ == "__main__":
    unittest.main()
