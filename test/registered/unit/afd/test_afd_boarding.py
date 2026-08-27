"""The boarding question, and the fact that it is asked again after every send.

Two batches offset by a round trip was the arrangement this replaced. It fixes the pairing in
advance, which needs a stable cadence, and a batch's cadence is its attention's -- set by contexts
that are not uniform inside one batch. A 1k request and a 128k one reach the same boundary 2378 us
apart, so an offset chosen for the batch is wrong for every request in it and slips further every
layer. Nothing in an aggregate shows that: throughput, occupancy and mean latency all look healthy
while the schedule drifts.

So the answer is recomputed rather than reused, and these cases pin both halves -- the rule
itself, and WHO asks it and when. The second half is the one no aggregate can see: a departure
that is ready the instant a send finishes, and is left to a timer instead, costs its callers real
latency on a queue that was never empty.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd import pool_server
from sglang.srt.afd.boarding import ready_to_depart
from sglang.srt.afd.pool_server import Departure
from sglang.srt.afd.protocol import OP_FFN, Frame
from sglang.test.test_utils import CustomTestCase


class TestTheRule(CustomTestCase):
    def test_a_full_stop_goes_now(self):
        self.assertTrue(
            ready_to_depart(waiting=2, min_batch=2, waited_s=0.0, max_wait_s=1.0)
        )

    def test_a_partial_stop_waits_and_then_stops_waiting(self):
        """A strict minimum with no timeout hangs the last caller of a draining workload rather
        than failing it, which is the worse of the two failures."""
        self.assertFalse(
            ready_to_depart(waiting=1, min_batch=2, waited_s=0.1, max_wait_s=1.0)
        )
        self.assertTrue(
            ready_to_depart(waiting=1, min_batch=2, waited_s=1.5, max_wait_s=1.0)
        )

    def test_an_empty_stop_never_goes(self):
        """A departure with no riders reads the layer's weights for nobody."""
        self.assertFalse(
            ready_to_depart(waiting=0, min_batch=1, waited_s=99.0, max_wait_s=1.0)
        )


class Silent:
    """A socket that accepts a reply and counts the bytes. The departure sends before it looks at
    the queue again, so there has to be somewhere for the reply to go."""

    def __init__(self):
        self.sent = 0

    def sendmsg(self, buffers):
        written = sum(len(memoryview(b)) for b in buffers)
        self.sent += written
        return written


class TestTheQueueIsListenedToAgainAfterEverySend(CustomTestCase):
    """The half that no aggregate shows.

    A caller answered by this departure is, at that instant, computing its own attention, and one
    that finished during the departure is already at the next stop. If the thread that just sent
    goes back to its socket, that ready work waits for the timer -- and the timer is a fraction of
    max_wait_s away, on a queue that was never empty.
    """

    def _departure(self, served):
        def forward(batch, layer):
            served.append((layer, int(batch.shape[0])))
            return batch

        return Departure(forward, 1, 5.0, "cpu")

    def test_a_second_ready_layer_departs_on_the_sending_thread(self):
        """max_wait_s is five seconds here: anything the timer would have taken cannot appear
        within this test, so what runs is what the sender drained."""
        served = []
        departure = self._departure(served)
        frame = Frame(1, 0, (torch.zeros(2, 4),), OP_FFN)
        later = Frame(2, 1, (torch.zeros(3, 4),), OP_FFN)
        # queued directly: it is already at the stop when the send for layer 0 finishes
        departure._waiting.setdefault(1, []).append((later, Silent()))

        departure.offer(frame, Silent())
        self.assertEqual(
            served,
            [(0, 2), (1, 3)],
            "the sender went back to its socket and left a ready call to the timer",
        )

    def test_the_drain_stops_when_the_queue_is_empty(self):
        served = []
        departure = self._departure(served)
        departure.offer(Frame(1, 0, (torch.zeros(1, 4),), OP_FFN), Silent())
        self.assertEqual(served, [(0, 1)])

    def test_a_call_that_calls_back_is_never_drained_here(self):
        """This thread owns the caller's socket. A callback taken from it waits for a message only
        it can receive -- the deadlock the derived branch's callback cases exist for, reached by a
        different route.

        Driven by an op registered HERE rather than by the group cut's LAYER. Standard AFD has no
        calls-back op of its own -- the ones it had were the group cut's and left with it -- and a
        case that borrowed one would have been deleted along with them, taking this property with
        it. The property is the queue's, not that op's."""
        from sglang.srt.afd.pool_server import register_departure

        an_op_that_calls_back = 9902
        register_departure(an_op_that_calls_back, lambda *a: None, calls_back=True)
        self.addCleanup(pool_server._DEPARTURES.pop, an_op_that_calls_back, None)
        self.addCleanup(pool_server.CALLS_BACK_OPS.discard, an_op_that_calls_back)

        served = []
        departure = self._departure(served)
        waiting = Frame(2, 1, (torch.zeros(3, 4),), an_op_that_calls_back)
        departure._waiting.setdefault(1, []).append((waiting, Silent()))
        departure.offer(Frame(1, 0, (torch.zeros(2, 4),), OP_FFN), Silent())
        self.assertEqual(
            served, [(0, 2)], "a calling-back departure was taken on this thread"
        )
        self.assertEqual(
            len(departure._waiting[1]), 1, "and it is still queued for the timer"
        )


if __name__ == "__main__":
    unittest.main()
