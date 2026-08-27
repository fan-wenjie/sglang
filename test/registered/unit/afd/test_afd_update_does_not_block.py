"""A deferred state update must not stop the pool computing while it is sent.

`OP_STATE_UPDATE` is filed off the critical path: nothing waits for a reply. That was read as
"costs nothing", and it was wrong. `send_frame` reaches `protocol._payload_of`, which copies with
a plain `.to("cpu")` -- a stream synchronisation -- and then writes the socket, all on the thread
that was in the middle of a span. No reply awaited, but the pool stopped anyway, once per linear
layer per token: 48 times a token on this model, against the handover's 17.

The deadline on an update is generous. It only has to land before the next token reads that
layer, which is a whole step away. So the send belongs on another thread and the copy belongs
behind an event.

The assertion is causal, as in `test_afd_handover_does_not_block`: `_post_update` must return
having sent nothing, and the send must follow the copy. FIFO order is asserted too -- two updates
to one slot that overtook each other would apply this token's key before the last one's, and the
state would be wrong with nothing raising.
"""

import queue
import threading
import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.pool_server import Departure
from sglang.srt.afd.protocol import OP_STATE_UPDATE
from sglang.test.test_utils import CustomTestCase

LAYER, REQUEST = 7, 11


class _CopyOnDemand:
    """`torch.cuda.Event` with completion under the test's control, so ordering is pinned."""

    released = threading.Event()

    def record(self):
        pass

    def synchronize(self):
        if not self.released.wait(timeout=5):
            raise AssertionError("the copy was never released by the test")


class TestADeferredUpdateIsSentOffTheSpansThread(CustomTestCase):
    def setUp(self):
        _CopyOnDemand.released = threading.Event()
        self.sent = []
        self.arrived = threading.Event()
        # a Departure is not constructed: it would start its own threads and bind a model, and
        # what is under test is two of its methods and the queue between them
        self.dep = types.SimpleNamespace(
            _outbox=queue.Queue(),
            _wire_lock=threading.RLock(),
        )
        # `_post_update` is a thin caller of the general one now, so the stand-in carries that
        # too -- bound to the real method, not replaced by a stub, because what is under test is
        # what those methods do with the queue between them.
        self.dep.post_unawaited = types.MethodType(Departure.post_unawaited, self.dep)

    def _post(self, *, rows=1):
        with unittest.mock.patch.object(torch.cuda, "Event", _CopyOnDemand):
            Departure._post_update(
                self.dep,
                object(),
                REQUEST,
                LAYER,
                (
                    torch.zeros(rows, 4),
                    torch.zeros(rows, 4),
                    torch.zeros(rows, 2),
                    torch.zeros(rows, 2),
                ),
            )

    def _drain_once(self, count=1):
        def record(sock, frame):
            self.sent.append((frame.layer, frame.op))
            if len(self.sent) >= count:
                self.arrived.set()

        with unittest.mock.patch("sglang.srt.afd.pool_server.send_frame", record):
            worker = threading.Thread(
                target=Departure._drain_outbox, args=(self.dep,), daemon=True
            )
            worker.start()
            _CopyOnDemand.released.set()
            self.assertTrue(self.arrived.wait(timeout=5), "the update was never sent")

    def test_posting_sends_nothing_and_returns(self):
        """The property: the span thread goes straight on to the next layer."""
        self._post()
        self.assertEqual(
            self.sent,
            [],
            "the update went out inside _post_update, so the span stopped to send it",
        )
        self.assertEqual(self.dep._outbox.qsize(), 1, "it was neither sent nor queued")

    def test_the_queued_update_is_sent_once_the_copy_lands(self):
        """Deferring must not mean dropping."""
        self._post()
        self._drain_once()
        self.assertEqual(self.sent, [(LAYER, OP_STATE_UPDATE)])

    def test_updates_keep_their_order(self):
        """Two updates to one slot that overtook each other would apply the keys out of order.

        Nothing downstream would raise: the state would simply be the wrong state, which is the
        failure mode this whole line keeps meeting.
        """
        for layer in (1, 2, 3):
            with unittest.mock.patch.object(torch.cuda, "Event", _CopyOnDemand):
                Departure._post_update(
                    self.dep,
                    object(),
                    REQUEST,
                    layer,
                    (
                        torch.zeros(1, 4),
                        torch.zeros(1, 4),
                        torch.zeros(1, 2),
                        torch.zeros(1, 2),
                    ),
                )
        self._drain_once(count=3)
        self.assertEqual([layer for layer, _ in self.sent], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
