"""The one send the window has to cover, asserted by ORDER rather than by a stopwatch.

Under the group cut only three things cross between the two ends, and only one of them can close
the window by blocking:

    OP_STATE_UPDATE   pool to host, no answer awaited -- `history_service` files it as "deferred,
                      no answer", so it is off the critical path and costs the span nothing
    OP_STATE_READ     pool to host and back, blocking -- but the host is inside `collect_kv`
                      waiting for the span while it happens, so the time is on the HOST's account
                      and is not something the window is meant to hide
    OP_SPAN_Q         the shifted read point. This one, and only this one, is what the window
                      exists to cover: it is sent one feed-forward before the span's output, and
                      the whole claim of shift 1 is that `last.mlp` runs on top of the send

So this file tests `_handover` and nothing else.

What it asserts is CAUSAL, not temporal. `hand_over` must return without having sent, leaving the
caller free to issue the feed-forward; the send must happen only after the copy completes, on
another thread. The naive version -- a plain `.cpu()` and a send inline -- synchronises the stream
before the last feed-forward is issued, so the pool sits idle for the copy and the send and only
then computes. It returns the same tokens. That is the point: nothing in any output distinguishes
the two, and a benchmark distinguishes them only by a smaller number, which has many other
explanations. This project has already paid for that lesson twice -- once on the host side, and
once when a whole throughput reading turned out to be two runs of the same arm.
"""

import threading
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.protocol import OP_SPAN_Q
from sglang.srt.afd import pool_span
from sglang.test.test_utils import CustomTestCase

GROUP = 3


class _CopyOnDemand:
    """Stands in for `torch.cuda.Event`, with the completion of the copy under the test's control.

    A real event fires when the GPU gets to it, which would make "did the send wait for the copy"
    a race. Here the test decides when the copy is done, so the ordering is pinned rather than
    sampled.
    """

    released = threading.Event()

    def record(self):
        pass

    def synchronize(self):
        if not self.released.wait(timeout=5):
            raise AssertionError("the copy was never released by the test")


class TestTheReadPointIsSentWithoutBlockingTheSpan(CustomTestCase):
    def setUp(self):
        _CopyOnDemand.released = threading.Event()
        self.sends = []
        self.sent = threading.Event()
        # only `_reply_pieces` is reached from `_handover`, so a Departure is not built:
        # constructing one would open sockets and bind a model for a test about ordering
        self.server = object()
        # patched for the whole case, not just around the call: the send happens on another
        # thread AFTER `hand_over` returns, which is the property under test, so a patch scoped
        # to the call would be undone before the send it is meant to observe
        self.riding, self.counts = [], []
        patch = unittest.mock.patch.object(pool_span, "_reply_pieces", self._record)
        patch.start()
        self.addCleanup(patch.stop)

    def _record(self, departure, riding, counts, group, out, op):
        self.sends.append((group, op))

    def _hand_over(self):
        hand_over = pool_span._handover(
            self.server, self.riding, self.counts, GROUP, self.sent
        )
        with unittest.mock.patch.object(torch.cuda, "Event", _CopyOnDemand):
            hand_over(torch.zeros(2, 4))
        return hand_over

    def test_it_returns_before_anything_is_sent(self):
        """The property the window rests on: the caller gets control back to issue `last.mlp`.

        With the copy deliberately unfinished, a correct handover has sent nothing and returned.
        The inline version would have blocked here until the send completed.
        """
        self._hand_over()
        self.assertEqual(
            self.sends,
            [],
            "the read point went out inside hand_over, so the feed-forward behind "
            "it had not been issued yet and the window is shut",
        )
        self.assertFalse(self.sent.is_set())

    def test_the_send_happens_once_the_copy_completes(self):
        """The other half: deferring it must not mean dropping it."""
        self._hand_over()
        _CopyOnDemand.released.set()
        self.assertTrue(self.sent.wait(timeout=5), "the handover never finished")
        self.assertEqual(self.sends, [(GROUP, OP_SPAN_Q)])

    def test_a_send_that_raises_still_releases_the_span(self):
        """The pool must fail rather than hang.

        The span thread waits on `sent` before sending the second half. A handover that died
        without setting it would leave the pool alive with nothing queued and the host blocked in
        `collect_kv` -- which is exactly what a previous mis-keyed reply looked like, and it cost a
        deployment round to tell the two apart.
        """

        def explode(*_args, **_kwargs):
            raise OSError("the caller went away")

        unittest.mock.patch.object(pool_span, "_reply_pieces", explode).start()
        self._hand_over()
        _CopyOnDemand.released.set()
        self.assertTrue(
            self.sent.wait(timeout=5),
            "the handover raised and never set `sent`, so the span thread waits out its timeout",
        )


if __name__ == "__main__":
    unittest.main()
