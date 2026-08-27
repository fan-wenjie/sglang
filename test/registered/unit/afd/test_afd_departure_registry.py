"""Who serves an op, and whether it calls back, arrive together or not at all.

The pool used to decide both by reading this file: a chain of `if`s picked the handler, and a
separate frozenset a hundred lines above said which ops must not depart on the connection thread.
Two places, and they disagreed once. `OP_LAYER` was added to the first and left out of the second,
so a linear layer's departure ran on the thread that owned the caller's socket, then asked that
caller for a recurrent state -- a message only that thread could have read. The pool hung. The
reading was

    no state reading for request 3 layer 0 within 30.0s

which names the far end and blames it, and the far end had answered immediately.

`register_departure(op, handler, calls_back=...)` is one call, so the pair cannot come apart. These
cases pin that, and pin that a second claim on one op is refused rather than resolved by import
order -- which would decide what arithmetic a caller received with nothing in the reply recording
which.
"""

import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd import pool_server
from sglang.test.test_utils import CustomTestCase

A_TEST_OP = (
    9901  # not a real op, and never sent: what is under test is the table, not the wire
)


class TestTheDepartureTableCannotComeApart(CustomTestCase):
    # Cleared before AND after. The table is module state, and this suite's base class retries a
    # failing case by calling the method again without running tearDown between attempts -- so a
    # case that only cleaned up afterwards failed its own retry on the registration it had just
    # made, and reported "retry() exceed maximum number of retries" rather than anything about
    # departures.
    def setUp(self):
        self._forget()

    def tearDown(self):
        self._forget()

    def _forget(self):
        pool_server._DEPARTURES.pop(A_TEST_OP, None)
        pool_server.CALLS_BACK_OPS.discard(A_TEST_OP)

    def test_a_handler_that_calls_back_is_in_the_set_that_keeps_it_off_the_socket_thread(
        self,
    ):
        """The deadlock above, as one assertion. Registering is what adds it -- there is no second
        place to remember."""
        self.assertNotIn(A_TEST_OP, pool_server.CALLS_BACK_OPS)
        pool_server.register_departure(A_TEST_OP, lambda *a: None, calls_back=True)
        self.assertIn(A_TEST_OP, pool_server.CALLS_BACK_OPS)

    def test_a_handler_that_does_not_call_back_stays_out_of_it(self):
        """Stated so the case above cannot pass by putting everything in the set, which would cost
        every ordinary departure a thread hop."""
        pool_server.register_departure(A_TEST_OP, lambda *a: None)
        self.assertNotIn(A_TEST_OP, pool_server.CALLS_BACK_OPS)

    def test_two_handlers_for_one_op_are_refused(self):
        """Import order would otherwise decide which arithmetic a caller got."""
        pool_server.register_departure(A_TEST_OP, lambda *a: None)
        with self.assertRaises(ValueError) as caught:
            pool_server.register_departure(A_TEST_OP, lambda *a: None)
        self.assertIn("import order", str(caught.exception).lower())

    def test_an_op_this_build_does_not_know_is_refused_by_name(self):
        """What standard AFD is, checked at the table rather than by reading the imports.

        The group cut's ops are not declared in this package at all -- a pool built from it has no
        entry for one and no constant naming one. What it must still do is REFUSE a frame carrying
        such a number, because the failure it replaces does not look like a failure: an unclaimed
        op used to fall through to the feed-forward, which answers whoever asked with arithmetic
        they did not ask for, in a reply shaped like a reply.

        The number is written out rather than imported. On the standard branch it was 9 --
        nothing there could import it. Here the derived line serves 9, so the case uses a number
        no arrangement claims: what it pins is the REFUSAL, not which op happens to be absent.
        """
        a_number_this_build_does_not_serve = 99

        self.assertNotIn(a_number_this_build_does_not_serve, pool_server._DEPARTURES)
        self.assertNotIn(a_number_this_build_does_not_serve, pool_server.CALLS_BACK_OPS)

        departure = pool_server.Departure.__new__(pool_server.Departure)
        frame = types.SimpleNamespace(op=a_number_this_build_does_not_serve)
        with self.assertRaises(ConnectionError) as caught:
            departure._depart(3, [(frame, None)])
        self.assertIn("nothing here serves it", str(caught.exception))

    def test_the_layer_op_left_with_the_other_arrangements(self):
        """OP_LAYER's server is gone, deliberately: the per-layer linear cut was an
        arrangement beside the span, and the submission keeps one arrangement. The op
        code stays reserved so an older host's LAYER frame is refused by name rather
        than queued as a feed-forward of the wrong meaning."""
        from sglang.srt.afd import pool_linear  # noqa: F401 -- the callback plumbing
        from sglang.srt.afd import pool_server
        from sglang.srt.afd.protocol import OP_LAYER

        self.assertNotIn(OP_LAYER, pool_server._DEPARTURES)

    def test_the_feed_forward_is_not_in_the_table_and_does_not_need_to_be(self):
        """It is what this pool IS, not something registered onto it.

        Stated because the case above refuses every op with no entry, and the feed-forward has
        none. The two must not converge on refusing everything -- which is what would happen if
        someone "completed" the table by adding OP_FFN to it and then dropped the branch that
        lets it through.
        """
        from sglang.srt.afd.protocol import OP_FFN

        self.assertNotIn(OP_FFN, pool_server._DEPARTURES)


if __name__ == "__main__":
    unittest.main()
