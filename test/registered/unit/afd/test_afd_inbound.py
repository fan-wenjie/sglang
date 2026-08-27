"""A request arriving on the reply socket, which is what the group cut makes the pool do.

The pool holds a linear layer's weights and the host holds its recurrent state, so the pool calls
BACK mid-span -- while the host is blocked waiting for that same span's reply. Two message
directions share one socket and the reader has to tell them apart.

Every case here guards a failure that is a hang or a swap rather than a wrong number:

  * an inbound request filed as a reply hangs the caller it was keyed as, and hands a state
    reading to whoever asked for that key
  * two threads inside `send_frame` on one socket interleave a header with somebody else's
    payload, and the far end reads the remainder as the next frame's header
  * a client with no handler must say which side was misconfigured, not close the socket
"""

import socket
import threading
import time
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.afd.pool_client import PoolClient, PoolClosed
from sglang.srt.afd.protocol import (
    INBOUND_OPS,
    OP_FFN,
    OP_STATE_APPLY,
    OP_STATE_EARLY,
    OP_STATE_MIX,
    OP_STATE_READ,
    OP_STATE_SCAN,
    OP_STATE_UPDATE,
    Frame,
    decode,
    send_frame,
)
from sglang.test.test_utils import CustomTestCase


class Pool:
    """A pool that answers a feed-forward, but calls back for a state reading on the way."""

    def __init__(self, callbacks=1, interleave=True):
        self.callbacks = callbacks
        self.interleave = interleave
        self.readings = []
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.ready = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        conn, _ = self.sock.accept()
        self.ready.set()
        try:
            while True:
                frame = decode(conn)
                if frame is None:
                    return
                if self.interleave:
                    for i in range(self.callbacks):
                        send_frame(
                            conn,
                            Frame(
                                frame.request_id,
                                frame.layer + i,
                                (torch.zeros(1, 4), torch.zeros(1, 1)),
                                OP_STATE_READ,
                            ),
                        )
                        got = decode(conn)
                        self.readings.append(
                            (got.op, got.layer, tuple(got.tensor.shape))
                        )
                send_frame(conn, Frame.one(frame.request_id, frame.layer, frame.tensor))
        except OSError:
            return

    def close(self):
        self.sock.close()


class TestARequestOnTheReplySocket(CustomTestCase):
    """The pool asks mid-call and the host answers without the outstanding call being disturbed."""

    def test_the_callback_is_answered_and_the_reply_still_arrives(self):
        pool = Pool(callbacks=1)
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)
        seen = []

        def serve(frame):
            seen.append(frame.op)
            return (torch.ones(1, 8),)

        client.serve = serve
        try:
            handle = client.issue(1, 3, torch.zeros(1, 4))
            out = client.collect(handle, "cpu")
            self.assertEqual(tuple(out.shape), (1, 4))
            self.assertEqual(seen, [OP_STATE_READ])
            self.assertEqual(pool.readings, [(OP_STATE_READ, 3, (1, 8))])
        finally:
            pool.close()

    def test_several_callbacks_in_one_call(self):
        """A span asks three times, once for each linear layer, before it answers."""
        pool = Pool(callbacks=3)
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)
        client.serve = lambda f: (torch.full((1, 8), float(f.layer)),)
        try:
            out = client.collect(client.issue(1, 3, torch.zeros(1, 4)), "cpu")
            self.assertEqual(tuple(out.shape), (1, 4))
            self.assertEqual([r[1] for r in pool.readings], [3, 4, 5])
        finally:
            pool.close()

    def test_a_deferred_op_needs_no_answer(self):
        """The state update is one-way: the state has to be right by the NEXT step, not this one."""
        answered = []

        class Deferring(Pool):
            def _serve(self):
                conn, _ = self.sock.accept()
                self.ready.set()
                frame = decode(conn)
                send_frame(
                    conn,
                    Frame(
                        frame.request_id,
                        frame.layer,
                        (torch.zeros(1, 4),),
                        OP_STATE_UPDATE,
                    ),
                )
                send_frame(conn, Frame.one(frame.request_id, frame.layer, frame.tensor))
                try:
                    while decode(conn) is not None:
                        pass
                except OSError:
                    pass

        pool = Deferring()
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)

        def serve(frame):
            answered.append(frame.op)
            return None  # nothing goes back

        client.serve = serve
        try:
            out = client.collect(client.issue(1, 3, torch.zeros(1, 4)), "cpu")
            self.assertEqual(tuple(out.shape), (1, 4))
            self.assertEqual(answered, [OP_STATE_UPDATE])
        finally:
            pool.close()


class TestAClientWithNoHandler(CustomTestCase):
    """The two ends disagree about who holds the state, and the message has to say so.

    Without it the socket dies and the host reports "closed mid-call" -- a message about a socket
    that names neither side's configuration, which is the failure the HELLO exchange was added for.
    """

    def test_it_names_the_disagreement(self):
        pool = Pool(callbacks=1)
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0, reconnect=False)
        try:
            with self.assertRaises(PoolClosed) as caught:
                client.collect(client.issue(1, 3, torch.zeros(1, 4)), "cpu")
            self.assertIn("recurrent state", str(client._failure))
        finally:
            pool.close()


class TestTheOpcodeListIsTheOnlyThingSeparatingThem(CustomTestCase):
    """A reply and a request are the same bytes apart from the opcode.

    If an op were on both lists the reader would answer a reply and file a request, and both
    callers would wait forever. Pinned because the lists are two literals in two files.

    This pins the SET. It catches an op added to INBOUND_OPS and asks whether that was meant; it
    does not catch an op added to the host's service and left out of INBOUND_OPS, which is the
    failure that deadlocks a live pair. `test_afd_inbound_ops_cover_the_service.py` catches that
    one by reading both files.
    """

    def test_inbound_ops_are_not_reply_ops(self):
        self.assertNotIn(OP_FFN, INBOUND_OPS)
        base = {OP_STATE_READ, OP_STATE_UPDATE, OP_STATE_SCAN, OP_STATE_MIX}
        early = {OP_STATE_EARLY, OP_STATE_APPLY}
        # membership travels with the handlers: the base tree lists its own four, and
        # importing the early-read package adds its two beside their register_op calls.
        # Either exact set is lawful; anything else is an op added to the protocol
        # without deciding which direction it travels -- the one thing that separates a
        # reply from a request on a socket that carries both.
        self.assertIn(
            set(INBOUND_OPS),
            (base, base | early),
            "an op was added to the protocol without deciding which direction it "
            "travels, or membership was added without its handler",
        )


if __name__ == "__main__":
    unittest.main()


class TestEachCallbackIsCountedAgainstItsOwnDenominator(CustomTestCase):
    """A per-call average over a MIXED population is not a measurement, and once was not caught.

    The first version of `_count_inbound` kept one pair of running totals across every op in
    `INBOUND_OPS`. Reads are the op the pool blocks on and there were 11,000 of them; the
    denominator was 30,500. The average came out at 2.07 ms and went into a write-up as "host
    compute, 45% of the state read". The real figure, timed inside the read alone, is 0.193 ms --
    the wire is 88% rather than 48%, and the work it pointed at was on the wrong machine.

    Nothing about 2.07 ms looked wrong. A mixed average is always plausible, which is exactly why
    this has to be pinned rather than trusted: the reading here is one slow READ against three
    instant UPDATEs, a shape whose mixed average (~5 ms) and whose true read (~20 ms) cannot be
    mistaken for each other.
    """

    class Mixed(Pool):
        """One state read, answered, then three one-way updates, then the reply."""

        def _serve(self):
            conn, _ = self.sock.accept()
            self.ready.set()
            frame = decode(conn)
            send_frame(
                conn,
                Frame(
                    frame.request_id, frame.layer, (torch.zeros(1, 4),), OP_STATE_READ
                ),
            )
            decode(conn)  # the host's reading
            for _ in range(3):
                send_frame(
                    conn,
                    Frame(
                        frame.request_id,
                        frame.layer,
                        (torch.zeros(1, 4),),
                        OP_STATE_UPDATE,
                    ),
                )
            send_frame(conn, Frame.one(frame.request_id, frame.layer, frame.tensor))
            try:
                while decode(conn) is not None:
                    pass
            except OSError:
                pass

    def _drive(self):
        pool = self.Mixed()
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)

        def serve(frame):
            if frame.op == OP_STATE_READ:
                time.sleep(0.02)
                return (torch.ones(1, 4),)
            return None

        client.serve = serve
        try:
            client.collect(client.issue(1, 3, torch.zeros(1, 4)), "cpu")
            return client.inbound_report()
        finally:
            pool.close()

    def test_each_op_carries_the_count_its_average_was_divided_by(self):
        report = self._drive()
        self.assertEqual(report["state_read"]["calls"], 1)
        self.assertEqual(report["state_update"]["calls"], 3)

    def test_the_slow_op_is_not_diluted_by_the_fast_ones(self):
        """The whole point: a read's own average, not a read averaged over four callbacks."""
        report = self._drive()
        # 20 ms of serving, divided by ONE read. Diluted over all four it would be about 5.
        self.assertGreater(report["state_read"]["serve_ms"], 15.0)
        self.assertLess(report["state_update"]["serve_ms"], 5.0)

    def test_an_op_that_is_not_a_callback_is_refused_rather_than_folded_in(self):
        """A population that is not a callback at all would restart the same failure."""
        pool = Pool(callbacks=0, interleave=False)
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)
        try:
            with self.assertRaises(ValueError) as caught:
                client._count_inbound(OP_FFN, 0.001, 0.0)
            self.assertIn("ffn", str(caught.exception))
        finally:
            pool.close()
