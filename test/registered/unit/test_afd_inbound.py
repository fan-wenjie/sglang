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
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.afd.pool_client import PoolClient, PoolClosed
from sglang.srt.afd.protocol import (
    INBOUND_OPS,
    OP_FFN,
    OP_STATE_READ,
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
                        send_frame(conn, Frame(frame.request_id, frame.layer + i,
                                               (torch.zeros(1, 4), torch.zeros(1, 1)),
                                               OP_STATE_READ))
                        got = decode(conn)
                        self.readings.append((got.op, got.layer, tuple(got.tensor.shape)))
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
                send_frame(conn, Frame(frame.request_id, frame.layer,
                                       (torch.zeros(1, 4),), OP_STATE_UPDATE))
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
            return None                      # nothing goes back

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
    """

    def test_inbound_ops_are_not_reply_ops(self):
        self.assertNotIn(OP_FFN, INBOUND_OPS)
        self.assertEqual(INBOUND_OPS, frozenset({OP_STATE_READ, OP_STATE_UPDATE}))


if __name__ == "__main__":
    unittest.main()
