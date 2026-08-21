"""Two message directions on one socket, end to end. The test three deadlocks were missing.

Under the group cut the pool calls BACK to the host mid-span, on the same socket the host is
blocked waiting for that span's reply. That has deadlocked three times, in three different places,
and every time the symptom was a hang: no traceback, no wrong value, no failing case. A watchdog
timeout says nothing about what was waiting.

The unit tests on each side pass in all three. What was missing is this: one pool, one client, and
a span that actually calls back.

Every case here is written to FAIL BY TIMING OUT rather than to hang, because a hanging test is
indistinguishable from a slow one and gets deleted.
"""

import socket
import threading
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from sglang.srt.afd.pool_client import PoolClient
from sglang.srt.afd.pool_server import Departure
from sglang.srt.afd.protocol import (
    OP_SPAN,
    OP_STATE_READ,
    Frame,
    decode,
    send_frame,
)
from sglang.test.test_utils import CustomTestCase

DEADLINE = 25.0


class Runner:
    """A span that calls back `asks` times before it answers, like a real one does per linear layer."""

    def __init__(self, asks=3):
        self.asks = asks
        self.seen = []

    def run(self, request_ids, group, attn_output, positions, on_query=None):
        for layer in range(self.asks):
            reading = self._local.ask_host(layer, request_ids, torch.zeros(1, 2, 4))
            self.seen.append(tuple(reading.shape))
        if on_query is not None:
            on_query(torch.zeros(1, 4))
        # (query, key, value), which is what the real runner returns. An earlier version of this
        # fake returned two, the departure thread died on the unpack, and the symptom was
        # indistinguishable from the deadlock this file is about.
        return torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, 4)


class Pool:
    """A real Departure over a real socket, with a span runner that calls back."""

    def __init__(self, asks=3, min_batch=1):
        self.departure = Departure(lambda b, l: b, min_batch, 0.005, "cpu")
        self.runner = Runner(asks)
        self.runner._local = threading.local()
        self.departure.span = self.runner
        self.departure.start()
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.error = None
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        conn, _ = self.sock.accept()
        try:
            while True:
                frame = decode(conn)
                if frame is None:
                    return
                if frame.op == OP_STATE_READ:
                    self.departure.file_reading(conn, frame)
                    continue
                if self.departure.answer_directly(frame, conn):
                    continue
                self.departure.offer(frame, conn)
        except BaseException as e:                       # noqa: BLE001 -- reported, not swallowed
            self.error = e

    def close(self):
        self.departure.stop()
        self.sock.close()


class TestASpanThatCallsBackCompletes(CustomTestCase):
    """The whole point. Three deadlocks, and none of them had a case that ran this.

    Fails by timing out. A version that hangs would be deleted for being slow, and the bug it
    guards is precisely a hang.
    """

    def a_client(self, pool):
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0, reconnect=False)
        client.serve = lambda frame: (torch.ones(1, 8),)
        return client

    def test_the_span_answers_and_the_callbacks_were_made(self):
        pool = Pool(asks=3)
        client = self.a_client(pool)
        done, result = threading.Event(), {}

        def call():
            try:
                handle = client.issue_frame(
                    1, 0, (torch.zeros(1, 4), torch.zeros(1, 1), torch.zeros(1, 1)), OP_SPAN)
                result["reply"] = client.collect_frame(handle._replace(op=OP_SPAN), "cpu")
            except BaseException as e:                   # noqa: BLE001
                result["error"] = e
            finally:
                done.set()

        threading.Thread(target=call, daemon=True).start()
        try:
            self.assertTrue(done.wait(DEADLINE),
                            "the span never answered: the callback and the reply are deadlocked "
                            "on one socket, which is the bug this file exists for")
            self.assertIsNone(result.get("error"), f"{result.get('error')}")
            self.assertEqual(len(pool.runner.seen), 3)
        finally:
            pool.close()

    def test_it_holds_when_the_departure_is_taken_by_the_connection_thread(self):
        """min_batch=1 means every offer completes a batch, so the CONNECTION thread departs it.

        That is the deployment's setting and the exact case that deadlocked: the thread that owns
        the socket becomes the thread waiting on a message that arrives on it.
        """
        pool = Pool(asks=2, min_batch=1)
        client = self.a_client(pool)
        done = threading.Event()

        def call():
            handle = client.issue_frame(
                1, 0, (torch.zeros(1, 4), torch.zeros(1, 1), torch.zeros(1, 1)), OP_SPAN)
            try:
                client.collect_frame(handle._replace(op=OP_SPAN), "cpu")
            except BaseException:                        # noqa: BLE001
                pass
            done.set()

        threading.Thread(target=call, daemon=True).start()
        try:
            self.assertTrue(done.wait(DEADLINE),
                            "deadlocked with min_batch=1, which is what the deployment runs")
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
