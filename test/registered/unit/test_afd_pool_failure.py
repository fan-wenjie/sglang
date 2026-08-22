"""What happens to a host when its pool goes away.

Without reconnection a pool restart is fatal to the host: every call after it raises, the
scheduler dies, and a server that was serving hundreds of requests stops because one of its two
processes was replaced. That is not a serving system.

The contract this pins is deliberately asymmetric:

    in-flight work FAILS, loudly. Its answers went with the old process, and a client that resent
    those frames would be guessing at whether the pool had already applied them -- which for a
    pool that holds histories means a key appended twice, a past with one token in it twice, and
    fluent output from it.

    future work RECOVERS. The next call opens a new connection.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import socket
import threading
import unittest

import torch
from sglang.srt.afd.pool_client import PoolClient, PoolClosed
from sglang.srt.afd.protocol import OP_FFN, Frame, decode, send_frame
from sglang.test.test_utils import CustomTestCase


class Echo:
    """A pool that returns what it is given, and can be told to die."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.served = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._conns = []
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        """Accept, and hand the connection over UNDER THE LOCK.

        Without it there is a window that made `test_the_degradation_is_counted` fail about two
        runs in three of the full suite and pass every time alone: `accept()` returns, `close()`
        runs `drop_connections()` and clears the list, and only then does this thread append -- so
        that connection is never dropped, its serving thread lives on, and a client that was
        supposed to find a dead pool gets its calls answered. The count then reads 0 fallbacks
        where 3 were expected, and the message says nothing about a race.
        """
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with self._lock:
                if self._stop.is_set():
                    conn.close()
                    return
                self._conns.append(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        try:
            while True:
                frame = decode(conn)
                if frame is None:
                    return
                self.served += 1
                send_frame(conn, Frame.one(frame.request_id, frame.layer, frame.tensor, OP_FFN))
        except OSError:
            return

    def drop_connections(self):
        with self._lock:
            conns, self._conns = self._conns, []
        for conn in conns:
            try:
                conn.close()
            except OSError:
                pass

    def close(self):
        self._stop.set()
        self.drop_connections()
        self.sock.close()


class TestAPoolRestartIsSurvivable(CustomTestCase):
    def setUp(self):
        self.pool = Echo()

    def tearDown(self):
        self.pool.close()

    def test_a_call_works_before_and_after_a_reconnect(self):
        client = PoolClient(f"127.0.0.1:{self.pool.port}", 5.0)
        try:
            hidden = torch.ones(2, 8)
            first = client.collect(client.issue(1, 0, hidden), "cpu")
            self.assertTrue(torch.equal(first, hidden))

            self.pool.drop_connections()
            self.assertTrue(client.reconnect(), "a live pool must be reconnectable")

            again = client.collect(client.issue(2, 0, hidden), "cpu")
            self.assertTrue(torch.equal(again, hidden), "the next call recovers")
            self.assertEqual(client.reconnects, 1)
        finally:
            client.close()

    def test_reconnecting_fails_the_calls_that_were_outstanding(self):
        """They are not retried. Resending a frame the pool may already have applied is how an
        append lands twice, and a history with one token in it twice reads as fluent."""
        client = PoolClient(f"127.0.0.1:{self.pool.port}", 5.0)
        try:
            with client._cond:
                client._slots[(9, 1)] = torch.ones(1, 1)
                client._replies[(9, 2, 5)] = (torch.ones(1, 1),)
            client.reconnect()
            with client._cond:
                self.assertEqual(len(client._slots), 0)
                self.assertEqual(len(client._replies), 0)
        finally:
            client.close()

    def test_reconnection_can_be_switched_off(self):
        """A deployment that would rather fail than serve through a flapping pool says so, and
        gets the old behaviour without a surprise."""
        client = PoolClient(f"127.0.0.1:{self.pool.port}", 5.0, reconnect=False)
        try:
            self.assertFalse(client.reconnect())
            self.assertEqual(client.reconnects, 0)
        finally:
            client.close()

    def test_it_gives_up_after_a_bounded_number_of_attempts(self):
        """A pool that flaps forever must not turn the host into a reconnection loop that never
        serves a token."""
        client = PoolClient(f"127.0.0.1:{self.pool.port}", 5.0, max_reconnects=2)
        try:
            self.assertTrue(client.reconnect())
            self.assertTrue(client.reconnect())
            self.assertFalse(client.reconnect(), "the third is refused")
        finally:
            client.close()

    def test_a_dead_pool_reports_failure_rather_than_hanging(self):
        """The call that was in flight when the pool went away raises; it does not wait forever
        for an answer that is not coming."""
        client = PoolClient(f"127.0.0.1:{self.pool.port}", 5.0, reconnect=False)
        try:
            handle = client.issue(1, 0, torch.ones(2, 8))
            client.collect(handle, "cpu")
            self.pool.close()
            with self.assertRaises((PoolClosed, OSError)):
                for i in range(50):
                    client.collect(client.issue(i + 2, 0, torch.ones(2, 8)), "cpu")
        finally:
            client.close()


class TestTheRouterRecoversAndNotJustTheClient(CustomTestCase):
    """The gap the first version of this file left, and a stress test found.

    Those cases asserted that `reconnect()` works. Nothing asserted that anything CALLS it at the
    moment a pool dies -- and the router only guarded its `issue`, while a dying pool fails at the
    `collect`, because that is where a call is outstanding when the process goes away. An
    exception there kills sglang's scheduler, so a pool restart stopped a server that was carrying
    hundreds of requests. The suite was green throughout.
    """

    def _model(self):
        import types

        mlp = types.SimpleNamespace()
        mlp.forward = lambda x: x * 3
        layer = types.SimpleNamespace(mlp=mlp)
        return types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]))

    def test_a_pool_that_dies_at_the_collect_does_not_raise(self):
        from sglang.srt.afd.roles import install_pool_routing

        pool = Echo()
        try:
            client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)
            model = self._model()
            routing = install_pool_routing(model, client, (0,))
            hidden = torch.ones(2, 8)
            self.assertTrue(torch.equal(model.model.layers[0].mlp.forward(hidden), hidden))

            pool.close()
            out = model.model.layers[0].mlp.forward(hidden)
            self.assertTrue(torch.equal(out, hidden * 3),
                            "served locally, from the weights this host also has")
            self.assertGreaterEqual(routing._local_fallbacks, 1)
            client.close()
        finally:
            pool.close()

    def test_the_degradation_is_counted(self):
        """A run that fell back for half its layers must not be able to report the arrangement's
        throughput under the arrangement's name."""
        from sglang.srt.afd.roles import install_pool_routing

        pool = Echo()
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0, reconnect=False)
        model = self._model()
        routing = install_pool_routing(model, client, (0,))
        # one call THROUGH the pool first, so the connection is established and served before it
        # is taken away. Closing a pool whose connection was still in the listen backlog tests
        # whichever side won a race, and this test is about what happens after a pool that WAS
        # working goes away.
        model.model.layers[0].mlp.forward(torch.ones(1, 4))
        self.assertEqual(pool.served, 1, "the pool never answered, so nothing was taken away")
        self.assertEqual(routing._local_fallbacks, 0)
        pool.close()
        for _ in range(3):
            model.model.layers[0].mlp.forward(torch.ones(1, 4))
        self.assertEqual(routing._local_fallbacks, 3)
        client.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
