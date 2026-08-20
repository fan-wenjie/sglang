"""A host and a pool started with different roles, which is the misconfiguration that will happen.

Two processes, two flag sets, and nothing connecting them but an address. A host configured to
send sweeps to a pool that only runs feed-forwards will send it a frame it cannot parse; the
connection thread dies and the host reports "closed mid-call" -- a message about a socket that
names neither side's configuration. That happened twice while this was being built, and both times
the diagnosis was minutes of reading logs to discover a flag.

The handshake makes it a startup failure that says which capability is missing.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import socket
import threading
import unittest

import torch
from sglang.srt.afd.pool_client import PoolClient, PoolClosed
from sglang.srt.afd.pool_server import Departure
from sglang.srt.afd.protocol import Frame, decode, send_frame
from sglang.test.test_utils import CustomTestCase


class Pool:
    """A pool with whatever capabilities the test gives it, answering HELLO and nothing else."""

    def __init__(self, cache=None, attention=None):
        self.departure = Departure(lambda b, l: b, 1, 0.005, "cpu")
        self.departure.cache = cache
        self.departure.attention = attention
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        try:
            while True:
                frame = decode(conn)
                if frame is None:
                    return
                if not self.departure.answer_directly(frame, conn):
                    send_frame(conn, Frame.one(frame.request_id, frame.layer, frame.tensor))
        except OSError:
            return

    def close(self):
        self.sock.close()


class TestTheHandshakeNamesWhatIsMissing(CustomTestCase):
    def test_a_weights_pool_satisfies_a_host_that_only_needs_one(self):
        pool = Pool()
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)
        try:
            served = client.require(PoolClient.NEEDS_FEED_FORWARD)
            self.assertEqual(served & PoolClient.NEEDS_FEED_FORWARD, PoolClient.NEEDS_FEED_FORWARD)
        finally:
            client.close()
            pool.close()

    def test_a_weights_pool_refuses_a_host_that_needs_a_cache(self):
        """The misconfiguration itself: two processes started with different roles."""
        pool = Pool()
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)
        try:
            with self.assertRaises(PoolClosed) as caught:
                client.require(PoolClient.NEEDS_CACHE)
            message = str(caught.exception)
            self.assertIn("cache", message)
            self.assertIn("different roles", message,
                          "the message has to point at the configuration, not the socket")
        finally:
            client.close()
            pool.close()

    def test_a_cache_pool_serves_a_cache_and_not_the_projections(self):
        """A cache pool holds no weights, so it cannot project a key -- and a host configured for
        the projection arm has to learn that before it sends one."""
        pool = Pool(cache=object())
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)
        try:
            client.require(PoolClient.NEEDS_CACHE)
            with self.assertRaises(PoolClosed):
                client.require(PoolClient.NEEDS_KV_PROJECTION)
        finally:
            client.close()
            pool.close()

    def test_a_pool_with_the_attention_service_serves_both(self):
        pool = Pool(attention=object())
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)
        try:
            client.require(PoolClient.NEEDS_CACHE | PoolClient.NEEDS_KV_PROJECTION)
        finally:
            client.close()
            pool.close()

    def test_every_missing_capability_is_named_at_once(self):
        pool = Pool()
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)
        try:
            with self.assertRaises(PoolClosed) as caught:
                client.require(PoolClient.NEEDS_CACHE | PoolClient.NEEDS_KV_PROJECTION)
            message = str(caught.exception)
            self.assertIn("cache", message)
            self.assertIn("projections", message)
        finally:
            client.close()
            pool.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
