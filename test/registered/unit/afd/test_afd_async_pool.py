"""The pool is asynchronous, batches across callers, and does not confuse two of them.

Check 4 in the afd-early-q skill, and the reason it exists: a naive port issues the feed-forward
and waits for it, which reproduces the synchronous arrangement exactly. Every later measurement
then shows no benefit, correctly, because there is none -- and nothing in the code looks wrong.
So the asynchrony is asserted here against a pool whose service time is known, rather than assumed
from the fact that the calls are on separate lines.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import threading
import time
import unittest

import torch

from sglang.srt.afd.pool_client import PoolClient
from sglang.srt.afd.pool_server import serve
from sglang.srt.afd.protocol import Frame, decode, encode
from sglang.test.test_utils import CustomTestCase

SERVICE_S = 0.05  # what one departure costs, made visible rather than measured
WIDTH = 16


class _Ready(threading.Event):
    port: int = 0


def _start_pool(min_batch=2, max_wait_s=0.02, service_s=SERVICE_S, record=None):
    """A pool whose feed-forward is a sleep, so the schedule is the only thing under test."""

    def forward(batch: torch.Tensor, layer: int) -> torch.Tensor:
        time.sleep(service_s)
        if record is not None:
            record.append((layer, int(batch.shape[0]), time.perf_counter()))
        return batch + 1.0

    ready = _Ready()
    thread = threading.Thread(
        target=serve,
        args=(forward, "127.0.0.1", 0, min_batch, max_wait_s, "cpu", ready),
        daemon=True,
    )
    thread.start()
    if not ready.wait(timeout=10):
        raise RuntimeError("the pool did not bind")
    return ready.port


def _hidden(tokens=2, value=0.0):
    return torch.full((tokens, WIDTH), value, dtype=torch.bfloat16)


class TestProtocol(CustomTestCase):
    def test_a_frame_survives_the_round_trip_bit_for_bit(self):
        import socket

        a, b = socket.socketpair()
        t = torch.randn(3, WIDTH).to(torch.bfloat16)
        a.sendall(encode(Frame.one(7, 11, t)))
        got = decode(b)
        self.assertEqual(got.request_id, 7)
        self.assertEqual(got.layer, 11)
        self.assertTrue(torch.equal(got.tensor, t), "bfloat16 crosses as its own bytes")

    def test_a_truncated_frame_is_an_error_not_a_short_read(self):
        import socket

        a, b = socket.socketpair()
        payload = encode(Frame.one(1, 1, _hidden()))
        a.sendall(payload[: len(payload) - 4])
        a.close()
        with self.assertRaises(ConnectionError):
            decode(b)


class TestAsynchrony(CustomTestCase):
    def test_issue_returns_before_the_pool_has_answered(self):
        """The whole arrangement. If issue() blocked, the sweep could not run underneath it."""
        port = _start_pool(min_batch=1)
        client = PoolClient(f"127.0.0.1:{port}", connect_timeout_s=5)
        try:
            t0 = time.perf_counter()
            handle = client.issue(1, 0, _hidden())
            issued = time.perf_counter() - t0
            self.assertLess(
                issued,
                SERVICE_S / 2,
                "issue() waited for the pool; that is the synchronous arrangement with extra "
                "machinery",
            )
            out = client.collect(handle, "cpu")
            self.assertEqual(tuple(out.shape), (2, WIDTH))
        finally:
            client.close()

    def test_work_between_issue_and_collect_is_not_paid_for_twice(self):
        """Sleep the sweep's length while the pool works; the total must be the max, not the sum."""
        port = _start_pool(min_batch=1)
        client = PoolClient(f"127.0.0.1:{port}", connect_timeout_s=5)
        try:
            t0 = time.perf_counter()
            handle = client.issue(2, 0, _hidden())
            time.sleep(SERVICE_S)  # this is where the sweep would run
            client.collect(handle, "cpu")
            elapsed = time.perf_counter() - t0
            self.assertLess(
                elapsed,
                1.7 * SERVICE_S,
                f"issue+work+collect took {elapsed:.3f}s against a {SERVICE_S:.3f}s service time; "
                f"nothing overlapped",
            )
        finally:
            client.close()

    def test_two_requests_at_one_layer_do_not_overwrite_each_other(self):
        """The slot-keying bug. It appears only with concurrency, so one request proves nothing."""
        port = _start_pool(min_batch=2)
        client = PoolClient(f"127.0.0.1:{port}", connect_timeout_s=5)
        try:
            h1 = client.issue(101, 7, _hidden(value=1.0))
            h2 = client.issue(202, 7, _hidden(value=2.0))
            out1 = client.collect(h1, "cpu")
            out2 = client.collect(h2, "cpu")
            self.assertTrue(
                torch.allclose(out1.float(), torch.full_like(out1.float(), 2.0))
            )
            self.assertTrue(
                torch.allclose(out2.float(), torch.full_like(out2.float(), 3.0))
            )
        finally:
            client.close()


class TestDeparture(CustomTestCase):
    def test_a_departure_carries_everyone_at_the_stop(self):
        """Two waiting and a third arriving in the same instant is one departure of three."""
        record = []
        port = _start_pool(min_batch=2, record=record)
        client = PoolClient(f"127.0.0.1:{port}", connect_timeout_s=5)
        try:
            handles = [client.issue(i, 3, _hidden(tokens=1)) for i in range(3)]
            for h in handles:
                client.collect(h, "cpu")
            riders = [n for _, n, _ in record]
            self.assertEqual(sum(riders), 3)
            self.assertLessEqual(
                len(record), 2, f"three callers took {len(record)} departures"
            )
        finally:
            client.close()

    def test_a_lone_caller_still_departs_on_the_timeout(self):
        """Without a max wait the last caller of a draining workload waits for a partner that
        never arrives, and the request hangs rather than fails."""
        port = _start_pool(min_batch=2, max_wait_s=0.05)
        client = PoolClient(f"127.0.0.1:{port}", connect_timeout_s=5)
        try:
            t0 = time.perf_counter()
            out = client.collect(client.issue(1, 0, _hidden()), "cpu")
            waited = time.perf_counter() - t0
            self.assertEqual(tuple(out.shape), (2, WIDTH))
            self.assertLess(
                waited, 2.0, "a lone caller must not wait forever for a partner"
            )
        finally:
            client.close()

    def test_a_late_rider_does_not_hold_the_bus(self):
        """The coach model, and the reason the batch is re-formed at every layer.

        A batch only has to be re-formed where latency VARIES. On this arrangement the varying
        stage is the HOST's own attention -- a feed-forward costs the same whatever the context
        (361 us), a linear attention likewise (200 us), and softmax attention runs from 19 us at
        1k to 2397 us at 128k. So the caller with the long context arrives late for every
        departure, and if a departure waited for it, every short-context caller on the pool would
        pay that caller's context length.

        It does not wait: the bus leaves on its timeout with whoever is aboard, and the late
        rider takes the next one. Asserted by ORDER -- the early caller's answer arrives before
        the late one has even issued -- because every aggregate is identical either way. Total
        throughput, riders histogram and mean latency all look the same when the fast caller has
        been made to wait for the slow one.
        """
        record = []
        port = _start_pool(min_batch=2, max_wait_s=0.02, service_s=0.01, record=record)
        early = PoolClient(f"127.0.0.1:{port}", connect_timeout_s=5)
        late = PoolClient(f"127.0.0.1:{port}", connect_timeout_s=5)
        try:
            handle = early.issue(1, 0, _hidden())
            answered = early.collect(handle, "cpu")  # departs on the timeout, alone
            self.assertEqual(tuple(answered.shape), (2, WIDTH))
            self.assertEqual(len(record), 1, "the early caller waited for the late one")

            # only NOW does the slow caller finish its attention and arrive
            late.collect(late.issue(2, 0, _hidden()), "cpu")
            self.assertEqual(
                len(record), 2, "the late rider never got a bus of its own"
            )
            # two rows each: one rider's own, twice, against the 4 a shared bus carries below
            self.assertEqual([n for _, n, _ in record], [2, 2])
        finally:
            early.close()
            late.close()

    def test_two_riders_at_one_stop_still_share_a_bus(self):
        """The other half of the same rule, and what makes the pool worth pooling: when both are
        on time they ride together and the weight read is paid once. A pool that always departed
        alone would be correct and pointless."""
        record = []
        port = _start_pool(min_batch=2, max_wait_s=1.0, service_s=0.01, record=record)
        a = PoolClient(f"127.0.0.1:{port}", connect_timeout_s=5)
        b = PoolClient(f"127.0.0.1:{port}", connect_timeout_s=5)
        try:
            ha, hb = a.issue(1, 0, _hidden()), b.issue(2, 0, _hidden())
            a.collect(ha, "cpu")
            b.collect(hb, "cpu")
            self.assertEqual(len(record), 1, "two riders at one stop took two buses")
            self.assertEqual(record[0][1], 4, "both callers' rows rode together")
        finally:
            a.close()
            b.close()

    def test_a_layer_never_departs_with_another_layer(self):
        """A dense stack's layer weights differ, so a departure is same-layer or it is wrong."""
        record = []
        port = _start_pool(min_batch=2, max_wait_s=0.02, record=record)
        client = PoolClient(f"127.0.0.1:{port}", connect_timeout_s=5)
        try:
            hs = [client.issue(1, 0, _hidden()), client.issue(2, 1, _hidden())]
            for h in hs:
                client.collect(h, "cpu")
            layers = [layer for layer, _, _ in record]
            self.assertEqual(sorted(layers), [0, 1], "each layer departed on its own")
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()


class TestPoolDoesNotCallItself(CustomTestCase):
    """A pool sharing a process with the host must not route work back to itself.

    Guards a hang that actually happened: the single-machine test put both roles on one model, the
    host replaced layer.mlp.forward with a router, and the pool -- which looked the method up per
    call -- found the router, sent the work back out and waited for its own reply. Idle GPU, no
    error, no timeout. On two machines the two models are separate and the bug cannot appear, so
    it would have been found by a benchmark that hung rather than by a test.
    """

    def test_the_pool_binds_its_forward_before_the_router_replaces_it(self):
        from sglang.srt.afd.roles import make_pool_forward

        class FakeMLP:
            def __init__(self, tag):
                self.tag = tag

            def forward(self, x):
                return x + self.tag

        class FakeLayer:
            def __init__(self, tag):
                self.mlp = FakeMLP(tag)

        class FakeInner:
            def __init__(self):
                self.layers = [FakeLayer(1.0), FakeLayer(2.0)]

        class FakeModel:
            def __init__(self):
                self.model = FakeInner()

        model = FakeModel()
        pool_forward = make_pool_forward(model)

        # the host installs a router afterwards, as it does in the shared-process case
        def router(_x):
            raise AssertionError("the pool re-entered the host's router")

        model.model.layers[0].mlp.forward = router

        out = pool_forward(torch.zeros(2, 4), 0)
        self.assertTrue(
            torch.allclose(out, torch.ones(2, 4)), "the pool ran the wrong layer"
        )

    def test_a_layer_outside_the_stack_is_refused(self):
        from sglang.srt.afd.roles import make_pool_forward

        class FakeInner:
            def __init__(self):
                self.layers = []

        class FakeModel:
            def __init__(self):
                self.model = FakeInner()

        with self.assertRaises(RuntimeError):
            make_pool_forward(FakeModel())(torch.zeros(1, 4), 3)
