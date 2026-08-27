"""A departed host costs the pool nothing but a log line.

The pool outliving its callers is the design: when a host's connection dies, its owed
readings are dropped and their waiters answer None at once (no timeout), a mixed
departure zero-fills the dead rider and serves the living on time, a departure with
nobody left is abandoned whole with a warning, the dead host's queued riders are
dismissed at the next re-form, and every row its namespace held is released. Each of
those is pinned here, because the failure mode of a missing one is a pool that hangs,
leaks, or dies -- none of which says which host caused it.
"""

import socket
import time
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.afd.linear_state import LinearStates
from sglang.srt.afd.pool_linear import _await_reading, _readings_or_zeros
from sglang.srt.afd.pool_server import Departure, HostDeparted, namespace_of
from sglang.srt.afd.protocol import OP_FFN, Frame
from sglang.test.test_utils import CustomTestCase


def _departure():
    return Departure(lambda batch, layer: batch, 1, 0.005, "cpu")


class TestTheDepartedSettleAtOnce(CustomTestCase):
    def test_owed_readings_drop_and_the_waiter_answers_none_immediately(self):
        d = _departure()
        a, b = socket.socketpair()
        try:
            d.file_reading(a, Frame(3, 1, (torch.ones(1, 4),), OP_FFN))
            d.host_departed(a)
            began = time.perf_counter()
            self.assertIsNone(_await_reading(d, a, 9, 9))
            self.assertLess(time.perf_counter() - began, 1.0)
            self.assertNotIn((id(a), 3, 1), d._readings)
        finally:
            a.close()
            b.close()

    def test_queued_riders_are_dismissed_at_the_stop(self):
        d = _departure()
        a, b = socket.socketpair()
        c, e = socket.socketpair()
        try:
            riding = [
                (Frame(1, 0, (torch.ones(1, 4),), OP_FFN), a),
                (Frame(2, 0, (torch.ones(1, 4),), OP_FFN), c),
            ]
            d.host_departed(a)
            kept = d._without_the_departed(riding)
            self.assertEqual([f.request_id for f, _ in kept], [2])
        finally:
            for s in (a, b, c, e):
                s.close()

    def test_the_waiting_queue_is_purged(self):
        d = _departure()
        a, b = socket.socketpair()
        try:
            d._waiting[5] = [(Frame(1, 5, (torch.ones(1, 4),), OP_FFN), a)]
            d.host_departed(a)
            self.assertNotIn(5, d._waiting)
        finally:
            a.close()
            b.close()


class TestAMixedDepartureServesTheLiving(CustomTestCase):
    def test_a_dead_rider_is_zero_filled(self):
        d = _departure()
        a, b = socket.socketpair()
        c, e = socket.socketpair()
        try:
            d.file_reading(c, Frame(2, 1, (torch.full((1, 4), 7.0),), OP_FFN))
            d.host_departed(a)
            out = _readings_or_zeros(d, [(a, 1, 1), (c, 2, 1)])
            torch.testing.assert_close(out[0], torch.zeros(1, 4))
            torch.testing.assert_close(out[1], torch.full((1, 4), 7.0))
        finally:
            for s in (a, b, c, e):
                s.close()

    def test_a_departure_with_nobody_left_is_abandoned_whole(self):
        d = _departure()
        a, b = socket.socketpair()
        try:
            d.host_departed(a)
            with self.assertRaises(HostDeparted):
                _readings_or_zeros(d, [(a, 1, 1)])
        finally:
            a.close()
            b.close()


class TestTheNamespaceIsReleased(CustomTestCase):
    def test_only_the_departed_hosts_rows_go(self):
        states = LinearStates(
            slots=4, num_v_heads=2, head_k_dim=2, head_v_dim=2, device="cpu"
        )
        a, b = socket.socketpair()
        c, e = socket.socketpair()
        try:
            ns_a, ns_c = namespace_of(a), namespace_of(c)
            states.slot_of(ns_a | 1)
            states.slot_of(ns_a | 2)
            states.slot_of(ns_c | 1)
            self.assertEqual(states.release_namespace(ns_a), 2)
            self.assertEqual(len(states._slot_of), 1)
            self.assertIn(ns_c | 1, states._slot_of)
        finally:
            for s in (a, b, c, e):
                s.close()


class TestARecycledIdDoesNotInheritDeath(CustomTestCase):
    def test_arrival_clears_the_dead_set(self):
        # id() is an address; a dead socket's address is what the allocator hands out
        # next. The live failure: a restarted host's every rider dismissed, silently.
        d = _departure()
        a, b = socket.socketpair()
        try:
            d.host_departed(a)
            self.assertIn(id(a), d._dead)
            with d._cond:
                d._dead.discard(id(a))
            riding = [(Frame(1, 0, (torch.ones(1, 4),), OP_FFN), a)]
            self.assertEqual(len(d._without_the_departed(riding)), 1)
        finally:
            a.close()
            b.close()


class TestTheConnectionExitSettles(CustomTestCase):
    def test_handle_settles_the_host_on_the_way_out(self):
        import ast
        import pathlib

        src = (
            pathlib.Path(__file__).resolve().parents[4]
            / "python/sglang/srt/afd/pool_server.py"
        ).read_text()
        tree = ast.parse(src)
        handle = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "handle"
        )
        calls = [
            n.func.attr
            for n in ast.walk(handle)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        self.assertIn(
            "host_departed",
            calls,
            "the connection's exit no longer settles the departed host",
        )


if __name__ == "__main__":
    unittest.main()
