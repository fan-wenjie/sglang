"""Freeing a request's history, which nothing else would ever do.

Two failures, both quiet. A slot that is never released grows until the pool refuses -- and it
refuses on some unlucky later request, not on the one that leaked. A slot that is reused without
being released hands its next occupant the previous one's past, which reads as a model answering
a question nobody asked.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch
from sglang.srt.afd.pool_attention import CachePool, KVHolder, LayerGeometry
from sglang.srt.afd.rendezvous import AppendLedger
from sglang.test.test_utils import CustomTestCase

HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
GEOMETRY = LayerGeometry(HEADS, KV_HEADS, HEAD_DIM, HEAD_DIM, HEAD_DIM**-0.5)


def _kv(n=1):
    t = torch.randn(n, KV_HEADS * HEAD_DIM)
    return t, t.clone()


class TestReleaseFreesTheHistory(CustomTestCase):
    def test_a_released_request_holds_nothing(self):
        pool = CachePool(KVHolder("cpu", 64), GEOMETRY)
        for layer in (0, 3, 7):
            for _ in range(4):
                pool.append_rows([11], layer, *_kv())
        before = pool.holder.bytes_held()
        self.assertGreater(before, 0)
        self.assertEqual(pool.holder.release(11), 3, "one entry per layer it reached")
        self.assertEqual(pool.holder.bytes_held(), 0)

    def test_releasing_one_request_leaves_the_others(self):
        pool = CachePool(KVHolder("cpu", 64), GEOMETRY)
        pool.append_rows([11], 0, *_kv())
        pool.append_rows([22], 0, *_kv())
        pool.holder.release(11)
        self.assertEqual(pool.holder.positions(11, 0), 0)
        self.assertEqual(pool.holder.positions(22, 0), 1)

    def test_a_reused_slot_starts_empty(self):
        """The failure this prevents: slot 11 finishes, slot 11 is handed to a new request, and
        the new one sweeps the old one's past."""
        pool = CachePool(KVHolder("cpu", 64), GEOMETRY)
        for _ in range(5):
            pool.append_rows([11], 0, *_kv())
        pool.holder.release(11)
        held = pool.append_rows([11], 0, *_kv())
        self.assertEqual(held, {11: 1}, "the new occupant's first token is its first position")

    def test_the_ledger_is_dropped_with_the_history(self):
        """They are two counts of the same thing kept by different processes. Releasing one and
        not the other makes the next query expect a history that was just thrown away, and the
        guard then refuses every call for that slot forever."""
        pool, ledger = CachePool(KVHolder("cpu", 64), GEOMETRY), AppendLedger()
        for _ in range(5):
            pool.append_rows([11], 0, *_kv())
            ledger.record(11, 0, 1)
        pool.holder.release(11)
        ledger.drop(11)
        pool.sweep_rows([11], 0, torch.randn(1, HEADS * HEAD_DIM),
                        expect=[ledger.posted(11, 0)])

    def test_without_dropping_the_ledger_the_guard_refuses(self):
        """Stated as a test because it is the symptom a half-done release produces: not a leak,
        but a request that can never be served again."""
        pool, ledger = CachePool(KVHolder("cpu", 64), GEOMETRY), AppendLedger()
        for _ in range(5):
            pool.append_rows([11], 0, *_kv())
            ledger.record(11, 0, 1)
        pool.holder.release(11)
        with self.assertRaises(RuntimeError):
            pool.sweep_rows([11], 0, torch.randn(1, HEADS * HEAD_DIM),
                            expect=[ledger.posted(11, 0)])

    def test_memory_returns_to_where_it_started(self):
        """A hundred requests through one pool leave it the size it began."""
        pool = CachePool(KVHolder("cpu", 64), GEOMETRY)
        baseline = pool.holder.bytes_held()
        for request_id in range(100):
            for _ in range(8):
                pool.append_rows([request_id], 0, *_kv())
            pool.holder.release(request_id)
        self.assertEqual(pool.holder.bytes_held(), baseline)


if __name__ == "__main__":
    unittest.main(verbosity=2)
