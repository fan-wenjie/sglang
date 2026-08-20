"""The one race this arrangement can lose silently: a query overtaking its own append.

A stage ends by computing this step's key and value, posts them to the database, and does NOT wait
-- the join uses the host's own copy, and the cache only has to hold them by the NEXT step. So
step n's append and step n+1's query are both in flight at once. If the query wins, the sweep
covers a history with a hole in it, nothing raises, and the model attends to a past missing one
token while its output stays fluent.

Ordering is real today because both frames travel one connection and the pool answers a
connection's frames in arrival order. That is a property of the transport, not of the design, and
the measurements in this tree recommend sharding the client across two links -- which would break
it. So the count travels with the query and is checked.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

import torch
from sglang.srt.afd.pool_attention import CachePool, KVHolder
from sglang.srt.afd.rendezvous import AppendLedger
from sglang.test.test_utils import CustomTestCase

HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256


def _pool(max_context=32):
    attn = types.SimpleNamespace(tp_q_head_num=HEADS, tp_k_head_num=KV_HEADS,
                                 qk_head_dim=HEAD_DIM, v_head_dim=HEAD_DIM,
                                 scaling=HEAD_DIM**-0.5)
    return CachePool(KVHolder("cpu", max_context), [types.SimpleNamespace(attn=attn)], {})


def _kv(n=1):
    return (torch.randn(n, KV_HEADS * HEAD_DIM), torch.randn(n, KV_HEADS * HEAD_DIM))


class TestAQueryMayNotRunAheadOfItsAppend(CustomTestCase):
    def test_the_counts_agreeing_is_the_normal_case(self):
        pool = _pool()
        k, v = _kv()
        pool.append(1, 0, k, v)
        pool.sweep(1, 0, torch.randn(1, HEADS * HEAD_DIM), expect=1)

    def test_a_query_expecting_more_than_the_pool_holds_is_refused(self):
        """The race itself. The host has posted two positions and only one has landed."""
        pool = _pool()
        k, v = _kv()
        pool.append(1, 0, k, v)
        with self.assertRaises(RuntimeError) as caught:
            pool.sweep(1, 0, torch.randn(1, HEADS * HEAD_DIM), expect=2)
        self.assertIn("hole", str(caught.exception))

    def test_a_query_expecting_fewer_is_refused_too(self):
        """The opposite slip -- an append counted twice, or a stale ledger -- would sweep a token
        that has not been emitted yet. Same check, no extra code, and it would otherwise be an
        attention over a position from the future."""
        pool = _pool()
        for _ in range(2):
            k, v = _kv()
            pool.append(1, 0, k, v)
        with self.assertRaises(RuntimeError):
            pool.sweep(1, 0, torch.randn(1, HEADS * HEAD_DIM), expect=1)

    def test_without_a_count_nothing_is_checked(self):
        """Backward compatible on purpose: a caller that does not track its appends still works,
        and gets the silent behaviour. The check is opt-in because the ledger is the caller's."""
        pool = _pool()
        k, v = _kv()
        pool.append(1, 0, k, v)
        pool.sweep(1, 0, torch.randn(1, HEADS * HEAD_DIM))

    def test_the_check_is_per_layer_and_per_request(self):
        pool = _pool()
        k, v = _kv()
        pool.append(1, 0, k, v)
        pool.append(1, 0, *_kv())
        pool.append(2, 0, *_kv())
        pool.sweep(1, 0, torch.randn(1, HEADS * HEAD_DIM), expect=2)
        pool.sweep(2, 0, torch.randn(1, HEADS * HEAD_DIM), expect=1)


class TestTheHostsLedger(CustomTestCase):
    def test_it_counts_positions_not_calls(self):
        """A prefill posts a whole chunk in one append. Counting calls would put the ledger a
        prompt's length behind on the very first query."""
        ledger = AppendLedger()
        self.assertEqual(ledger.record(1, 0, 17), 17)
        self.assertEqual(ledger.record(1, 0, 1), 18)
        self.assertEqual(ledger.posted(1, 0), 18)

    def test_layers_are_counted_apart(self):
        """In a dataflow schedule a request does not advance one layer per step -- layers progress
        independently, so a count per layer is the only one that stays true."""
        ledger = AppendLedger()
        ledger.record(1, 0, 5)
        ledger.record(1, 3, 2)
        self.assertEqual(ledger.posted(1, 0), 5)
        self.assertEqual(ledger.posted(1, 3), 2)
        self.assertEqual(ledger.posted(1, 7), 0)

    def test_dropping_a_request_clears_only_its_own(self):
        ledger = AppendLedger()
        ledger.record(1, 0, 3)
        ledger.record(2, 0, 4)
        self.assertEqual(ledger.drop(1), 1)
        self.assertEqual(ledger.posted(1, 0), 0)
        self.assertEqual(ledger.posted(2, 0), 4)

    def test_the_ledger_and_the_pool_agree_through_a_decode_stream(self):
        """The two counts are kept by different processes off different events; walking a stream
        is what shows they stay equal."""
        pool, ledger = _pool(), AppendLedger()
        for step in range(6):
            pool.sweep(1, 0, torch.randn(1, HEADS * HEAD_DIM), expect=ledger.posted(1, 0))
            k, v = _kv()
            pool.append(1, 0, k, v)
            ledger.record(1, 0, 1)
            self.assertEqual(ledger.posted(1, 0), pool.holder.positions(1, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
