"""A batch whose rows belong to different requests, which is every batch a server runs.

Until this, every pool path was a batch-of-one path: the cache pool took one request id per frame
and applied it to every row. A decode forward carries one token from each of N requests and each
has its own history, so that filed N requests' keys under one request's past and swept one
request's token against another's. Both are fluent and wrong.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

import torch
from sglang.srt.afd.pool_attention import CachePool, KVHolder, LayerGeometry
from sglang.srt.afd.sweep_ahead import row_request_ids
from sglang.test.test_utils import CustomTestCase

HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
GEOMETRY = LayerGeometry(HEADS, KV_HEADS, HEAD_DIM, HEAD_DIM, HEAD_DIM**-0.5)


def _pool(max_context=64):
    return CachePool(KVHolder("cpu", max_context), GEOMETRY)


def _kv(n, fill):
    t = torch.full((n, KV_HEADS * HEAD_DIM), float(fill))
    return t, t.clone()


class TestRowsGoToTheirOwnHistory(CustomTestCase):
    def test_appending_a_mixed_batch_files_each_row_under_its_own_request(self):
        pool = _pool()
        k, v = torch.cat([_kv(1, 1)[0], _kv(1, 2)[0], _kv(1, 3)[0]]), None
        k = torch.stack([torch.full((KV_HEADS * HEAD_DIM,), float(i)) for i in (1, 2, 3)])
        held = pool.append_rows([11, 22, 33], 0, k, k)
        self.assertEqual(held, {11: 1, 22: 1, 33: 1})
        self.assertEqual(pool.holder.positions(11, 0), 1)
        self.assertEqual(pool.holder.positions(22, 0), 1)

    def test_a_request_with_several_rows_keeps_them_together(self):
        """A chunk arrives as contiguous rows of one request; the causal boundary inside it is
        positional, so splitting or reordering them is not the same chunk."""
        pool = _pool()
        k = torch.stack([torch.full((KV_HEADS * HEAD_DIM,), float(i)) for i in range(5)])
        held = pool.append_rows([7, 7, 7, 9, 9], 0, k, k)
        self.assertEqual(held, {7: 3, 9: 2})

    def test_sweeping_a_mixed_batch_reads_each_row_from_its_own_past(self):
        """The check that matters: two requests with different histories, swept in one frame,
        must get what they would have got alone."""
        pool, alone = _pool(), _pool()
        for step in range(3):
            k = torch.randn(1, KV_HEADS * HEAD_DIM)
            pool.append_rows([11], 0, k, k)
            alone.append_rows([11], 0, k, k)
        for step in range(5):
            k = torch.randn(1, KV_HEADS * HEAD_DIM)
            pool.append_rows([22], 0, k, k)

        q = torch.randn(2, HEADS * HEAD_DIM)
        together, lse_together = pool.sweep_rows([11, 22], 0, q)
        one, lse_one = alone.sweep_rows([11], 0, q[:1])
        self.assertTrue(torch.allclose(together[0], one[0], atol=1e-5),
                        "request 11's row must not see request 22's history")
        self.assertTrue(torch.allclose(lse_together[0], lse_one[0], atol=1e-5))

    def test_a_row_count_that_does_not_match_the_ids_is_refused(self):
        pool = _pool()
        with self.assertRaises(RuntimeError):
            pool.sweep_rows([11, 22], 0, torch.randn(3, HEADS * HEAD_DIM))
        with self.assertRaises(RuntimeError):
            pool.append_rows([11], 0, torch.randn(2, KV_HEADS * HEAD_DIM),
                             torch.randn(2, KV_HEADS * HEAD_DIM))


class TestTheHostNamesTheRowsCorrectly(CustomTestCase):
    def test_decode_rows_are_the_requests(self):
        batch = types.SimpleNamespace(req_pool_indices=torch.tensor([5, 6, 7]),
                                      extend_seq_lens=None)
        self.assertEqual(row_request_ids(batch).tolist(), [5, 6, 7])

    def test_extend_rows_repeat_a_request_for_every_token_it_brought(self):
        """Prefill is where a batch size and a row count part company, and where reading the ids
        off the batch size alone files most of a prompt under the wrong request."""
        batch = types.SimpleNamespace(req_pool_indices=torch.tensor([5, 6]),
                                      extend_seq_lens=torch.tensor([3, 2]))
        self.assertEqual(row_request_ids(batch).tolist(), [5, 5, 5, 6, 6])


if __name__ == "__main__":
    unittest.main(verbosity=2)
