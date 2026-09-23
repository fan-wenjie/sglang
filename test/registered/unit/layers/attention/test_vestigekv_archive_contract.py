"""The scan's unit changes with pooling; the row mapping must not.

Derived property, not a mirror: the archive's indexing contract is that a
scan entry maps back to the pool rows it scored. Unpooled that is the archive
selection itself; pooled, one entry covers P closed-prefix positions whose
pool row ids are NOT consecutive (a paged allocator assigns them), so the
mapping has to go through _pos_all. A rewrite that expanded to consecutive
row ids would recall the wrong tokens and nothing downstream would notice.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

T, R, POOL = 32, 8, 4


def _tier(pool_size):
    with envs.SGLANG_VESTIGEKV_ARCHIVE_POOL.override(pool_size):
        t = RecallTier(r=R)
    t._csk_all = torch.arange(T * R, dtype=torch.float16).view(T, R)
    t._rho_all = torch.arange(T, dtype=torch.float32)
    # deliberately NOT consecutive: this is what a paged allocator hands over
    t._pos_all = (torch.arange(T, dtype=torch.int32) * 7 + 3) % 211
    t._arch_idx = torch.arange(0, T, 2, dtype=torch.int32)  # every other row
    return t


class TestScanUnit(CustomTestCase):
    def test_unpooled_scans_the_archive_selection(self):
        t = _tier(1)
        csk, rho, spread, n = t.scan_operands
        self.assertEqual(n, T // 2)
        self.assertIsNone(spread)
        self.assertEqual(csk.shape[0], T // 2)

    def test_pooled_scans_every_group_of_the_closed_prefix(self):
        t = _tier(POOL)
        csk, rho, spread, n = t.scan_operands
        self.assertEqual(n, T // POOL)
        self.assertIsNotNone(spread)
        self.assertEqual(spread.shape[0], T // POOL)

    def test_pooling_cuts_the_scanned_entries_against_the_archive(self):
        self.assertLess(_tier(POOL).scan_operands[3], _tier(1).scan_operands[3])


class TestRowMapping(CustomTestCase):
    def test_unpooled_fired_entries_map_through_the_archive(self):
        t = _tier(1)
        got = t.scan_rows(torch.tensor([0, 3]))
        want = t._pos_all[t._arch_idx.to(torch.int64)][[0, 3]]
        self.assertEqual(got.tolist(), want.tolist())

    def test_pooled_group_maps_to_its_own_rows_not_consecutive_ids(self):
        t = _tier(POOL)
        got = t.scan_rows(torch.tensor([2]))
        want = t._pos_all[2 * POOL : 3 * POOL]
        self.assertEqual(got.tolist(), want.tolist())
        # the guard this test exists for: those ids are not a run
        self.assertNotEqual(got.tolist(), list(range(int(got[0]), int(got[0]) + POOL)))

    def test_a_group_past_the_built_prefix_is_clipped_not_read_out_of_bounds(self):
        t = _tier(POOL)
        t._pos_all = t._pos_all[: T - 2]
        got = t.scan_rows(torch.tensor([T // POOL - 1]))
        self.assertEqual(got.shape[0], POOL - 2)


if __name__ == "__main__":
    unittest.main()
