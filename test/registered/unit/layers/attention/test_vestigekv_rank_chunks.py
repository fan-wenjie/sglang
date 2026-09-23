"""Kernel tile sizes derived from the sketch rank (vestigekv/defaults.py)."""

import unittest

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-core")


class TestRankScaledChunks(CustomTestCase):
    """The kernels that hold a [rank, chunk] basis slice on chip keep that
    footprint across ranks: the chunk halves as the rank doubles, never below
    the 16-wide tensor-core minimum, and rank 64 keeps its tuned chunk."""

    def test_footprint_is_rank_invariant_down_to_the_minimum(self):
        self.assertEqual(D.d_block_for_rank(64), 64)
        self.assertEqual(D.d_block_for_rank(128), 32)
        self.assertEqual(D.d_block_for_rank(256), 16)
        self.assertEqual(D.d_block_for_rank(512), 16)  # floor
        self.assertEqual(D.d_block_for_rank(128, 64), 32)

    def test_a_rank_below_the_tuned_one_does_not_grow_the_chunk(self):
        # The proportional rule read downwards hands rank 16 a 256-wide chunk.
        # The prologue's other on-chip slices are [H, chunk], so they grow with
        # it and capture died needing 131088 B of shared memory against
        # SM120's 101376. The chunk is capped at the tuned width instead.
        self.assertEqual(D.d_block_for_rank(32), 64)
        self.assertEqual(D.d_block_for_rank(16), 64)
        self.assertEqual(D.d_block_for_rank(8), 64)

    def test_operand_rows_halve_past_rank_128(self):
        # Rank 256 at 64 rows overran the SM120 shared-memory limit on the box
        # (102400 B needed, 101376 B available); rank 64 and 128 keep 64 rows.
        self.assertEqual(D.a_block_for_rank(64), 64)
        self.assertEqual(D.a_block_for_rank(128), 64)
        self.assertEqual(D.a_block_for_rank(256), 32)


if __name__ == "__main__":
    unittest.main(verbosity=3)
