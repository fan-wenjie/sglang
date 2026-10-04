# SPDX-License-Identifier: Apache-2.0
"""The decode step's new row must reach every layer's kept table.

Bug this guards (2026-10-04): the per-step append lived inside the CSR pack
kernel, which also built a CSR the DSA fork never reads. Deleting the CSR took
the append with it, and nothing failed -- both registered suites passed and the
server started. The kept table simply stopped growing between closes, which
showed up only as a counter move in a served smoke (kept 1120 -> 1005 per call
at 32k, fetched 555 -> 371) two queue jobs later.

Black-box contract: after one step, every layer's kept_len is one higher and
the row at the old length is this step's pool row. Duplicate lanes (shared trash
slots) advance the length exactly once, and the colliding lanes agree on the
slot they wrote, because they all append at the same pre-step length.
"""

import unittest

import torch

from sglang.srt.layers.attention.vestigekv.dsa_branch import kept_append


class TestKeptAppend(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        self.dev = "cuda"

    def _stacks(self, L, R1, CAP, start):
        kept = torch.zeros(L, R1, CAP, dtype=torch.int32, device=self.dev)
        klen = torch.full((L, R1), start, dtype=torch.int32, device=self.dev)
        return kept, klen

    def test_every_layer_gets_this_steps_row(self):
        L, R1, CAP, start = 11, 9, 64, 7
        kept, klen = self._stacks(L, R1, CAP, start)
        slots = torch.tensor([1, 3, 5], dtype=torch.int64, device=self.dev)
        loc = torch.tensor([100, 200, 300], dtype=torch.int32, device=self.dev)
        kept_append(slots=slots, loc=loc, kept_stack=kept, klen_stack=klen)
        for li in range(L):
            for i, s in enumerate((1, 3, 5)):
                self.assertEqual(int(klen[li, s]), start + 1, f"layer {li} slot {s}")
                self.assertEqual(int(kept[li, s, start]), int(loc[i]))
        # a slot no lane named is untouched
        self.assertEqual(int(klen[0, 0]), start)
        self.assertEqual(int(kept[0, 0, start]), 0)

    def test_a_duplicate_lane_advances_the_length_once(self):
        """Shared trash lanes name the same slot; the torch scatter this kernel
        replaced let the last write win and advanced the length once."""
        L, R1, CAP, start = 3, 5, 32, 2
        kept, klen = self._stacks(L, R1, CAP, start)
        slots = torch.tensor([2, 2, 2], dtype=torch.int64, device=self.dev)
        loc = torch.tensor([11, 22, 33], dtype=torch.int32, device=self.dev)
        kept_append(slots=slots, loc=loc, kept_stack=kept, klen_stack=klen)
        for li in range(L):
            self.assertEqual(int(klen[li, 2]), start + 1)
            self.assertEqual(int(kept[li, 2, start]), 33)  # last write wins

    def test_repeated_steps_grow_the_table_monotonically(self):
        L, R1, CAP = 2, 4, 40
        kept, klen = self._stacks(L, R1, CAP, 0)
        slots = torch.tensor([1], dtype=torch.int64, device=self.dev)
        for step in range(12):
            loc = torch.tensor([500 + step], dtype=torch.int32, device=self.dev)
            kept_append(slots=slots, loc=loc, kept_stack=kept, klen_stack=klen)
            self.assertEqual(int(klen[0, 1]), step + 1)
        self.assertEqual(
            kept[0, 1, :12].tolist(), [500 + s for s in range(12)]
        )


if __name__ == "__main__":
    unittest.main()
