"""The slot index is cached on slot NUMBERS, which is what makes reuse safe.

Building it is a host-to-device copy of a Python list, paid once per layer per step for a tensor
that has not changed -- 48 times a token on this model.

What makes the cache safe is what it is keyed on. A slot handed from one request to the next has
the same number, so the same tensor is the right tensor; nothing per-request is held. Keyed on
request ids instead it would be a per-request table on a side that is supposed to hold none, and
keyed on nothing it would hand one batch another's rows.

Capped because the key space is ARRANGEMENTS of slots: a batch that changes shape every step would
add an entry every step forever.
"""

import torch

from sglang.srt.afd import linear_history as lh
from sglang.test.test_utils import CustomTestCase


class TestTheSlotIndexCache(CustomTestCase):
    def setUp(self):
        lh._SLOT_INDEX.clear()

    def test_the_same_slots_give_the_same_tensor(self):
        a = lh._slot_index([2, 0, 1], torch.device("cpu"))
        b = lh._slot_index([2, 0, 1], torch.device("cpu"))
        self.assertIs(a, b, "the index was rebuilt for slots it already had")

    def test_the_value_is_the_slots_in_order(self):
        got = lh._slot_index([3, 1, 2], torch.device("cpu"))
        self.assertEqual(got.tolist(), [3, 1, 2])

    def test_a_different_order_is_a_different_tensor(self):
        """Order carries which row belongs to which request. A cache that ignored it would hand
        one request another's history, fluently."""
        a = lh._slot_index([0, 1], torch.device("cpu"))
        b = lh._slot_index([1, 0], torch.device("cpu"))
        self.assertNotEqual(a.tolist(), b.tolist())

    def test_the_two_dtypes_do_not_collide(self):
        """The state kernels take int32 and index_select takes int64; one entry for both would
        hand a kernel the wrong dtype."""
        a = lh._slot_index([0, 1], torch.device("cpu"))
        b = lh._row_index([0, 1], torch.device("cpu"))
        self.assertEqual(a.dtype, torch.long)
        self.assertEqual(b.dtype, torch.int32)

    def test_it_does_not_grow_without_limit(self):
        for i in range(lh._SLOT_INDEX_CAP + 20):
            lh._slot_index([i], torch.device("cpu"))
        self.assertLessEqual(
            len(lh._SLOT_INDEX),
            lh._SLOT_INDEX_CAP,
            "the cache grew past its cap; a batch that changes shape every step would hold an "
            "entry for every step it ever ran",
        )
