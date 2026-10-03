"""The index-k cache can stand in for the salience ring, byte for byte.

Derived property, not a mirror: the ring held a second copy of DSA's indexer
key, quantised by the same act_quant math, so replacing it is only safe if a
view over the cache reproduces the ring's (key, scale) pair for every slot.

This file used to assert that a global slot id indexes a flat [n_slots, dim+4]
view directly. That was the layout bug, not a property: within a page the keys
come first as a block and the scales follow, so the flat view reads key bytes
as scales. The layout itself, and that the interleaved read disagrees with it,
are pinned in kernels/ops/attention/test_vestigekv_index_view.py. What is left
here is what that file does not cover: the views alias, and a buffer sigma
cannot address is refused rather than silently misread.
"""

import unittest

import torch

from sglang.srt.layers.attention.vestigekv.dsa_index_view import index_page_views
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DIM, QBLK, SLOTS, PAGES = 128, 128, 64, 5


def _buffer():
    """A page-major index-k buffer: per page, the key block then the scales."""
    keys = (
        (torch.arange(PAGES * SLOTS * DIM, dtype=torch.float32) % 7 - 3)
        .to(torch.float8_e4m3fn)
        .view(PAGES, SLOTS, DIM)
    )
    scale = (torch.arange(PAGES * SLOTS, dtype=torch.float32) + 1).view(
        PAGES, SLOTS
    ) * 0.125
    buf = torch.empty(PAGES, SLOTS * DIM + SLOTS * 4, dtype=torch.uint8)
    buf[:, : SLOTS * DIM] = keys.reshape(PAGES, SLOTS * DIM).view(torch.uint8)
    buf[:, SLOTS * DIM :] = scale.contiguous().view(torch.uint8).view(PAGES, SLOTS * 4)
    return buf, keys, scale


class TestIndexPageViews(CustomTestCase):
    def test_the_views_reproduce_the_stored_pair(self):
        buf, keys, scale = _buffer()
        got_k, got_s = index_page_views(
            buf, index_head_dim=DIM, quant_block_size=QBLK, slots_per_page=SLOTS
        )
        self.assertEqual(tuple(got_k.shape), (PAGES, SLOTS, DIM))
        self.assertEqual(tuple(got_s.shape), (PAGES, SLOTS))
        self.assertTrue(torch.equal(got_k.float(), keys.float()))
        self.assertTrue(torch.equal(got_s, scale))

    def test_the_views_alias_and_do_not_copy(self):
        buf, _, _ = _buffer()
        got_k, got_s = index_page_views(
            buf, index_head_dim=DIM, quant_block_size=QBLK, slots_per_page=SLOTS
        )
        self.assertEqual(got_k.untyped_storage().data_ptr(),
                         buf.untyped_storage().data_ptr())
        self.assertEqual(got_s.untyped_storage().data_ptr(),
                         buf.untyped_storage().data_ptr())

    def test_rejects_a_geometry_sigma_cannot_score(self):
        # Two scales per row is a different statistic, not a reshape away.
        buf, _, _ = _buffer()
        with self.assertRaises(AssertionError):
            index_page_views(
                buf, index_head_dim=DIM, quant_block_size=QBLK // 2,
                slots_per_page=SLOTS,
            )

    def test_rejects_a_page_whose_width_is_not_the_layout(self):
        buf, _, _ = _buffer()
        with self.assertRaises(AssertionError):
            index_page_views(
                buf, index_head_dim=DIM, quant_block_size=QBLK,
                slots_per_page=SLOTS // 2,
            )

    def test_rejects_a_non_byte_buffer(self):
        with self.assertRaises(AssertionError):
            index_page_views(
                torch.zeros(PAGES, SLOTS * DIM + SLOTS * 4, dtype=torch.float32),
                index_head_dim=DIM, quant_block_size=QBLK, slots_per_page=SLOTS,
            )


if __name__ == "__main__":
    unittest.main()
