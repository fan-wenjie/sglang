"""The index-k cache can stand in for the salience ring, byte for byte.

Derived property, not a mirror: the ring held a second copy of DSA's indexer
key, quantised by the same act_quant math, so replacing it is only safe if a
view over the page-major cache reproduces the ring's (key, scale) pair for
every slot -- including that a global slot id indexes the flat view directly,
which is the layout fact nothing in the vestigekv package can show.
"""

import unittest

import torch

from sglang.srt.layers.attention.vestigekv.dsa_index_view import index_key_views
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DIM, QBLK, PAGE, PAGES = 128, 128, 64, 5
ROW = DIM + 4


def _buffer():
    """A page-major index-k buffer with a known key and scale per slot."""
    n = PAGES * PAGE
    keys = (torch.arange(n * DIM, dtype=torch.float32) % 7 - 3).to(
        torch.float8_e4m3fn
    ).view(n, DIM)
    scale = (torch.arange(n, dtype=torch.float32) + 1) * 0.125
    buf = torch.empty(n, ROW, dtype=torch.uint8)
    buf[:, :DIM] = keys.view(torch.uint8)
    buf[:, DIM:] = scale.view(torch.float32).unsqueeze(1).view(torch.uint8).view(n, 4)
    return buf.view(PAGES, PAGE * ROW), keys, scale


class TestIndexKeyViews(CustomTestCase):
    def test_a_global_slot_id_indexes_the_flat_view_directly(self):
        buf, keys, scale = _buffer()
        got_k, got_s = index_key_views(buf, index_head_dim=DIM, quant_block_size=QBLK)
        # slot 130 is page 2, row 2 -- the caller passes 130, not (2, 2)
        for slot in (0, 1, 63, 64, 130, PAGES * PAGE - 1):
            self.assertTrue(
                torch.equal(
                    got_k[slot, :DIM].view(torch.uint8), keys[slot].view(torch.uint8)
                ),
                f"key mismatch at slot {slot}",
            )
            self.assertEqual(float(got_s[slot]), float(scale[slot]))

    def test_the_views_alias_and_do_not_copy(self):
        buf, _, _ = _buffer()
        got_k, got_s = index_key_views(buf, index_head_dim=DIM, quant_block_size=QBLK)
        self.assertEqual(got_k.data_ptr(), buf.data_ptr())
        self.assertEqual(got_s.untyped_storage().data_ptr(), buf.untyped_storage().data_ptr())

    def test_rejects_a_geometry_sigma_cannot_score(self):
        buf, _, _ = _buffer()
        # two scales per row: sigma takes one per row, so this must not be
        # silently read as if the first scale covered the whole key
        with self.assertRaises(AssertionError):
            index_key_views(buf, index_head_dim=DIM, quant_block_size=QBLK // 2)

    def test_rejects_a_non_byte_buffer(self):
        with self.assertRaises(AssertionError):
            index_key_views(
                torch.zeros(PAGES, PAGE * ROW, dtype=torch.float32),
                index_head_dim=DIM,
                quant_block_size=QBLK,
            )


if __name__ == "__main__":
    unittest.main()
