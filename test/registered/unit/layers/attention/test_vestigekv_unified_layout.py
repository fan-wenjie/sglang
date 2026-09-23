"""The appended record block must not disturb DSA's offsets, and rho must not
drift from the basis it was measured against.

Derived property, not a mirror: the layout's whole claim is that VestigeKV's
fields can ride in DSA's page row without moving anything DSA's own accessors
compute, so what is pinned is the offset arithmetic and the aliasing. The
generation check is the other half -- a sketch and a residual written under a
retired basis still scan and still calibrate, and the bound they produce is
wrong in a way no downstream test can see.
"""

import unittest

import torch

from sglang.srt.layers.attention.vestigekv.dsa_unified.layout import (
    page_row_bytes,
    slot_index,
    scatter_sketch,
    sketch_block_bytes,
    sketch_views,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE, DIM, QBLK, SK, PAGES = 64, 128, 128, 64, 4


def _buf():
    dsa, total = page_row_bytes(PAGE, DIM, QBLK, SK)
    return torch.zeros(PAGES, total, dtype=torch.uint8), dsa, total


class TestOffsets(CustomTestCase):
    def test_dsa_block_keeps_its_size_and_starts_at_zero(self):
        dsa, total = page_row_bytes(PAGE, DIM, QBLK, SK)
        self.assertEqual(dsa, PAGE * (DIM + DIM // QBLK * 4))
        self.assertEqual(total - dsa, PAGE * sketch_block_bytes(SK))

    def test_a_wider_sketch_only_grows_the_appended_block(self):
        d1, t1 = page_row_bytes(PAGE, DIM, QBLK, 64)
        d2, t2 = page_row_bytes(PAGE, DIM, QBLK, 128)
        self.assertEqual(d1, d2)
        self.assertGreater(t2, t1)


class TestViews(CustomTestCase):
    def test_a_pool_slot_indexes_the_views_directly(self):
        buf, dsa, _ = _buf()
        sk, rho = sketch_views(buf, page_size=PAGE, index_head_dim=DIM,
                               quant_block_size=QBLK, sketch_dim=SK)
        self.assertEqual(tuple(sk.shape), (PAGES, PAGE, SK))
        self.assertEqual(tuple(rho.shape), (PAGES, PAGE))
        # slot 130 is page 2 row 2; the split must land there and nowhere else
        pg, rw = slot_index(torch.tensor([130]), PAGE)
        self.assertEqual((int(pg[0]), int(rw[0])), (2, 2))
        rho[pg, rw] = 7.5
        self.assertAlmostEqual(float(rho[2, 2]), 7.5)
        self.assertAlmostEqual(float(rho[2, 3]), 0.0)

    def test_the_views_alias_and_do_not_copy(self):
        buf, _, _ = _buf()
        sk, rho = sketch_views(buf, page_size=PAGE, index_head_dim=DIM,
                               quant_block_size=QBLK, sketch_dim=SK)
        self.assertEqual(sk.untyped_storage().data_ptr(), buf.untyped_storage().data_ptr())
        self.assertEqual(rho.untyped_storage().data_ptr(), buf.untyped_storage().data_ptr())

    def test_writing_the_sketch_leaves_dsa_bytes_untouched(self):
        buf, dsa, _ = _buf()
        buf[:, :dsa] = 0xAB
        before = buf[:, :dsa].clone()
        sk, rho = sketch_views(buf, page_size=PAGE, index_head_dim=DIM,
                               quant_block_size=QBLK, sketch_dim=SK)
        sk[...] = torch.full_like(sk, 1.0)
        rho[...] = 3.0
        self.assertTrue(torch.equal(buf[:, :dsa], before))

    def test_a_stale_row_width_is_refused_not_reinterpreted(self):
        short = torch.zeros(PAGES, PAGE * DIM, dtype=torch.uint8)
        with self.assertRaises(AssertionError):
            sketch_views(short, page_size=PAGE, index_head_dim=DIM,
                         quant_block_size=QBLK, sketch_dim=SK)


class TestBasisGeneration(CustomTestCase):
    def _args(self, buf, gen, rec):
        slots = torch.tensor([3, 70, 140])
        return dict(buf=buf, slots=slots,
                    sketch=torch.ones(3, SK, dtype=torch.float8_e4m3fn),
                    rho=torch.ones(3, dtype=torch.float32),
                    page_size=PAGE, index_head_dim=DIM, quant_block_size=QBLK,
                    basis_gen=gen, record_gen=rec)

    def test_a_first_write_and_a_same_generation_rewrite_are_allowed(self):
        buf, _, _ = _buf()
        rec = torch.zeros(PAGES * PAGE, dtype=torch.int32)
        scatter_sketch(**self._args(buf, 1, rec))
        scatter_sketch(**self._args(buf, 1, rec))
        self.assertEqual(int(rec[3]), 1)

    def test_rho_must_stay_fp32(self):
        buf, _, _ = _buf()
        rec = torch.zeros(PAGES * PAGE, dtype=torch.int32)
        a = self._args(buf, 1, rec)
        a["rho"] = a["rho"].half()
        with self.assertRaises(AssertionError):
            scatter_sketch(**a)

    def test_the_generation_is_recorded_so_a_refit_can_be_detected(self):
        buf, _, _ = _buf()
        rec = torch.zeros(PAGES * PAGE, dtype=torch.int32)
        scatter_sketch(**self._args(buf, 4, rec))
        self.assertEqual([int(rec[i]) for i in (3, 70, 140)], [4, 4, 4])
        self.assertEqual(int(rec[5]), 0)


class TestInterleaved(CustomTestCase):
    """The interleaved layout's claim is that a flat slot id addresses it."""

    def _ibuf(self):
        from sglang.srt.layers.attention.vestigekv.dsa_unified.layout import (
            interleaved_token_bytes,
        )
        rec = interleaved_token_bytes(DIM, QBLK, SK)
        return torch.zeros(PAGES, PAGE * rec, dtype=torch.uint8), rec

    def test_a_record_holds_both_fields_and_costs_their_sum(self):
        from sglang.srt.layers.attention.vestigekv.dsa_unified.layout import (
            dsa_token_bytes, interleaved_token_bytes, sketch_block_bytes,
        )
        self.assertEqual(
            interleaved_token_bytes(DIM, QBLK, SK),
            dsa_token_bytes(DIM, QBLK) + sketch_block_bytes(SK),
        )

    def test_a_flat_slot_id_addresses_it_with_no_page_split(self):
        from sglang.srt.layers.attention.vestigekv.dsa_unified.layout import (
            interleaved_views,
        )
        buf, _ = self._ibuf()
        key, ks, sk, rho = interleaved_views(
            buf, page_size=PAGE, index_head_dim=DIM, quant_block_size=QBLK, sketch_dim=SK
        )
        self.assertEqual(key.shape[0], PAGES * PAGE)
        rho[130] = 2.5
        ks[130] = 9.0
        self.assertAlmostEqual(float(rho[130]), 2.5)
        self.assertAlmostEqual(float(ks[130]), 9.0)
        self.assertAlmostEqual(float(rho[131]), 0.0)

    def test_every_field_aliases_the_one_buffer(self):
        from sglang.srt.layers.attention.vestigekv.dsa_unified.layout import (
            interleaved_views,
        )
        buf, _ = self._ibuf()
        p = buf.untyped_storage().data_ptr()
        for t in interleaved_views(buf, page_size=PAGE, index_head_dim=DIM,
                                   quant_block_size=QBLK, sketch_dim=SK):
            self.assertEqual(t.untyped_storage().data_ptr(), p)

    def test_the_two_layouts_cost_the_same_bytes(self):
        from sglang.srt.layers.attention.vestigekv.dsa_unified.layout import (
            interleaved_token_bytes, page_row_bytes,
        )
        _, appended_total = page_row_bytes(PAGE, DIM, QBLK, SK)
        self.assertEqual(appended_total, PAGE * interleaved_token_bytes(DIM, QBLK, SK))


if __name__ == "__main__":
    unittest.main()
