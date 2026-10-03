"""The index-k cache is paged with a split key and scale block, not interleaved.

`index_buf_accessor` is the layout's definition: GetK slices
``[: page_size * index_head_dim]`` and GetS slices
``[page_size * index_head_dim :]``, so within a page the keys come first and
the scales follow. A flat ``[n_slots, index_head_dim + 4]`` view does not give
that order, and this module took one for a while.

What it cost: tier 1 ranks rows by sigma over these keys and its top-k IS the
keep decision, so on a rope-less MLA the kept set was chosen from key bytes
read as scales. A written page's dequantised key norms came back inf where the
correct read gives about 16, and the "scale" read 6.2e7 against the true
constant 0.015625. Nothing in a fetch count or a latency shows that.

Size cannot catch it: a page of 8448 bytes is both 64 x 132 and
64 x 128 + 64 x 4, and the aiter accessor really does use the interleaved
view. So these build a buffer in the canonical layout and check the values,
not the shapes.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

PAGE, DIM = 64, 128


def _canonical_buffer(num_pages, keys_f32, scales):
    """[num_pages, PAGE*DIM + PAGE*4] uint8, keys block then scale block."""
    buf = torch.zeros(num_pages, PAGE * DIM + PAGE * 4, dtype=torch.uint8,
                      device="cuda")
    k8 = keys_f32.to(torch.float8_e4m3fn).view(torch.uint8)      # [P, PAGE, DIM]
    buf[:, : PAGE * DIM] = k8.reshape(num_pages, PAGE * DIM)
    buf[:, PAGE * DIM :] = scales.contiguous().view(torch.uint8).reshape(
        num_pages, PAGE * 4
    )
    return buf


class TestIndexViewLayout(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.P = 3
        # fp8-representable values so the round trip is exact
        self.keys = (
            torch.randint(-8, 9, (self.P, PAGE, DIM), device="cuda").float() / 8.0
        )
        self.scales = torch.full((self.P, PAGE), 0.015625, device="cuda")
        self.buf = _canonical_buffer(self.P, self.keys, self.scales)

    def test_rows_match_the_canonical_accessor_arithmetic(self):
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import index_rows

        slots = torch.tensor([0, 1, 63, 64, 65, 127, 128, 191], device="cuda")
        got = index_rows(
            self.buf, slots, index_head_dim=DIM, quant_block_size=DIM,
            slots_per_page=PAGE,
        )
        p, t = slots // PAGE, slots % PAGE
        want = self.keys[p, t] * self.scales[p, t][:, None]
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_the_interleaved_read_disagrees(self):
        """The bug, as a measurement: the old view is not merely different.

        If this ever passes, the two layouts have converged and the fix is
        moot -- which would be worth knowing.
        """
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import index_rows

        slots = torch.arange(PAGE * self.P, device="cuda")
        good = index_rows(
            self.buf, slots, index_head_dim=DIM, quant_block_size=DIM,
            slots_per_page=PAGE,
        )
        flat = self.buf.reshape(-1, DIM + 4)
        bad_k = flat.view(torch.float8_e4m3fn)[:, :DIM].float()
        bad_s = flat.view(torch.float32)[:, DIM // 4]
        bad = bad_k * bad_s[:, None]
        self.assertFalse(
            torch.allclose(good[: bad.shape[0]], bad[: good.shape[0]]),
            "the interleaved read agrees with the paged one; the layouts have "
            "converged and this guard is obsolete",
        )
        # The interleaved read's "scale" is whatever key bytes sit at that
        # offset. On the real pool those decode to 6.2e7; on a synthetic
        # buffer of small fp8 values they decode small, so the magnitude is
        # not the invariant -- that it is NOT the stored scale is.
        self.assertFalse(
            torch.allclose(bad_s, self.scales.reshape(-1)[: bad_s.numel()]),
            "the interleaved column happened to read the stored scale; this "
            "buffer cannot distinguish the layouts",
        )

    def test_a_wrong_page_size_is_refused_not_silently_read(self):
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import (
            index_page_views,
        )

        with self.assertRaises(AssertionError):
            index_page_views(
                self.buf, index_head_dim=DIM, quant_block_size=DIM, slots_per_page=32
            )

    def test_sigma_is_finite_on_a_canonical_buffer(self):
        """The symptom that was visible and unexamined: inf key norms."""
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import index_sigma

        slots = torch.arange(PAGE * self.P, device="cuda")
        sig = index_sigma(
            buf=self.buf, slots=slots, index_head_dim=DIM, quant_block_size=DIM,
            slots_per_page=PAGE, block=PAGE,
        )
        self.assertTrue(bool(torch.isfinite(sig).all()), f"sigma not finite: {sig}")
        # One sigma per TOKEN, not per block: "blockwise" is the transform
        # window. _advance_sigma concatenates these and takes topk over them
        # as the keep decision, which only type-checks per token.
        self.assertEqual(int(sig.numel()), PAGE * self.P)


if __name__ == "__main__":
    unittest.main(verbosity=2)


POOL = 4


class TestPooledGroupRead(CustomTestCase):
    """Reaching a token's salience when the indexer pools 4:1.

    `_kpool_decode_update_and_maybe_write_cache_kernel` writes one entry per
    group and addresses it as

        pool_id        = pos // pool_size
        token_page_row = (pool_id // slots_per_page) * pool_size
        page           = block_tables[req, token_page_row]
        offset         = pool_id % slots_per_page

    so one index page holds slots_per_page * pool_size tokens' worth of entries
    and three of every four token pages hold none. These build a buffer by that
    rule and check the read finds it -- the per-token read cannot, which is the
    second half of the index-k defect.
    """

    def setUp(self):
        torch.manual_seed(1)
        self.n_groups = 3 * PAGE                 # 3 index pages' worth
        self.pages = 4 * 3 + 1                   # token pages, 4 per index page
        self.buf = torch.zeros(
            self.pages, PAGE * DIM + PAGE * 4, dtype=torch.uint8, device="cuda"
        )
        # block table: token page j -> some scattered physical page
        g = torch.Generator(device="cuda").manual_seed(2)
        self.block_row = torch.randperm(
            self.pages, device="cuda", generator=g
        )
        self.truth = torch.zeros(self.n_groups, DIM, device="cuda")
        scale = 0.015625
        for pid in range(self.n_groups):
            page = int(self.block_row[(pid // PAGE) * POOL])
            off = pid % PAGE
            v = (torch.randint(-8, 9, (DIM,), device="cuda").float() / 8.0)
            k8 = v.to(torch.float8_e4m3fn).view(torch.uint8)
            self.buf[page, off * DIM : (off + 1) * DIM] = k8
            sb = PAGE * DIM + off * 4
            self.buf[page, sb : sb + 4] = torch.tensor(
                [scale], device="cuda"
            ).view(torch.uint8)
            self.truth[pid] = v.to(torch.float8_e4m3fn).float() * scale

    def _read(self, pool_ids):
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import (
            index_group_keys,
        )

        return index_group_keys(
            self.buf, pool_ids, self.block_row,
            index_head_dim=DIM, slots_per_page=PAGE, pool_size=POOL,
        )

    def test_the_group_read_finds_what_the_writer_stored(self):
        pool_ids = torch.arange(self.n_groups, device="cuda")
        torch.testing.assert_close(self._read(pool_ids), self.truth,
                                   rtol=0, atol=0)

    def test_it_crosses_index_page_boundaries(self):
        """The `* pool_size` in the block-table index is the easy thing to drop."""
        pool_ids = torch.tensor(
            [PAGE - 1, PAGE, PAGE + 1, 2 * PAGE - 1, 2 * PAGE], device="cuda"
        )
        torch.testing.assert_close(self._read(pool_ids),
                                   self.truth[pool_ids.cpu()], rtol=0, atol=0)

    def test_the_per_token_read_finds_mostly_nothing(self):
        """The symptom that led here: 3 of 4 token-slot reads are zero."""
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import index_rows

        # token slots as the old path used them: positions, not pool ids
        slots = torch.arange(self.n_groups * POOL, device="cuda")
        got = index_rows(
            self.buf, slots, index_head_dim=DIM, quant_block_size=DIM,
            slots_per_page=PAGE,
        )
        zero_frac = float((got.norm(dim=1) == 0).float().mean())
        self.assertGreater(
            zero_frac, 0.5,
            f"only {zero_frac:.2f} of per-token reads were zero; if this "
            "buffer is readable per token the pooling assumption is wrong",
        )


class TestGroupSigmaWindow(CustomTestCase):
    def test_the_physical_cutoff_does_not_move(self):
        """kappa counts bins, so the same token span keeps the same cutoff.

        4096 tokens at kappa 16 cuts below 4096/16 = 256 tokens. 1024 groups of
        4 at kappa 16 cuts below 1024/16 = 64 groups, which is the same 256
        tokens. If this ever needs a kappa rescale, the window rule is wrong.
        """
        from sglang.srt.layers.attention.vestigekv import defaults as D
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import (
            group_sigma_window,
        )

        w = group_sigma_window(D.CLOSE_BLOCK, POOL)
        self.assertEqual(w, D.CLOSE_BLOCK // POOL)
        tokens_per_bin_flat = D.CLOSE_BLOCK / D.LOWPASS_KAPPA
        tokens_per_bin_grouped = (w / D.LOWPASS_KAPPA) * POOL
        self.assertAlmostEqual(tokens_per_bin_flat, tokens_per_bin_grouped,
                               places=9)

    def test_a_window_that_is_not_a_multiple_of_the_pool_is_refused(self):
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import (
            group_sigma_window,
        )

        with self.assertRaises(AssertionError):
            group_sigma_window(4094, POOL)


class TestGroupSelector(CustomTestCase):
    """Tier 2 as a fixed budget over groups, in DSA's units.

    archive_pool.py's conclusion was that pooling "requires tier 2 to change
    its fire rule to a fixed budget first", because a threshold with a sound
    bound cannot survive pooling. These pin the budget arithmetic and the
    per-layer reduction, which are the two places the parity with DSA lives:
    index_topk = 2048 ROWS is 512 groups of 4, and DSA selects one row set per
    layer rather than a per-head union.
    """

    def test_the_budget_is_stated_in_rows_and_spent_in_groups(self):
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import select_groups

        scores = torch.randn(4096, device="cuda")
        sel = select_groups(scores, 2048, POOL)
        self.assertEqual(sel.numel(), 2048 // POOL)
        self.assertEqual(sel.numel(), 512)
        # and it really is the top, not an arbitrary 512
        want = scores.topk(512).indices.sort().values
        torch.testing.assert_close(sel.sort().values, want)

    def test_a_budget_that_is_not_a_multiple_of_the_pool_is_refused(self):
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import select_groups

        with self.assertRaises(AssertionError):
            select_groups(torch.randn(64, device="cuda"), 2047, POOL)

    def test_fewer_groups_than_budget_is_not_an_error(self):
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import select_groups

        sel = select_groups(torch.randn(10, device="cuda"), 2048, POOL)
        self.assertEqual(sel.numel(), 10)

    def test_the_layer_score_is_best_over_heads(self):
        """One set per layer: a group any head wants is a group the layer wants."""
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import group_scores

        G, H = 32, 8
        keys = torch.randn(G, DIM, device="cuda")
        q = torch.randn(H, DIM, device="cuda")
        got = group_scores(keys, q)
        want = (q.float() @ keys.T.float()).amax(dim=0)
        torch.testing.assert_close(got, want)
        self.assertEqual(tuple(got.shape), (G,))

    def test_a_head_gate_weights_rather_than_tie_breaks(self):
        from sglang.srt.layers.attention.vestigekv.dsa_index_view import group_scores

        G, H = 16, 4
        keys = torch.randn(G, DIM, device="cuda")
        q = torch.randn(H, DIM, device="cuda")
        w = torch.rand(H, device="cuda")
        got = group_scores(keys, q, head_weights=w)
        want = ((q.float() @ keys.T.float()) * w.reshape(-1, 1)).sum(dim=0)
        torch.testing.assert_close(got, want)
        # a gate is not a max: zeroing all but one head must leave that head's row
        w0 = torch.zeros(H, device="cuda")
        w0[2] = 1.0
        torch.testing.assert_close(
            group_scores(keys, q, head_weights=w0),
            (q[2].float() @ keys.T.float()),
        )
