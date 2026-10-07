"""D1: VestigeKV's selection inside DSA's own transform kernel.

Each case guards a failure that no other case here would catch:

  threshold parity      the forked rule must be the SAME rule the served arm
                        already scored 1.000 on multikey with. D1 is a cost
                        change; if the threshold moves, it is not.
  emitted set           kept u fired u tail, in DSA's dst_token_indices layout.
  OVERFLOW -> DSA       a quantile admits however many clear it, so the budget
                        can be exceeded where DSA's rank rule cannot. The
                        fallback must be DSA's exact top-k, bit for bit, and it
                        must be WITNESSED -- a silent row drop reproduces every
                        byte and latency counter (see the trash-slot incident).
  no double emit        a kept group that also clears the threshold is seeded
                        once, not twice; the km mask is what prevents it.
  padding and tail      everything past the selection is -1, and the unpooled
                        tail sits directly after it.
  short row             length <= K keeps DSA's early path, which selects
                        nothing because there is nothing to select.
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


def _skip():
    return not torch.cuda.is_available()


class TestVkSelectTransform(CustomTestCase):
    POOL = 4
    GROUP_TOPK = 512
    TOPK = GROUP_TOPK * POOL

    def setUp(self):
        if _skip():
            self.skipTest("CUDA required")
        torch.manual_seed(0)
        self.dev = torch.device("cuda")

    # ---- helpers ---------------------------------------------------------

    def _scores_and_kept(self, *, bs, G, n_groups):
        """Scores in the regime the rule actually runs in, and tier 1's table.

        The kept groups must be the HIGH-scoring ones. That is not cosmetic: the
        threshold is a quantile OF them, so if they were a representative sample
        the quantile would land mid-distribution and admit a quarter of every
        group on the row. With iid scores this test fired 1800 of 4096 and sat
        in the fallback on every call. In the served arm sigma captures 84% of
        DSA's in-budget mass with 0.78% of the rows, and ~98 of 32768 groups
        fire -- so elevated kept scores ARE the realistic case, and a test in
        the other regime measures a configuration that never runs.
        """
        logits = torch.randn(bs, G, device=self.dev, dtype=torch.float32)
        capk = n_groups * self.POOL
        pos = torch.zeros(bs, capk, dtype=torch.int32, device=self.dev)
        for b in range(bs):
            gs = torch.randperm(G, device=self.dev)[:n_groups].sort().values
            logits[b, gs] += 6.0  # clear of the bulk, as the spectral pick is
            rows = (gs[:, None] * self.POOL + torch.arange(
                self.POOL, device=self.dev)[None, :]).reshape(-1)
            pos[b, : rows.numel()] = rows.to(torch.int32)
        klen = torch.full((bs,), capk, dtype=torch.int32, device=self.dev)
        return logits, pos, klen

    def _kept(self, *, bs, G, n_groups, dtype=torch.int32):
        _, pos, klen = self._scores_and_kept(bs=bs, G=G, n_groups=n_groups)
        return pos.to(dtype), klen

    def _thr_and_scratch(self, logits, pos, klen, q, kg_max):
        from sglang.srt.layers.attention.vestigekv.dsa.select import (
            kept_threshold,
            select_scratch,
        )

        bs, G = logits.shape
        sc = select_scratch(bs=bs, G=G, kg_max=kg_max, dev=self.dev)
        slots = torch.arange(bs, dtype=torch.int32, device=self.dev)
        kept_threshold(
            logits=logits, kept_pos=pos, klen=klen, slots=slots, q=q,
            pool=self.POOL, scratch=sc, kg_max=kg_max,
        )
        return sc

    def _run(self, logits, lengths, sc, seq_lens=None):
        from sglang.kernels.ops.attention.dsa.vk_select_transform import (
            vk_select_transform,
        )

        return vk_select_transform(
            score=logits, lengths=lengths, thresholds=sc["thr"],
            pool_size=self.POOL, topk=self.TOPK,
            kept_groups=sc["kg"], kept_lens=sc["kgn"], kept_mask=sc["km"],
            ovf_counts=sc["ovf"], seq_lens=seq_lens,
        )

    # ---- cases -----------------------------------------------------------

    def test_the_threshold_matches_the_served_rule(self):
        """Same bin, same lower edge, as _branch_hist + _branch_hit give."""
        from sglang.srt.layers.attention.vestigekv.dsa import branch as B

        bs, G, ng, q = 2, 4096, 48, 0.75
        logits = torch.randn(bs, G, device=self.dev, dtype=torch.float32)
        pos, klen = self._kept(bs=bs, G=G, n_groups=ng)
        sc = self._thr_and_scratch(logits, pos, klen, q, kg_max=ng)

        # the served path, verbatim
        old = B.branch_scratch(bs=bs, G=G * self.POOL, nslot=bs,
                               fetch_w=self.TOPK, pool=self.POOL,
                               fence_groups=None, dev=self.dev)
        slots = torch.arange(bs, dtype=torch.int32, device=self.dev)
        import triton
        B._branch_kept_kernel[(triton.cdiv(klen[0].item(), 64), bs)](
            pos, klen, slots, logits, old["km"], old["range"],
            pos.shape[1], G, POOL=self.POOL, BK=64, num_warps=4,
        )
        NG = triton.cdiv(G, B.GBLOCK)
        B._branch_hist_kernel[(NG, bs)](
            logits, old["km"], old["range"], old["hist"], G,
            BLOCK=B.GBLOCK, NB=B.NBIN, num_warps=4,
        )
        plen = torch.full((bs,), G, dtype=torch.int32, device=self.dev)
        B._branch_hit_kernel[(NG, bs)](
            logits, old["km"], old["hist"], old["range"], plen,
            old["smax"], old["smaxn"], old["hit"], old["counts"], old["thr"],
            float(q), G * self.POOL, G, bs, BLOCK=B.GBLOCK,
            BUCKET=B.GBLOCK, NB=B.NBIN, NC=B.NCODE, num_warps=4,
        )
        torch.testing.assert_close(sc["thr"], old["thr"], rtol=0, atol=1e-6)

    def test_overflow_falls_back_to_dsa_bit_for_bit(self):
        """A threshold that admits more than the budget must BE DSA, and say so."""
        from sglang.kernels.ops.attention.dsa.kpool_topk_transform import (
            fast_kpool_topk_transform_fused,
        )

        # G > kCandMax (4096) so the -inf threshold spills into DSA's radix,
        # which is the path this case is about. At G == kCandMax it would take
        # the on-candidate select instead, exact only to a bin.
        bs, G, ng = 1, 8192, 8
        logits = torch.randn(bs, G, device=self.dev, dtype=torch.float32)
        pos, klen = self._kept(bs=bs, G=G, n_groups=ng)
        sc = self._thr_and_scratch(logits, pos, klen, 0.75, kg_max=ng)
        # Drive the threshold below everything: every group clears it, so the
        # fired count is G >> 512 and the lane must overflow.
        sc["thr"].fill_(float("-inf"))
        lengths = torch.full((bs,), G, dtype=torch.int32, device=self.dev)

        got = self._run(logits, lengths, sc)
        want = fast_kpool_topk_transform_fused(
            score=logits, lengths=lengths, pool_size=self.POOL, topk=self.TOPK,
        )
        self.assertGreater(int(sc["ovf"].sum()), 0, "overflow was not witnessed")
        # As SETS, not elementwise: DSA's own kernel emits through
        # atomicAdd(&s_counter, 1), so its selection order is a race and two
        # runs of DSA need not agree position by position either.
        self.assertEqual(set(got[0].tolist()), set(want[0].tolist()))

    def test_a_non_overflowing_row_emits_kept_union_fired(self):
        bs, G, ng, q = 1, 4096, 32, 0.75
        logits, pos, klen = self._scores_and_kept(bs=bs, G=G, n_groups=ng)
        sc = self._thr_and_scratch(logits, pos, klen, q, kg_max=ng)
        lengths = torch.full((bs,), G, dtype=torch.int32, device=self.dev)
        out = self._run(logits, lengths, sc)
        self.assertEqual(int(sc["ovf"].sum()), 0, "fired past the budget: the "
                         "kept set is not separated from the bulk")

        kept = set(sc["kg"][0, : int(sc["kgn"][0])].tolist())
        fired = set(
            (torch.nonzero(logits[0] > sc["thr"][0]).flatten().tolist())
        ) - kept
        want_groups = kept | fired
        emitted = out[0][out[0] >= 0].tolist()
        got_groups = {t // self.POOL for t in emitted}
        self.assertEqual(got_groups, want_groups)
        # POOL contiguous rows per selected group, and no duplicates.
        self.assertEqual(len(emitted), len(set(emitted)))
        self.assertEqual(len(emitted), self.POOL * len(want_groups))

    def test_the_rest_of_the_row_is_minus_one(self):
        bs, G, ng = 1, 4096, 32
        logits = torch.randn(bs, G, device=self.dev, dtype=torch.float32)
        pos, klen = self._kept(bs=bs, G=G, n_groups=ng)
        sc = self._thr_and_scratch(logits, pos, klen, 0.75, kg_max=ng)
        lengths = torch.full((bs,), G, dtype=torch.int32, device=self.dev)
        out = self._run(logits, lengths, sc)
        row = out[0]
        n_valid = int((row >= 0).sum())
        self.assertTrue(bool((row[:n_valid] >= 0).all()), "selection is not a prefix")
        self.assertTrue(bool((row[n_valid:] == -1).all()), "padding is not -1")

    def test_the_unpooled_tail_sits_after_the_selection(self):
        bs, G, ng = 1, 4096, 32
        logits = torch.randn(bs, G, device=self.dev, dtype=torch.float32)
        pos, klen = self._kept(bs=bs, G=G, n_groups=ng)
        sc = self._thr_and_scratch(logits, pos, klen, 0.75, kg_max=ng)
        lengths = torch.full((bs,), G, dtype=torch.int32, device=self.dev)
        # seq_lens % pool = 3 unpooled tokens past the pooled history.
        seq = torch.full((bs,), G * self.POOL + 3, dtype=torch.int32,
                         device=self.dev)
        out = self._run(logits, lengths, sc, seq_lens=seq)
        row = out[0]
        n_valid = int((row >= 0).sum())
        tail = row[n_valid - 3 : n_valid].tolist()
        self.assertEqual(tail, [G * self.POOL, G * self.POOL + 1,
                                G * self.POOL + 2])

    def test_a_short_row_keeps_dsas_early_path(self):
        """length <= K: there is nothing to select, so everything is attended."""
        from sglang.kernels.ops.attention.dsa.kpool_topk_transform import (
            fast_kpool_topk_transform_fused,
        )

        bs, G, ng = 1, 256, 8
        logits = torch.randn(bs, G, device=self.dev, dtype=torch.float32)
        pos, klen = self._kept(bs=bs, G=G, n_groups=ng)
        sc = self._thr_and_scratch(logits, pos, klen, 0.75, kg_max=ng)
        lengths = torch.full((bs,), G, dtype=torch.int32, device=self.dev)
        got = self._run(logits, lengths, sc)
        want = fast_kpool_topk_transform_fused(
            score=logits, lengths=lengths, pool_size=self.POOL, topk=self.TOPK,
        )
        # length <= K takes DSA's early path in both kernels, which emits by
        # position and IS order-defined, so this one compares elementwise.
        torch.testing.assert_close(got, want, rtol=0, atol=0)



    # ---- the bitonic tier: keeps tier 1 while cutting the rest ------------

    def test_the_bitonic_tier_keeps_kept_and_cuts_the_rest_by_score(self):
        bs, G, ng = 1, 4096, 32
        logits = torch.randn(bs, G, device=self.dev, dtype=torch.float32)
        pos, klen = self._kept(bs=bs, G=G, n_groups=ng)
        sc = self._thr_and_scratch(logits, pos, klen, 0.75, kg_max=ng)
        n_kept = int(sc["kgn"][0])
        budget = self.GROUP_TOPK - n_kept

        # A threshold that admits ~700 non-kept groups: past the budget, well
        # inside kCandMax, so the bitonic tier runs rather than the radix one.
        km = sc["km"][0].bool()
        free = logits[0][~km]
        want_fired = 700
        sc["thr"].fill_(float(free.sort(descending=True).values[want_fired]))
        lengths = torch.full((bs,), G, dtype=torch.int32, device=self.dev)

        out = self._run(logits, lengths, sc)
        self.assertGreater(int(sc["ovf"].sum()), 0, "the cut was not witnessed")

        emitted = out[0][out[0] >= 0].tolist()
        got = {t // self.POOL for t in emitted}
        kept = set(sc["kg"][0, :n_kept].tolist())
        self.assertTrue(kept <= got, "the bitonic tier dropped tier 1")

        # The non-kept part must be exactly the highest-scoring candidates.
        cand = torch.nonzero((logits[0] > sc["thr"][0]) & ~km).flatten()
        order = cand[logits[0][cand].argsort(descending=True)]
        self.assertEqual(got - kept, set(order[:budget].tolist()))
        self.assertEqual(len(got), self.GROUP_TOPK)

    def test_a_seed_wider_than_the_budget_is_refused(self):
        """Dropping kept groups quietly is the one thing it must not do."""
        from sglang.kernels.ops.attention.dsa.vk_select_transform import (
            vk_select_transform,
        )

        bs, G = 1, 4096
        logits = torch.randn(bs, G, device=self.dev, dtype=torch.float32)
        wide = torch.zeros(bs, self.GROUP_TOPK + 1, dtype=torch.int32, device=self.dev)
        with self.assertRaises(Exception):
            vk_select_transform(
                score=logits,
                lengths=torch.full((bs,), G, dtype=torch.int32, device=self.dev),
                thresholds=torch.zeros(bs, device=self.dev),
                pool_size=self.POOL, topk=self.TOPK,
                kept_groups=wide,
                kept_lens=torch.zeros(bs, dtype=torch.int32, device=self.dev),
            )


if __name__ == "__main__":
    unittest.main()
