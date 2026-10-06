"""The 2-kernel branch rule against the 5-kernel one it replaces.

Not bit-identical by construction: the old compaction emits fired groups in
POSITION order (a prefix sum over groups in order) while this one appends
atomically, so the row ORDER differs and attention is order-invariant. What must
agree is the SET of rows and the length. Where the two genuinely differ is the
overflow cut -- the old cap quantises to 24 score levels, this sorts -- so that
case is tested against the exact rule instead.
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


class TestVkSelectFetch(CustomTestCase):
    POOL = 4
    WG = 512
    G = 4096

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(0)
        self.dev = torch.device("cuda")

    def _case(self, *, n_kept_groups=32, lift=6.0, bs=1):
        G, POOL = self.G, self.POOL
        dev = self.dev
        logits = torch.randn(bs, G, device=dev, dtype=torch.float32)
        capk = n_kept_groups * POOL
        pos = torch.zeros(bs, capk, dtype=torch.int32, device=dev)
        for b in range(bs):
            gs = torch.randperm(G, device=dev)[:n_kept_groups].sort().values
            logits[b, gs] += lift
            rows = (gs[:, None] * POOL
                    + torch.arange(POOL, device=dev)[None, :]).reshape(-1)
            pos[b, : rows.numel()] = rows.to(torch.int32)
        klen = torch.full((bs,), capk, dtype=torch.int32, device=dev)
        slots = torch.arange(bs, dtype=torch.int32, device=dev)
        # r2t maps a token position to a physical row; a permutation makes a
        # mapping bug visible, where identity would hide one.
        maxctx = G * POOL
        r2t = torch.randperm(maxctx, device=dev, dtype=torch.int32).repeat(bs, 1)
        return logits, pos, klen, slots, r2t

    def _old(self, logits, pos, klen, slots, r2t, q):
        from sglang.srt.layers.attention.vestigekv.dsa import branch as B

        bs, G = logits.shape
        W = self.WG * self.POOL
        sc = B.branch_scratch(bs=bs, G=G * self.POOL, nslot=bs, fetch_w=W,
                              pool=self.POOL, fence_groups=None, dev=self.dev)
        fetch = torch.full((bs, W), -1, dtype=torch.int32, device=self.dev)
        flen = torch.zeros(bs, dtype=torch.int32, device=self.dev)
        fovf = torch.zeros(bs, dtype=torch.int32, device=self.dev)
        plen = torch.full((bs,), G, dtype=torch.int32, device=self.dev)
        B.branch_fire(logits=logits, pool_lens=plen, slots=slots, kept_pos=pos,
                      klen=klen, q=q, scratch=sc, fetch_buf=fetch,
                      fetch_len=flen, fetch_ovf=fovf, r2t=r2t, pool=self.POOL,
                      fence_groups=None, kept_rows_max=int(klen[0]))
        return fetch, flen, fovf, sc

    def _new(self, logits, pos, klen, slots, r2t, q):
        from sglang.kernels.ops.attention.dsa.vk_select_transform import (
            vk_select_fetch,
        )
        from sglang.srt.layers.attention.vestigekv.dsa.select import (
            kept_threshold,
            select_scratch,
        )

        bs, G = logits.shape
        W = self.WG * self.POOL
        kg = int(klen[0]) // self.POOL
        sel = select_scratch(bs=bs, G=G, kg_max=kg, dev=self.dev)
        kept_threshold(logits=logits, kept_pos=pos, klen=klen, slots=slots, q=q,
                       pool=self.POOL, scratch=sel, kg_max=kg)
        fetch = torch.full((bs, W), -1, dtype=torch.int32, device=self.dev)
        flen = torch.zeros(bs, dtype=torch.int32, device=self.dev)
        fovf = torch.zeros(bs, dtype=torch.int32, device=self.dev)
        plen = torch.full((bs,), G, dtype=torch.int32, device=self.dev)
        vk_select_fetch(logits=logits, pool_lens=plen, thresholds=sel["thr"],
                        kept_mask=sel["km"], slots=slots, r2t=r2t,
                        fetch_buf=fetch, fetch_len=flen, fetch_ovf=fovf,
                        ovf_counts=sel["ovf"], group_budget=self.WG,
                        pool=self.POOL, group_topk=self.WG)
        return fetch, flen, fovf, sel

    def _wide(self, logits, pos, klen, slots, r2t, q):
        from sglang.kernels.ops.attention.dsa.vk_select_transform import (
            vk_collect_emit,
        )
        from sglang.srt.layers.attention.vestigekv.dsa.select import (
            kept_threshold,
            select_scratch,
        )

        bs, G = logits.shape
        W = self.WG * self.POOL
        kg = int(klen[0]) // self.POOL
        sel = select_scratch(bs=bs, G=G, kg_max=kg, dev=self.dev)
        kept_threshold(logits=logits, kept_pos=pos, klen=klen, slots=slots, q=q,
                       pool=self.POOL, scratch=sel, kg_max=kg)
        fetch = torch.full((bs, W), -1, dtype=torch.int32, device=self.dev)
        flen = torch.zeros(bs, dtype=torch.int32, device=self.dev)
        fovf = torch.zeros(bs, dtype=torch.int32, device=self.dev)
        plen = torch.full((bs,), G, dtype=torch.int32, device=self.dev)
        vk_collect_emit(logits=logits, pool_lens=plen, thresholds=sel["thr"],
                        kept_mask=sel["km"], cand_score=sel["cand_score"],
                        cand_idx=sel["cand_idx"], ncand=sel["ncand"],
                        slots=slots, r2t=r2t, fetch_buf=fetch, fetch_len=flen,
                        fetch_ovf=fovf, ovf_counts=sel["ovf"],
                        group_budget=self.WG, pool=self.POOL)
        return fetch, flen, fovf, sel

    def test_the_wide_path_matches_the_chain(self):
        """The 2.58x result is worthless unless it is the same answer."""
        args = self._case()
        of, ol, oo, _ = self._old(*args, q=0.75)
        wf, wl, wo, sel = self._wide(*args, q=0.75)
        self.assertEqual(int(sel["ovf"].sum()), 0)
        self.assertEqual(int(ol[0]), int(wl[0]), "fetch_len disagrees")
        n = int(ol[0])
        self.assertEqual(set(of[0, :n].tolist()), set(wf[0, :n].tolist()))
        self.assertEqual(int(oo[0]), int(wo[0]))

    def test_the_wide_path_resets_its_counter(self):
        """ncand is global and reused every layer-step; a missed reset would
        accumulate and silently truncate from the second call on."""
        args = self._case()
        first = None
        for _ in range(3):
            wf, wl, _, sel = self._wide(*args, q=0.75)
            n = int(wl[0])
            got = set(wf[0, :n].tolist())
            if first is None:
                first = (n, got)
            self.assertEqual((n, got), first, "the counter did not reset")
            self.assertEqual(int(sel["ncand"][0]), 0)

    def test_the_wide_path_cuts_by_score_on_overflow(self):
        logits, pos, klen, slots, r2t = self._case(lift=0.0)
        wf, wl, _, sel = self._wide(logits, pos, klen, slots, r2t, q=0.75)
        self.assertGreater(int(sel["ovf"].sum()), 0, "expected an overflow")
        self.assertEqual(int(wl[0]), self.WG * self.POOL)
        thr = float(sel["thr"][0])
        km = sel["km"][0].bool()
        cand = torch.nonzero((logits[0] > thr) & ~km).flatten()
        best = cand[logits[0][cand].argsort(descending=True)][: self.WG]
        got = set()
        for i in range(0, int(wl[0]), self.POOL):
            r = int(wf[0, i])
            got.add(int((r2t[0] == r).nonzero()[0]) // self.POOL)
        self.assertEqual(got, set(best.tolist()))

    def test_the_two_paths_emit_the_same_rows(self):
        """The drop-in claim: same set, same length, same fence flag."""
        args = self._case()
        of, ol, oo, _ = self._old(*args, q=0.75)
        nf, nl, no, sel = self._new(*args, q=0.75)
        self.assertEqual(int(sel["ovf"].sum()), 0, "this case must not overflow")
        self.assertEqual(int(ol[0]), int(nl[0]), "fetch_len disagrees")
        n = int(ol[0])
        self.assertEqual(set(of[0, :n].tolist()), set(nf[0, :n].tolist()))
        self.assertEqual(int(oo[0]), int(no[0]))

    def test_the_rows_are_pool_contiguous_through_r2t(self):
        """Each selected group must land as its POOL rows, r2t-mapped."""
        logits, pos, klen, slots, r2t = self._case()
        nf, nl, _, sel = self._new(logits, pos, klen, slots, r2t, q=0.75)
        n = int(nl[0])
        self.assertEqual(n % self.POOL, 0)
        thr = float(sel["thr"][0])
        km = sel["km"][0].bool()
        want = set(torch.nonzero((logits[0] > thr) & ~km).flatten().tolist())
        got = set()
        for i in range(0, n, self.POOL):
            rows = nf[0, i : i + self.POOL].tolist()
            # invert the permutation to recover the token positions
            tp = [int((r2t[0] == r).nonzero()[0]) for r in rows]
            self.assertEqual(tp, list(range(tp[0], tp[0] + self.POOL)))
            self.assertEqual(tp[0] % self.POOL, 0)
            got.add(tp[0] // self.POOL)
        self.assertEqual(got, want)

    def test_overflow_keeps_the_highest_scoring_exactly(self):
        """Where the two rules differ: the old cap quantises, this sorts."""
        # A representative kept set puts the quantile mid-distribution, which is
        # how a lane fires far past the budget.
        logits, pos, klen, slots, r2t = self._case(lift=0.0)
        nf, nl, _, sel = self._new(logits, pos, klen, slots, r2t, q=0.75)
        self.assertGreater(int(sel["ovf"].sum()), 0, "expected an overflow")
        self.assertEqual(int(nl[0]), self.WG * self.POOL, "budget not filled")
        thr = float(sel["thr"][0])
        km = sel["km"][0].bool()
        cand = torch.nonzero((logits[0] > thr) & ~km).flatten()
        best = cand[logits[0][cand].argsort(descending=True)][: self.WG]
        got = set()
        for i in range(0, int(nl[0]), self.POOL):
            r = int(nf[0, i])
            got.add(int((r2t[0] == r).nonzero()[0]) // self.POOL)
        self.assertEqual(got, set(best.tolist()))

    def test_a_budget_wider_than_the_buffer_is_refused(self):
        from sglang.kernels.ops.attention.dsa.vk_select_transform import (
            vk_select_fetch,
        )
        from sglang.srt.layers.attention.vestigekv.dsa.select import (
            kept_threshold,
            select_scratch,
        )

        logits, pos, klen, slots, r2t = self._case()
        bs, G = logits.shape
        kg = int(klen[0]) // self.POOL
        sel = select_scratch(bs=bs, G=G, kg_max=kg, dev=self.dev)
        kept_threshold(logits=logits, kept_pos=pos, klen=klen, slots=slots,
                       q=0.75, pool=self.POOL, scratch=sel, kg_max=kg)
        narrow = torch.full((bs, 16), -1, dtype=torch.int32, device=self.dev)
        with self.assertRaises(Exception):
            vk_select_fetch(
                logits=logits,
                pool_lens=torch.full((bs,), G, dtype=torch.int32, device=self.dev),
                thresholds=sel["thr"], kept_mask=sel["km"], slots=slots, r2t=r2t,
                fetch_buf=narrow,
                fetch_len=torch.zeros(bs, dtype=torch.int32, device=self.dev),
                fetch_ovf=torch.zeros(bs, dtype=torch.int32, device=self.dev),
                ovf_counts=sel["ovf"], group_budget=self.WG, pool=self.POOL,
                group_topk=self.WG,
            )


if __name__ == "__main__":
    unittest.main()
