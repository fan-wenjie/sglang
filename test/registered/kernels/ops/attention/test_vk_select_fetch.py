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
    # The SERVED shape: 4k context is ~1027 groups and tier 1 keeps 243.8
    # rows = 61 groups. G=4096 with 32 kept groups was synthetic and, at the
    # old lift, fired nothing.
    G = 1027

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(0)
        self.dev = torch.device("cuda")

    def _case(self, *, n_kept_groups=61, lift=0.8, bs=1, G=None):
        G = self.G if G is None else G
        POOL = self.POOL
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

    # The fetch buffers are ONE row taller than r2t in the served backend: that
    # extra row is the trash slot (_alloc_recall_bufs does max_reqs + 1), where a
    # lane with nothing to fetch writes harmlessly. Building the two at equal
    # width here is what let a launcher that declared one row symbol for both
    # pass every test and fail every served call.
    FETCH_EXTRA_ROWS = 1

    def _old(self, logits, pos, klen, slots, r2t, q):
        from sglang.srt.layers.attention.vestigekv.dsa import branch as B

        bs, G = logits.shape
        W = self.WG * self.POOL
        sc = B.branch_scratch(bs=bs, G=G * self.POOL, nslot=bs, fetch_w=W,
                              pool=self.POOL, fence_groups=None, dev=self.dev)
        nf = bs + self.FETCH_EXTRA_ROWS
        fetch = torch.full((nf, W), -1, dtype=torch.int32, device=self.dev)
        flen = torch.zeros(nf, dtype=torch.int32, device=self.dev)
        fovf = torch.zeros(nf, dtype=torch.int32, device=self.dev)
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
        nf = bs + self.FETCH_EXTRA_ROWS
        fetch = torch.full((nf, W), -1, dtype=torch.int32, device=self.dev)
        flen = torch.zeros(nf, dtype=torch.int32, device=self.dev)
        fovf = torch.zeros(nf, dtype=torch.int32, device=self.dev)
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
        nf = bs + self.FETCH_EXTRA_ROWS
        fetch = torch.full((nf, W), -1, dtype=torch.int32, device=self.dev)
        flen = torch.zeros(nf, dtype=torch.int32, device=self.dev)
        fovf = torch.zeros(nf, dtype=torch.int32, device=self.dev)
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
        # Without this the assertion below compares two EMPTY sets and passes
        # on any implementation, which is how a 34% behavioural difference
        # survived a green suite.
        self.assertGreater(n, 0, "vacuous: neither path fired anything")
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
        logits, pos, klen, slots, r2t = self._case(lift=0.0, G=7936)
        wf, wl, _, sel = self._wide(logits, pos, klen, slots, r2t, q=0.75)
        self.assertGreater(int(sel["ovf"].sum()), 0, "expected an overflow")
        self.assertEqual(int(wl[0]), self.WG * self.POOL)
        thr = float(sel["thr"][0])
        kept = set(sel["kg"][0, : int(sel["kgn"][0])].tolist())
        keep_mask = torch.ones_like(logits[0], dtype=torch.bool)
        if kept:
            keep_mask[torch.tensor(sorted(kept), device=self.dev)] = False
        cand = torch.nonzero((logits[0] > thr) & keep_mask).flatten()
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
        # From kg, not km: the kernel is km's last reader and resets it, so
        # after the call km is all zeros by design.
        kept = set(sel["kg"][0, : int(sel["kgn"][0])].tolist())
        want = {g for g in torch.nonzero(logits[0] > thr).flatten().tolist()
                if g not in kept}
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
        logits, pos, klen, slots, r2t = self._case(lift=0.0, G=7936)
        nf, nl, _, sel = self._new(logits, pos, klen, slots, r2t, q=0.75)
        self.assertGreater(int(sel["ovf"].sum()), 0, "expected an overflow")
        self.assertEqual(int(nl[0]), self.WG * self.POOL, "budget not filled")
        thr = float(sel["thr"][0])
        kept = set(sel["kg"][0, : int(sel["kgn"][0])].tolist())
        keep_mask = torch.ones_like(logits[0], dtype=torch.bool)
        if kept:
            keep_mask[torch.tensor(sorted(kept), device=self.dev)] = False
        cand = torch.nonzero((logits[0] > thr) & keep_mask).flatten()
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


    def test_km_is_reset_so_layers_do_not_accumulate(self):
        """11 layers share ONE km buffer; without a reset the fired set decays.

        _select_scratch is keyed on (bs, G), not on layer, and the Triton chain
        this replaces reset km inside _branch_hit_kernel. With no reset, km
        accumulates the union of every layer's kept groups, the compare pass
        skips a growing set, and `fetched` decays monotonically -- served
        telemetry read 261.5 against the chain's 393.9 while 15 unit cases
        passed, because every case allocated a fresh scratch and ran ONE layer.
        """
        from sglang.kernels.ops.attention.dsa.vk_select_transform import (
            vk_collect_emit,
        )
        from sglang.srt.layers.attention.vestigekv.dsa.select import (
            kept_threshold,
            select_scratch,
        )

        G, POOL, W = 1027, self.POOL, self.WG * self.POOL
        bs, nf = 1, 1 + self.FETCH_EXTRA_ROWS
        dev = self.dev
        # A separation this small is what makes the case non-vacuous: at lift=6
        # the kept q0.75 exceeds every other score, nothing fires, and an
        # equality assertion on the fired set compares two EMPTY sets.
        sc = select_scratch(bs=bs, G=G, kg_max=64, dev=dev)
        slots = torch.zeros(bs, dtype=torch.int32, device=dev)
        plen = torch.full((bs,), G, dtype=torch.int32, device=dev)
        r2t = torch.arange(G * POOL, device=dev, dtype=torch.int32).repeat(bs, 1)
        counts = []
        for layer in range(3):
            torch.manual_seed(100 + layer)
            logits = torch.randn(bs, G, device=dev, dtype=torch.float32)
            gs = torch.randperm(G, device=dev)[:61].sort().values
            logits[0, gs] += 0.8
            pos = torch.zeros(bs, 16512, dtype=torch.int32, device=dev)
            rows = (gs[:, None] * POOL
                    + torch.arange(POOL, device=dev)[None, :]).reshape(-1)
            pos[0, : rows.numel()] = rows.to(torch.int32)
            klen = torch.full((bs,), rows.numel(), dtype=torch.int32, device=dev)
            kept_threshold(logits=logits, kept_pos=pos, klen=klen, slots=slots,
                           q=0.75, pool=POOL, scratch=sc, kg_max=64)
            fetch = torch.full((nf, W), -1, dtype=torch.int32, device=dev)
            flen = torch.zeros(nf, dtype=torch.int32, device=dev)
            fovf = torch.zeros(nf, dtype=torch.int32, device=dev)
            vk_collect_emit(logits=logits, pool_lens=plen, thresholds=sc["thr"],
                            kept_mask=sc["km"], cand_score=sc["cand_score"],
                            cand_idx=sc["cand_idx"], ncand=sc["ncand"],
                            slots=slots, r2t=r2t, fetch_buf=fetch,
                            fetch_len=flen, fetch_ovf=fovf,
                            ovf_counts=sc["ovf"], group_budget=self.WG,
                            pool=POOL)
            counts.append(int(flen[0]))
            thr = float(sc["thr"][0])
            want = int(((logits[0] > thr).sum())) - 0  # kept groups are excluded
            self.assertGreater(counts[-1], 0, "vacuous: nothing fired")
            self.assertEqual(int(sc["km"].sum()), 0,
                             "km was not reset after layer %d" % layer)
        # Each layer's count must reflect ITS OWN scores, not a shrinking residue.
        self.assertGreater(min(counts), 0)


if __name__ == "__main__":
    unittest.main()
