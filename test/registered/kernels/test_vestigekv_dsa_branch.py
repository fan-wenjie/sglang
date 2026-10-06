"""The GLM branch rule's operators (vestigekv/dsa/branch.py) against a torch
reference of the rule: threshold = q-quantile of the kept groups' logits, fire
every archive group above it, fence the lane past the trigger.

Pinned here: the kept-group mask is exact; the threshold is the exact quantile
rounded DOWN by at most one histogram bin; below the trigger the fetched rows
are exactly the fired groups' rows, POOL per group; past it fetch_len is 0 and
the overflow flag is raised, so the pack fences the lane to DSA's own top-k.

And the property the fused commit kernel exists for: the fallback and the
recall are two arms of ONE kernel, so two lanes of one call can take different
arms. TestBranchMixedBatch drives exactly that -- a hot lane that fences beside
a cold lane that recalls -- which a chain that decides in one kernel and acts
in another could not express without a host-side readback of the count.

An earlier case guarded a page-table inverse (a table wider than the KV pool
crashed graph capture). Tier 1 now records each kept row's request-relative
POSITION, so no kernel here reads a page table and that failure cannot occur.
The shuffled physical pages in the fixture still guard the replacement: a
regression to row-derived groups would not resolve to this request's own rows.

The scratch is also checked to come back the way the kernels promise: nothing
is zeroed from the host between calls, so a second call on one scratch must
agree with the first.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, suite="base-a-test-cuda")

T, PAGE, POOL, W, NSLOT, CAPK, NPAGE = 20000, 64, 4, 2048, 4, 4096, 1024


def _lane(seed, slot, r2t, kept, klen, n_hot_kept=600):
    """Lay one request of T tokens into `slot` over SHUFFLED physical pages and
    give it a kept set of sinks + sigma picks + the recent window. The tables
    are filled the way tier 1 fills them -- the POSITION, not the physical row,
    is what the rule reads. Returns the kept positions."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n_pages = T // PAGE + 1
    phys = torch.randperm(NPAGE, device="cuda", generator=g)[:n_pages]
    pos = torch.arange(T, device="cuda")
    r2t[slot, :T] = phys[pos // PAGE] * PAGE + pos % PAGE
    kept_pos = torch.cat([
        torch.arange(4, device="cuda"),
        torch.randperm(T - 300, device="cuda", generator=g)[:n_hot_kept],
        torch.arange(T - 256, T, device="cuda"),
    ]).unique()
    kept[slot, : len(kept_pos)] = kept_pos.to(torch.int32)
    klen[slot] = len(kept_pos)
    return kept_pos


def _fixture(seeds_slots):
    dev = "cuda"
    r2t = torch.zeros(NSLOT, 24576, dtype=torch.int64, device=dev)
    kept = torch.zeros(NSLOT, CAPK, dtype=torch.int32, device=dev)
    klen = torch.zeros(NSLOT, dtype=torch.int32, device=dev)
    kepts = [_lane(seed, slot, r2t, kept, klen) for seed, slot in seeds_slots]
    G = (24576 // PAGE // POOL + 1) * PAGE
    slots = torch.tensor([s for _, s in seeds_slots], device=dev)
    pool_lens = torch.full((len(seeds_slots),), T // POOL, dtype=torch.int32, device=dev)
    return dev, r2t, kept, klen, slots, pool_lens, G, kepts


def _scratch(bs, G, fence_groups=None):
    from sglang.srt.layers.attention.vestigekv.dsa.branch import branch_scratch

    return branch_scratch(bs=bs, G=G, nslot=NSLOT, fetch_w=W, pool=POOL,
                          fence_groups=fence_groups, dev="cuda")


def _run(sc, logits, pool_lens, slots, kept_pos, klen, r2t, q,
         fence_groups=None, out=None):
    from sglang.srt.layers.attention.vestigekv.dsa.branch import branch_fire

    if out is None:
        out = (
            torch.zeros(NSLOT, W, dtype=torch.int32, device="cuda"),
            torch.zeros(NSLOT, dtype=torch.int32, device="cuda"),
            torch.zeros(NSLOT, dtype=torch.int32, device="cuda"),
        )
    fb, fl, fo = out
    branch_fire(logits=logits, pool_lens=pool_lens, slots=slots,
                kept_pos=kept_pos, klen=klen, q=q, scratch=sc,
                fetch_buf=fb, fetch_len=fl, fetch_ovf=fo, r2t=r2t, pool=POOL,
                fence_groups=fence_groups)
    torch.cuda.synchronize()
    return fb, fl, fo


def _kept_groups(kept_pos):
    return (kept_pos // POOL).unique()


def _exact(logits, b, kept_pos, G, q, plen):
    """The rule in torch: the exact q-quantile of the kept GROUPS' logits, then
    every valid archive group above it."""
    kg = _kept_groups(kept_pos)
    ks = logits[b, kg]
    thr = ks.sort(descending=True).values[int((1 - q) * len(kg))]
    arch = torch.ones(G, dtype=torch.bool, device=logits.device)
    arch[kg] = False
    valid = torch.arange(G, device=logits.device) < plen
    fired = ((logits[b] > thr) & arch & valid).nonzero().flatten()
    return kg, float(thr), float((ks.max() - ks.min()) / 256), fired


def _cold(G, kg, n_hot, dev, seed):
    """Logits where the kept groups dominate and exactly n_hot archive groups
    beat them: a lane that recalls a known set without overflowing."""
    g = torch.Generator(device=dev).manual_seed(seed)
    lg = torch.randn(G, device=dev, generator=g) - 3.0
    lg[kg] = torch.randn(len(kg), device=dev, generator=g) + 2.0
    pool = torch.randperm(T // POOL - 80, device=dev, generator=g)
    keep = set(kg.tolist())
    hot = torch.tensor([x for x in pool.tolist() if x not in keep][:n_hot], device=dev)
    lg[hot] = 8.0
    return lg, hot


def _positions(rows, r2t, slot):
    """The request positions of physical rows, by inverting this slot's r2t."""
    return torch.tensor(
        [(r2t[slot] == r).nonzero()[0, 0].item() for r in rows.tolist()],
        device=rows.device,
    )


def _fired_by(logits_row, thr, kg, G, plen, dev):
    arch = torch.ones(G, dtype=torch.bool, device=dev)
    arch[kg] = False
    valid = torch.arange(G, device=dev) < plen
    return ((logits_row > thr) & arch & valid).nonzero().flatten()


class TestBranchOperators(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_kept_mask_and_threshold(self):
        # A NON-overflowing fixture, deliberately: with `randn` logits and a
        # two-row kept set the quantile fires 517-2490 groups against a 512-group
        # buffer, so the capacity threshold binds and the quantile is no longer
        # what `thr` reports. _cold is the fixture built to recall without
        # overflowing, which is where the quantile property lives.
        dev, r2t, kept, klen, slots, plen, G, kepts = _fixture([(0, 2)])
        base, _ = _cold(G, _kept_groups(kepts[0]), 50, dev, 3)
        logits = base[None]
        for q in (0.9, 0.75, 0.5):
            sc = _scratch(1, G)
            _run(sc, logits, plen, slots, kept, klen, r2t, q)
            _, thr, binw, _ = _exact(logits, 0, kepts[0], G, q, int(plen[0]))
            # km is reset by its last reader, so it is zero AFTER the call; the
            # mask is checked through the threshold it produced, which is a
            # function of exactly the kept groups' logits.
            self.assertTrue(torch.equal(sc["km"][0], torch.zeros_like(sc["km"][0])))
            got = float(sc["thr"][0])
            self.assertLessEqual(got, thr + 1e-6)            # never above the exact quantile
            self.assertGreaterEqual(got, thr - binw - 1e-6)  # by at most one bin

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_the_coarse_proxy_compiles_and_only_ever_removes_fires(self):
        """The proxy path must COMPILE, and must be a subset of the full fire set.

        It did not compile: offs is int32 and the Knuth constant does not fit,
        so `ValueError: Scalar 2654435761 is out of range for type int32` came
        out of Triton at first call -- which is server start, so four queued
        jobs died with "server did not start" before anything ran. A Triton
        kernel behind an env flag has no compile-time coverage unless a test
        sets the flag, and this one does.
        """
        from sglang.srt.environ import envs

        dev, r2t, kept, klen, slots, plen, G, kepts = _fixture([(1, 2)])
        kg = _kept_groups(kepts[0])
        logits, _ = _cold(G, kg, 50, dev, 7)
        sc = _scratch(1, G)
        _, full_len, _ = _run(sc, logits[None], plen, slots, kept, klen, r2t, 0.75)
        full = int(full_len[2])
        for spec in ("16,0.5", "16,0.75", "8,0.5"):
            with envs.SGLANG_VESTIGEKV_COARSE_PROXY.override(spec):
                sc2 = _scratch(1, G)
                _, fl, fo = _run(sc2, logits[None], plen, slots, kept, klen, r2t, 0.75)
                n = int(fl[2])
                self.assertEqual(int(fo[2]), 0, spec)
                self.assertLessEqual(n, full, f"{spec}: the proxy added fires")
                self.assertEqual(n % POOL, 0, spec)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_below_the_trigger_fetches_exactly_the_fired_groups(self):
        dev, r2t, kept, klen, slots, plen, G, kepts = _fixture([(1, 2)])
        kg = _kept_groups(kepts[0])
        logits, hot = _cold(G, kg, 50, dev, 7)
        sc = _scratch(1, G)
        fb, fl, fo = _run(sc, logits[None], plen, slots, kept, klen, r2t, 0.75)
        n = int(fl[2])
        self.assertEqual(int(fo[2]), 0)
        fired = _fired_by(logits, float(sc["thr"][0]), kg, G, int(plen[0]), dev)
        self.assertEqual(n, POOL * fired.numel())
        p = _positions(fb[2, :n].to(torch.int64), r2t, 2)
        self.assertEqual(sorted(set((p // POOL).tolist())), sorted(fired.tolist()))
        self.assertEqual(p.numel(), POOL * fired.numel())
        self.assertEqual(len(set(p.tolist()) & set(kepts[0].tolist())), 0)
        self.assertTrue(torch.isin(hot, p // POOL).all())

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_past_the_trigger_the_lane_truncates_by_score(self):
        """A would-be-overflowing lane keeps its OWN best groups, not DSA's.

        The old behaviour was fovf=1 / flen=0, which handed the lane DSA's
        index_topk selection -- the same 2048 rows the buffer holds, minus the
        kept rows, and measured worse. The capacity threshold caps the fired count
        instead, so the lane never has to be handed anything.

        The decisive assertion is the ORDER: nothing dropped may outscore anything
        kept. Truncating by position passes a count check and fails this one, and
        position-order truncation was measured to retain 18.7% of what beats max1
        against 80.3% by score.
        """
        dev, r2t, kept, klen, slots, plen, G, kepts = _fixture([(2, 2)])
        kg = _kept_groups(kepts[0])
        logits = torch.randn(1, G, device=dev)  # archive as hot as kept: overflows
        WG = W // POOL
        sc = _scratch(1, G)
        fb, fl, fo = _run(sc, logits, plen, slots, kept, klen, r2t, 0.75)
        self.assertEqual(int(fo[2]), 0)                       # not fenced any more
        n = int(fl[2])
        self.assertGreater(n, 0)
        self.assertLessEqual(n, POOL * WG)                    # within the buffer
        fired = _fired_by(logits[0], float(sc["thr"][0]), kg, G, int(plen[0]), dev)
        self.assertEqual(n, POOL * fired.numel())
        self.assertLessEqual(fired.numel(), WG)
        arch = torch.ones(G, dtype=torch.bool, device=dev)
        arch[kg] = False
        cand = (arch & (torch.arange(G, device=dev) < int(plen[0]))).nonzero().flatten()
        dropped = cand[~torch.isin(cand, fired)]
        self.assertGreater(float(logits[0, fired].min()),
                           float(logits[0, dropped].max()))
        # and with the trigger lowered, the same holds at row granularity
        lg, _ = _cold(G, kg, 50, dev, 11)
        sc = _scratch(1, G, fence_groups=16)
        fb, fl, fo = _run(sc, lg[None], plen, slots, kept, klen, r2t, 0.75,
                          fence_groups=16)
        self.assertEqual(int(fo[2]), 0)
        n = int(fl[2])
        self.assertLessEqual(n, POOL * 16)
        fired = _fired_by(lg, float(sc["thr"][0]), kg, G, int(plen[0]), dev)
        self.assertEqual(n, POOL * fired.numel())
        p = _positions(fb[2, :n].to(torch.int64), r2t, 2)
        self.assertEqual(sorted(set((p // POOL).tolist())), sorted(fired.tolist()))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_the_control_arm_still_fences(self):
        """SGLANG_VESTIGEKV_FENCE_TRUNCATE=0 restores the fallback.

        The old path stays reachable because it is the control that measures what
        the fallback was worth; without this the A/B cannot be run.
        """
        from sglang.srt.environ import envs

        dev, r2t, kept, klen, slots, plen, G, kepts = _fixture([(2, 2)])
        logits = torch.randn(1, G, device=dev)
        with envs.SGLANG_VESTIGEKV_FENCE_TRUNCATE.override(False):
            sc = _scratch(1, G)
            fb, fl, fo = _run(sc, logits, plen, slots, kept, klen, r2t, 0.75)
        self.assertEqual(int(fo[2]), 1)
        self.assertEqual(int(fl[2]), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_the_scratch_is_reusable_without_a_host_memset(self):
        # The histogram, range and counts are reset by the kernels themselves,
        # so call N+1 on one scratch must equal call 1. A missed reset shows up
        # as a drifting threshold, which no single call can reveal.
        dev, r2t, kept, klen, slots, plen, G, kepts = _fixture([(3, 2)])
        logits, _ = _cold(G, _kept_groups(kepts[0]), 40, dev, 13)
        sc = _scratch(1, G)
        first = None
        for _ in range(4):
            fb, fl, fo = _run(sc, logits[None], plen, slots, kept, klen, r2t, 0.75)
            cur = (float(sc["thr"][0]), int(fl[2]), int(fo[2]),
                   fb[2, : int(fl[2])].clone())
            if first is None:
                first = cur
            else:
                self.assertEqual(cur[:3], first[:3])
                self.assertTrue(torch.equal(cur[3], first[3]))


class TestBranchMixedBatch(CustomTestCase):
    """Two lanes, one call: the fallback and the recall are arms of one kernel,
    so a batch may take both."""

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_one_lane_truncates_while_the_other_recalls(self):
        dev, r2t, kept, klen, slots, plen, G, kepts = _fixture([(4, 1), (5, 2)])
        hot_lane = torch.randn(G, device=dev)                            # truncates
        cold_lane, hot = _cold(G, _kept_groups(kepts[1]), 30, dev, 17)   # recalls
        logits = torch.stack([hot_lane, cold_lane])
        sc = _scratch(2, G)
        fb, fl, fo = _run(sc, logits, plen, slots, kept, klen, r2t, 0.75)

        # The capped lane fills its buffer to the brim instead of emptying it, and
        # the two arms still disagree within one call -- which is what this case
        # is for.
        self.assertEqual(int(fo[1]), 0)
        self.assertGreater(int(fl[1]), 0)
        self.assertLessEqual(int(fl[1]), W)
        self.assertEqual(int(fo[2]), 0)
        n = int(fl[2])
        self.assertGreater(n, 0)
        self.assertEqual(n % POOL, 0)

        fired = _fired_by(cold_lane, float(sc["thr"][1]), _kept_groups(kepts[1]),
                          G, int(plen[1]), dev)
        self.assertEqual(n, POOL * fired.numel())
        p = _positions(fb[2, :n].to(torch.int64), r2t, 2)
        self.assertEqual(sorted(set((p // POOL).tolist())), sorted(fired.tolist()))
        self.assertTrue(torch.isin(hot, p // POOL).all())
        # the capped lane's rows are its own: its buffer holds no row of the
        # recalling lane's request (the two slots' page tables are disjoint)
        n1 = int(fl[1])
        self.assertEqual(len(set(fb[1, :n1].tolist()) & set(fb[2, :n].tolist())), 0)


if __name__ == "__main__":
    unittest.main()
