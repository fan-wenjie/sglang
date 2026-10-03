"""The GLM branch rule's operators (vestigekv/dsa_branch.py) against a torch
reference of the rule: threshold = q-quantile of the kept groups' logits,
fire every archive group above it, fence the lane past the budget.

Pinned here: the kept-group mask is exact; the threshold is the exact
quantile rounded DOWN by at most one histogram bin; below the budget the
fetched rows are exactly the fired groups' rows (4 per group, hottest-first
is NOT promised -- position order); past the budget fetch_len is 0 and the
overflow flag is raised, so the pack fences the lane.
"""
import unittest

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, suite="base-a-test-cuda")

T, PAGE, POOL, W, NSLOT = 20000, 64, 4, 2048, 4


def _request(seed):
    torch.manual_seed(seed)
    dev = "cuda"
    n_pages = T // PAGE + 1
    phys_pages = torch.randperm(1024, device=dev)[:n_pages]
    r2t = torch.zeros(NSLOT, 24576, dtype=torch.int64, device=dev)
    pos = torch.arange(T, device=dev)
    r2t[2, :T] = phys_pages[pos // PAGE] * PAGE + pos % PAGE
    page_table = torch.zeros(1, 512, dtype=torch.int32, device=dev)
    page_table[0, :n_pages] = phys_pages.to(torch.int32)
    G = (24576 // PAGE // POOL + 1) * PAGE
    pool_lens = torch.tensor([T // POOL], device=dev)
    kept_pos = torch.cat([torch.arange(4, device=dev),
                          torch.randperm(T - 300, device=dev)[:600],
                          torch.arange(T - 256, T, device=dev)]).unique()
    capK = 4096
    kept = torch.zeros(NSLOT, capK, dtype=torch.int32, device=dev)
    kept[2, :len(kept_pos)] = r2t[2, kept_pos].to(torch.int32)
    klen = torch.zeros(NSLOT, dtype=torch.int32, device=dev)
    klen[2] = len(kept_pos)
    return dev, r2t, page_table, G, pool_lens, kept_pos, kept, klen


def _run(dev, r2t, page_table, G, pool_lens, kept, klen, logits, q, fence_groups=None):
    from sglang.srt.layers.attention.vestigekv.dsa_branch import branch_fire, branch_scratch

    sc = branch_scratch(bs=1, G=G, nslot=NSLOT, fetch_w=W, pool=POOL,
                        fence_groups=fence_groups, dev=dev, n_pages=1024 + 2)
    fb = torch.zeros(NSLOT, W, dtype=torch.int32, device=dev)
    fl = torch.zeros(NSLOT, dtype=torch.int32, device=dev)
    fo = torch.zeros(NSLOT, dtype=torch.int32, device=dev)
    branch_fire(logits=logits, pool_lens=pool_lens, page_table=page_table,
                slots=torch.tensor([2], device=dev), kept=kept, klen=klen, q=q,
                scratch=sc, fetch_buf=fb, fetch_len=fl, fetch_ovf=fo, r2t=r2t,
                pool=POOL, page_size=PAGE, fence_groups=fence_groups)
    torch.cuda.synchronize()
    return sc, fb, fl, fo


def _reference(logits, kept_pos, G, q, pool_lens):
    kg = (kept_pos // POOL).unique()
    ks = logits[0, kg]
    thr = ks.sort(descending=True).values[int((1 - q) * len(kg))]
    arch = torch.ones(G, dtype=torch.bool, device=logits.device)
    arch[kg] = False
    valid = torch.arange(G, device=logits.device) < int(pool_lens[0])
    fired = ((logits[0] > thr) & arch & valid).nonzero().flatten()
    binw = float((ks.max() - ks.min()) / 256)
    return kg, float(thr), binw, fired


class TestBranchOperators(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_kept_mask_and_threshold(self):
        dev, r2t, page_table, G, pool_lens, kept_pos, kept, klen = _request(0)
        logits = torch.randn(1, G, device=dev)
        for q in (0.9, 0.75, 0.5):
            sc, fb, fl, fo = _run(dev, r2t, page_table, G, pool_lens, kept, klen, logits, q)
            kg, thr, binw, _ = _reference(logits, kept_pos, G, q, pool_lens)
            km_ref = torch.zeros(G, dtype=torch.bool, device=dev)
            km_ref[kg] = True
            self.assertTrue(torch.equal(sc["km"][0].bool(), km_ref))
            got = float(sc["thr"][0])
            self.assertLessEqual(got, thr + 1e-6)          # never above the exact quantile
            self.assertGreaterEqual(got, thr - binw - 1e-6)  # by at most one bin

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_below_budget_fetches_exactly_the_fired_groups(self):
        dev, r2t, page_table, G, pool_lens, kept_pos, kept, klen = _request(1)
        kg = (kept_pos // POOL).unique()
        logits = torch.randn(1, G, device=dev) - 3.0
        logits[0, kg] = torch.randn(len(kg), device=dev) + 2.0
        hot = torch.tensor([g for g in torch.randperm(T // POOL - 80, device=dev)[:300].tolist()
                            if g not in set(kg.tolist())][:50], device=dev)
        logits[0, hot] = 8.0
        sc, fb, fl, fo = _run(dev, r2t, page_table, G, pool_lens, kept, klen, logits, 0.75)
        n, ovf = int(fl[2]), int(fo[2])
        self.assertEqual(ovf, 0)
        thr = float(sc["thr"][0])
        valid = torch.arange(G, device=dev) < int(pool_lens[0])
        arch = torch.ones(G, dtype=torch.bool, device=dev)
        arch[kg] = False
        fired = ((logits[0] > thr) & arch & valid).nonzero().flatten()
        self.assertEqual(n, 4 * fired.numel())
        rows = fb[2, :n].to(torch.int64)
        pos = torch.tensor([(r2t[2] == r).nonzero()[0, 0].item() for r in rows.tolist()], device=dev)
        self.assertEqual(sorted(set((pos // POOL).tolist())), sorted(fired.tolist()))
        self.assertTrue(torch.isin(hot, pos // POOL).all())
        # every fetched row's group is a fired group, 4 rows each, none kept
        self.assertEqual(len(set(pos.tolist()) & set(kept_pos.tolist())), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_past_the_budget_the_lane_is_fenced(self):
        dev, r2t, page_table, G, pool_lens, kept_pos, kept, klen = _request(2)
        logits = torch.randn(1, G, device=dev)  # archive as hot as kept: thousands fire
        sc, fb, fl, fo = _run(dev, r2t, page_table, G, pool_lens, kept, klen, logits, 0.75)
        self.assertEqual(int(fo[2]), 1)
        self.assertEqual(int(fl[2]), 0)
        # and with a low fence trigger a modest fire fences too
        logits = torch.randn(1, G, device=dev) - 3.0
        kg = (kept_pos // POOL).unique()
        logits[0, kg] = torch.randn(len(kg), device=dev) + 2.0
        hot = torch.tensor([g for g in torch.randperm(T // POOL - 80, device=dev)[:300].tolist()
                            if g not in set(kg.tolist())][:50], device=dev)
        logits[0, hot] = 8.0
        sc, fb, fl, fo = _run(dev, r2t, page_table, G, pool_lens, kept, klen, logits, 0.75, fence_groups=16)
        self.assertEqual(int(fo[2]), 1)
        self.assertEqual(int(fl[2]), 0)


if __name__ == "__main__":
    unittest.main()
