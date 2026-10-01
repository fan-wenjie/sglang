"""The SPLIT prologue -- the one the served decode path actually runs.

Nothing tested it before this file. `test_vestigekv_fused_prologue.py` covers
`fused_prologue`, while `batched_step` calls `fused_prologue_split`, so the
kernel on the hot path had no direct coverage and a change to it could pass the
whole suite. That was discovered by changing it.

It also pins the qperp_rel output the width controller reads, and the register
cost of the constexpr that emits it: a spilling variant slows every step,
including the ones whose feature never fires.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-small")

H, R, KV, DD = 32, 64, 512, 64


def _case(P=4, NKm=256, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    qbuf = torch.randn(1, P, H, 576, device=dev, dtype=torch.bfloat16)
    li = torch.zeros(P, dtype=torch.int64, device=dev)
    slot = torch.arange(P, dtype=torch.int64, device=dev)
    kr = torch.randn(P, NKm, 576, device=dev, dtype=torch.bfloat16)
    v = torch.stack([
        torch.linalg.qr(torch.randn(KV, R, device=dev))[0].T.contiguous()
        for _ in range(P)
    ])
    nk = torch.randint(NKm // 2, NKm, (P,), device=dev, dtype=torch.int64)
    al = torch.full((P,), 1000, dtype=torch.int64, device=dev)
    thr = torch.rand(P, device=dev) * 3
    return qbuf, li, slot, kr, v, nk, thr, al


def _run(qrel=None, P=4):
    from sglang.srt.layers.attention.vestigekv import fused_prologue as FP

    qbuf, li, slot, kr, v, nk, thr, al = _case(P=P)
    dev = "cuda"
    out = (
        torch.zeros(P, H, device=dev),
        torch.zeros(P, DD, H, device=dev, dtype=torch.bfloat16),
        torch.zeros(P, R, H, device=dev, dtype=torch.float16),
        torch.zeros(P, H, device=dev),
    )
    NS = FP._NSPLIT
    parts = tuple(torch.zeros(P, NS, H, device=dev) for _ in range(3))
    FP.fused_prologue_split(
        qbuf, li, slot, kr, v, nk, thr, 1.0 / (192 ** 0.5),
        out=out, partials=parts, a_len=al, qrel=qrel,
    )
    torch.cuda.synchronize()
    return qbuf, v, slot, out


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestPrologueSplit(CustomTestCase):
    def test_qres_matches_a_torch_reference(self):
        qbuf, v, slot, out = _run()
        qc = qbuf[0, slot, :, :KV].float()
        qsk = torch.einsum("phd,prd->phr", qc, v)
        ref = (qc - torch.einsum("phr,prd->phd", qsk, v)).norm(dim=-1)
        torch.testing.assert_close(out[3], ref, rtol=2e-2, atol=2e-2)

    def test_qrel_is_the_residual_fraction(self):
        """qperp_rel = ||q_res|| / ||q_c||, which is what the width controller
        thresholds per layer."""
        P = 4
        qrel = torch.zeros(P, H, device="cuda")
        qbuf, v, slot, out = _run(qrel=qrel, P=P)
        qc = qbuf[0, slot, :, :KV].float()
        qsk = torch.einsum("phd,prd->phr", qc, v)
        res = (qc - torch.einsum("phr,prd->phd", qsk, v)).norm(dim=-1)
        ref = res / qc.norm(dim=-1)
        torch.testing.assert_close(qrel, ref, rtol=2e-2, atol=2e-2)
        self.assertTrue(bool(((qrel >= 0) & (qrel <= 1.001)).all()),
                        f"qperp_rel outside [0,1]: {qrel.min()}..{qrel.max()}")

    def test_qrel_off_leaves_the_buffer_untouched(self):
        """The constexpr must really gate the store, or 'off' is not off."""
        P = 4
        qrel = torch.full((P, H), -7.0, device="cuda")
        _run(qrel=None, P=P)
        self.assertTrue(bool((qrel == -7.0).all()), "wrote with WRITE_REL off")

    def test_the_qrel_variant_does_not_spill(self):
        """A spilling build slows every step, not just the ones that use it."""
        from sglang.srt.layers.attention.vestigekv import fused_prologue as FP

        def compiled():
            # triton 3.7: JITFunction.device_caches[dev] = (by_key, ...)
            # -- the idiom mexp/kimi/fence_disasm.py already uses.
            cache = FP._prologue_merge_kernel.device_caches[
                torch.cuda.current_device()][0]
            return list(cache.values())

        _run(qrel=None)
        base = compiled()[-1]
        _run(qrel=torch.zeros(4, H, device="cuda"))
        rel = [k for k in compiled() if k is not base][-1]
        self.assertEqual(rel.n_spills, 0,
                         f"qrel variant spills {rel.n_spills} bytes")
        self.assertLessEqual(
            rel.n_regs, base.n_regs + 4,
            f"qrel variant costs {rel.n_regs - base.n_regs} registers "
            f"({base.n_regs} -> {rel.n_regs}); both norms were already live",
        )


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestPerHeadMarginSelector(CustomTestCase):
    """margin chosen per head from this step's own qperp_rel.

    It lives in the kernel because qperp_rel is computed there and the margin is
    consumed there; deciding outside would make it stale by a step for nothing.
    And it is the MARGIN rather than cc because on the served arm firing is not
    certificate-driven -- Spearman between qperp_rel and the fire fraction is
    -0.292 on the layer carrying 82% of the fetch, so scaling cc cannot move
    admission while the margin shifts the threshold directly.
    """

    def _run_sel(self, relthr, margin, margin_hi, P=4):
        from sglang.srt.layers.attention.vestigekv import fused_prologue as FP

        qbuf, li, slot, kr, v, nk, thr, al = _case(P=P)
        dev = "cuda"
        out = (torch.zeros(P, H, device=dev),
               torch.zeros(P, DD, H, device=dev, dtype=torch.bfloat16),
               torch.zeros(P, R, H, device=dev, dtype=torch.float16),
               torch.zeros(P, H, device=dev))
        qrel = torch.zeros(P, H, device=dev)
        parts = tuple(torch.zeros(P, FP._NSPLIT, H, device=dev) for _ in range(3))
        FP.fused_prologue_split(
            qbuf, li, slot, kr, v, nk, thr, 1.0 / (192 ** 0.5),
            out=out, partials=parts, a_len=al, qrel=qrel,
            relthr=relthr, margin=margin, margin_hi=margin_hi,
        )
        torch.cuda.synchronize()
        return out[0], qrel

    def test_threshold_at_zero_selects_the_wide_margin_everywhere(self):
        """qperp_rel >= 0 always, so thr=0 must give margin_hi for every head;
        compare against a plain run at that same margin."""
        P = 4
        lo = self._run_sel(torch.zeros(P, device="cuda"), 0.0, 3.0)[0]
        ref = self._run_sel(None, 3.0, None)[0]
        torch.testing.assert_close(lo, ref, rtol=1e-5, atol=1e-5)

    def test_threshold_above_one_selects_the_narrow_margin_everywhere(self):
        P = 4
        hi = self._run_sel(torch.full((P,), 2.0, device="cuda"), 0.0, 3.0)[0]
        ref = self._run_sel(None, 0.0, None)[0]
        torch.testing.assert_close(hi, ref, rtol=1e-5, atol=1e-5)

    def test_the_split_is_per_head_and_follows_qperp_rel(self):
        """With the threshold at each slot's own median, about half the heads
        should take the wide margin -- and exactly the ones above it."""
        P = 4
        _, qrel = self._run_sel(torch.zeros(P, device="cuda"), 0.0, 3.0)
        med = qrel.median(dim=1).values
        wide, _ = self._run_sel(med, 0.0, 3.0)
        narrow, _ = self._run_sel(torch.full((P,), 9.0, device="cuda"), 0.0, 3.0)
        allwide, _ = self._run_sel(torch.zeros(P, device="cuda"), 0.0, 3.0)
        took_wide = (wide - narrow).abs() > 1e-6
        self.assertTrue(bool((took_wide == (qrel > med[:, None])).all()),
                        "per-head selection did not follow qperp_rel")
        frac = took_wide.float().mean().item()
        self.assertGreater(frac, 0.3, f"only {frac:.2f} of heads took the wide margin")
        self.assertLess(frac, 0.7, f"{frac:.2f} of heads took the wide margin")
        self.assertFalse(bool(torch.equal(allwide, narrow)), "margin had no effect")

    def test_the_selector_does_not_spill(self):
        from sglang.srt.layers.attention.vestigekv import fused_prologue as FP

        def compiled():
            return list(FP._prologue_merge_kernel.device_caches[
                torch.cuda.current_device()][0].values())

        self._run_sel(None, 0.0, None)
        base = compiled()[-1]
        self._run_sel(torch.zeros(4, device="cuda"), 0.0, 3.0)
        sel = [k for k in compiled() if k is not base][-1]
        self.assertEqual(sel.n_spills, 0, f"selector spills {sel.n_spills} B")
        self.assertLessEqual(sel.n_regs, base.n_regs + 6,
                             f"selector costs {sel.n_regs - base.n_regs} regs "
                             f"({base.n_regs} -> {sel.n_regs})")


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestRankCutoff(CustomTestCase):
    """Bound-ordered truncation: the cutoff is what fits W, not a tuned width.

    When a lane fires more rows than the buffer holds, compact_fired keeps "the
    first W in position order", and archive position has nothing to do with
    whether a row beats the kept maximum. Measured offline on the r=0 dumps with
    the denominator fixed across W, the positional prefix retains 61.9% of the
    rows that truly beat max1 at the production W=4096 and bound order retains
    99.1%; at W=256 it is 24.9% against 81.6%.
    """

    def _hist_case(self, lens, ranks, W, nbin=None):
        from sglang.srt.layers.attention.vestigekv import defaults as D
        from sglang.srt.layers.attention.vestigekv.fused_prologue import (
            rank_cutoff,
        )
        nbin = D.RANK_NBIN if nbin is None else nbin
        dev = "cuda"
        P = len(lens)
        Am = max(lens)
        a_len = torch.tensor(lens, dtype=torch.int64, device=dev)
        a_off = torch.tensor([i * Am for i in range(P)], dtype=torch.int64,
                             device=dev)
        hit = torch.zeros(P * Am, dtype=torch.int8, device=dev)
        for i, (n, r) in enumerate(zip(lens, ranks)):
            hit[i * Am: i * Am + n] = r[:n].to(torch.int8).to(dev)
        nb = (Am + 1023) // 1024
        hist = torch.zeros(P, nb, nbin, dtype=torch.int32, device=dev)
        cutoff = torch.zeros(P, dtype=torch.int32, device=dev)
        counts = torch.zeros(P, nb, dtype=torch.int32, device=dev)
        rank_cutoff(hit, a_len, a_off, Am, W, hist, cutoff, counts, nbin=nbin)
        return hit, a_len, a_off, Am, cutoff, counts, nbin

    def test_the_cutoff_is_the_lowest_bin_that_fits(self):
        from sglang.srt.layers.attention.vestigekv import defaults as D
        torch.manual_seed(0)
        n = 4096
        r = (torch.rand(n) ** 3 * D.RANK_NBIN).to(torch.int32) + 1
        r[torch.rand(n) < 0.3] = 0
        W = 256
        hit, a_len, a_off, Am, cut, counts, nbin = self._hist_case([n], [r], W)
        k = int(cut[0])
        h = hit[:n].int()
        self.assertLessEqual(int((h >= k).sum()), W, "cutoff does not fit W")
        if k > 1:
            self.assertGreater(int((h >= k - 1).sum()), W,
                               "one bin lower would also fit; cutoff too high")
        self.assertEqual(int(counts[0].sum()), int((h >= k).sum()),
                         "per-bucket counts disagree with the cutoff")

    def test_a_lane_that_does_not_overflow_gets_cutoff_one(self):
        """The arm must be invisible where there is nothing to truncate, which
        is most steps: the branch arm's median fire is 12 rows."""
        r = torch.full((17,), 7, dtype=torch.int32)
        _, _, _, _, cut, counts, _ = self._hist_case([17], [r], 256)
        self.assertEqual(int(cut[0]), 1)
        self.assertEqual(int(counts[0].sum()), 17)

    def test_no_cutoff_fits_falls_back_to_the_top_bin_not_to_nothing(self):
        """If the strongest bin alone exceeds W, cut would run off the end and
        match nothing. Clamping to NBIN keeps the top bin and lets the
        positional prefix decide within it -- never worse than today."""
        from sglang.srt.layers.attention.vestigekv import defaults as D
        n = 4096
        r = torch.full((n,), D.RANK_NBIN, dtype=torch.int32)
        _, _, _, _, cut, counts, _ = self._hist_case([n], [r], 256)
        self.assertEqual(int(cut[0]), D.RANK_NBIN)
        self.assertEqual(int(counts[0].sum()), n, "fetched nothing")

    def test_cutoff_one_reproduces_the_unranked_compaction_exactly(self):
        """Second no-op proof, at the compaction rather than the scan: a cutoff
        of 1 selects hit >= 1, which is exactly hit != 0."""
        from sglang.srt.layers.attention.vestigekv.fused_prologue import (
            compact_fired,
        )
        dev = "cuda"
        torch.manual_seed(2)
        P, Am, W, L, NSLOT = 2, 2048, 64, 1, 4
        a_len = torch.tensor([Am, Am // 2], dtype=torch.int64, device=dev)
        a_off = torch.tensor([0, Am], dtype=torch.int64, device=dev)
        hit = (torch.rand(P * Am, device=dev) < 0.02).to(torch.int8)
        arch = torch.arange(P * Am, dtype=torch.int64, device=dev)
        li = torch.zeros(P, dtype=torch.int64, device=dev)
        slot = torch.arange(P, dtype=torch.int64, device=dev)
        nb = (Am + 1023) // 1024
        def run(cut):
            buf = torch.zeros(L, NSLOT, W, dtype=torch.int64, device=dev)
            ln = torch.zeros(L, NSLOT, dtype=torch.int64, device=dev)
            ovf = torch.zeros(L, NSLOT, dtype=torch.int32, device=dev)
            oc = torch.zeros(L, dtype=torch.int32, device=dev)
            scratch = (torch.zeros(P, nb, dtype=torch.int32, device=dev),
                       torch.zeros(P, nb, dtype=torch.int32, device=dev),
                       torch.zeros(P, dtype=torch.int32, device=dev))
            # per-lane bucket counts; lanes have different a_len, so each is
            # padded into its own row rather than stacked
            for p in range(P):
                hp = hit[int(a_off[p]):int(a_off[p]) + int(a_len[p])].int()
                for bi in range(nb):
                    seg = hp[bi * 1024:(bi + 1) * 1024]
                    scratch[0][p, bi] = int(seg.sum()) if seg.numel() else 0
            compact_fired(hit, arch, a_len, a_off, li, slot, buf, ln, ovf, oc,
                          scratch, Am, cutoff=cut)
            torch.cuda.synchronize()
            return ln.clone(), buf.clone()
        l0, b0 = run(None)
        l1, b1 = run(torch.ones(P, dtype=torch.int32, device=dev))
        torch.testing.assert_close(l1, l0, rtol=0, atol=0)
        torch.testing.assert_close(b1, b0, rtol=0, atol=0)
