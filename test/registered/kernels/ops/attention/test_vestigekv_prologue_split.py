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
