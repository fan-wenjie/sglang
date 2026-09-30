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
