"""Tier-side storage invariants.

The tier used to carry `_side_all`, a [closed, 64] bf16 copy of every closed
row's tail, purely so a block close could slice the archive's sidecars out of
it. Those bytes are the pool's own -- `_pos_all` names every closed row -- so
the copy is gone and the sidecars are re-read by row id. That is only correct
if the re-read is EXACTLY what the slice used to produce, which is what these
tests pin.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, suite="base-b-test-1-gpu-small")

H, R, POOL = 32, 64, 8192


def _mk(n_tok, seed):
    from sglang.srt.layers.attention.vestigekv import defaults as D
    from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

    g = torch.Generator(device="cuda").manual_seed(seed)
    B = torch.randn(24, 576, device="cuda", generator=g)
    A = torch.randn(POOL, 24, device="cuda", generator=g)
    mag = torch.rand(POOL, 1, device="cuda", generator=g) ** 4 * 8 + 0.2
    kbuf = ((A @ B) / 24**0.5 * mag).to(torch.bfloat16)
    slots = torch.randperm(POOL, device="cuda", generator=g)[:n_tok]
    nkeep = max(1, int(D.RHO * n_tok))
    keep = torch.zeros(n_tok, dtype=torch.bool, device="cuda")
    keep[kbuf[slots].float().norm(dim=-1).topk(nkeep).indices] = True
    qcal = torch.randn(32, H, 576, device="cuda", generator=g)
    qpos = torch.randint(0, n_tok, (32,), device="cuda", generator=g)
    t = RecallTier(r=R, topj=-1)
    t.build(kbuf, slots, keep, qcal, qpos)
    t.arch = slots[t.arch]  # the backend's remap; see vestigekv_mla_backend
    return t, kbuf, slots, keep, D


class TestTierSidecarFromPool(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_refresh_sidecar_equals_the_slice_it_replaced(self):
        t, kbuf, slots, keep, D = _mk(6000, 11)
        n = slots.numel()

        # what extend_closed used to accumulate, kept here as the reference
        side_all_ref = kbuf[slots][:, D.KV_LORA_RANK :].clone()
        t.extend_closed(kbuf[slots], slots)

        g = torch.Generator(device="cuda").manual_seed(5)
        new_keep = torch.zeros(n, dtype=torch.bool, device="cuda")
        new_keep[torch.randperm(n, device="cuda", generator=g)[: n // 32]] = True
        kept_slots = slots[new_keep]
        t.refresh_membership(new_keep, kbuf[kept_slots], kbuf)

        arch_idx = (~new_keep).nonzero().flatten()
        self.assertGreater(arch_idx.numel(), 0, "an empty archive proves nothing")
        self.assertTrue(
            torch.equal(t.side, side_all_ref[arch_idx].contiguous()),
            "pool re-read differs from the _side_all slice it replaced",
        )
        self.assertTrue(torch.equal(t.arch, slots[arch_idx]))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_tier_no_longer_carries_a_sidecar_copy_of_the_prefix(self):
        """The saving is the point: nothing may re-introduce a [closed, 64]."""
        t, kbuf, slots, keep, D = _mk(6000, 12)
        t.extend_closed(kbuf[slots], slots)
        self.assertFalse(
            hasattr(t, "_side_all") and getattr(t, "_side_all") is not None,
            "_side_all is back; the per-request saving it costs grows with context",
        )
        closed = slots.numel()
        for name in ("_csk_all", "_rho_all", "_pos_all"):
            v = getattr(t, name)
            self.assertEqual(v.shape[0], closed, f"{name} must span the closed prefix")


class TestLazySidecar(CustomTestCase):
    """Dropping the sidecar must be recoverable, which is what makes it safe.

    An earlier optimisation released tier operands outright and stranded the
    next reader on a None; that one had to be reverted. This one drops a cache
    whose property re-derives it from the pool, so the same call pattern that
    broke then is merely a gather now. The test drives exactly that pattern.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_drop_then_use_re_derives_the_same_bytes(self):
        t, kbuf, slots, keep, D = _mk(6000, 21)
        before = t.side.clone()
        self.assertGreater(before.shape[0], 0)

        t.drop_side()
        self.assertIsNone(t._side_mat, "drop must actually release the cache")

        after = t.side  # the pattern that stranded the released version
        self.assertTrue(
            torch.equal(before, after),
            "re-derived sidecar differs from the materialised one",
        )
        self.assertIs(t.side, after, "second read must hit the cache, not re-gather")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_drop_survives_a_membership_refresh(self):
        t, kbuf, slots, keep, D = _mk(6000, 22)
        t.extend_closed(kbuf[slots], slots)
        n = slots.numel()
        g = torch.Generator(device="cuda").manual_seed(7)
        nk2 = torch.zeros(n, dtype=torch.bool, device="cuda")
        nk2[torch.randperm(n, device="cuda", generator=g)[: n // 32]] = True
        t.drop_side()
        t.refresh_membership(nk2, kbuf[slots[nk2]], kbuf)
        arch_idx = (~nk2).nonzero().flatten()
        self.assertTrue(
            torch.equal(t.side, kbuf[slots[arch_idx]][:, D.KV_LORA_RANK :]),
            "sidecar after a refresh does not match the pool rows arch names",
        )
