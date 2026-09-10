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
    # build now returns pool row ids directly; no remap
    return t, kbuf, slots, keep, D


def _backfill(t, kbuf, slots):
    """Extend the closed-prefix caches the way the close path does.

    The watermark is `_pos_all.shape[0]`; only rows past it are projected.
    Extending blindly double-counts, which is exactly what the backend's
    `delta = closed_slots[cached:c1]` exists to prevent.
    """
    cached = 0 if t._pos_all is None else t._pos_all.shape[0]
    if cached < slots.numel():
        delta = slots[cached:]
        t.extend_closed(kbuf[delta], delta)
    return cached


class TestTierSidecarFromPool(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_refresh_sidecar_equals_the_slice_it_replaced(self):
        t, kbuf, slots, keep, D = _mk(6000, 11)
        n = slots.numel()

        # what extend_closed used to accumulate, kept here as the reference
        side_all_ref = kbuf[slots][:, D.KV_LORA_RANK :].clone()
        _backfill(t, kbuf, slots)

        g = torch.Generator(device="cuda").manual_seed(5)
        new_keep = torch.zeros(n, dtype=torch.bool, device="cuda")
        new_keep[torch.randperm(n, device="cuda", generator=g)[: n // 32]] = True
        t.refresh_membership(new_keep, kbuf)

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
        _backfill(t, kbuf, slots)
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
        _backfill(t, kbuf, slots)
        n = slots.numel()
        g = torch.Generator(device="cuda").manual_seed(7)
        nk2 = torch.zeros(n, dtype=torch.bool, device="cuda")
        nk2[torch.randperm(n, device="cuda", generator=g)[: n // 32]] = True
        t.drop_side()
        t.refresh_membership(nk2, kbuf)
        arch_idx = (~nk2).nonzero().flatten()
        self.assertTrue(
            torch.equal(t.side, kbuf[slots[arch_idx]][:, D.KV_LORA_RANK :]),
            "sidecar after a refresh does not match the pool rows arch names",
        )


class TestLazyKeptRows(CustomTestCase):
    """kept_rows is the pool's own rows under kept_slots, and nothing else."""

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_kept_rows_are_the_pool_rows_kept_slots_names(self):
        t, kbuf, slots, keep, D = _mk(6000, 31)
        want = kbuf[slots[keep]]
        self.assertGreater(want.shape[0], 0)
        self.assertTrue(
            torch.equal(t.kept_rows, want),
            "materialised kept rows differ from the pool rows kept_slots names",
        )
        t.drop_kept_rows()
        self.assertIsNone(t._kept_mat)
        self.assertTrue(torch.equal(t.kept_rows, want), "re-derivation differs")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_build_records_ids_and_the_rows_are_droppable(self):
        """Calibration does score the kept rows, so a build legitimately leaves
        the cache warm. What must hold is that the rows are recoverable from
        ids alone, so serving can drop them -- which is where the saving is."""
        t, kbuf, slots, keep, D = _mk(6000, 32)
        self.assertEqual(t.kept_slots.dtype, torch.int32)
        self.assertEqual(t.kept_slots.numel(), int(keep.sum()))
        t.drop_kept_rows()
        self.assertIsNone(t._kept_mat)
        self.assertTrue(torch.equal(t.kept_rows, kbuf[slots[keep]]))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_an_empty_kept_set_is_detected_without_a_gather(self):
        """The empty-kept check reads a length; reading it off the row table
        would gather the whole thing to learn one integer."""
        t, kbuf, slots, keep, D = _mk(6000, 34)
        t.drop_kept_rows()
        out = torch.zeros(2, 512, dtype=torch.int64, device="cuda")
        ol = torch.zeros(2, dtype=torch.int64, device="cuda")
        t.query_fixed(torch.randn(H, 576, device="cuda"), out, ol, 0)
        self.assertIsNotNone(t.kept_slots)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_refresh_leaves_nothing_materialised(self):
        t, kbuf, slots, keep, D = _mk(6000, 33)
        _backfill(t, kbuf, slots)
        n = slots.numel()
        g = torch.Generator(device="cuda").manual_seed(9)
        nk2 = torch.zeros(n, dtype=torch.bool, device="cuda")
        nk2[torch.randperm(n, device="cuda", generator=g)[: n // 32]] = True
        _ = t.kept_rows, t.side  # force both caches to exist
        t.refresh_membership(nk2, kbuf)
        self.assertIsNone(t._kept_mat, "refresh re-materialised the kept rows")
        self.assertIsNone(t._side_mat, "refresh re-materialised the sidecar")
        self.assertTrue(torch.equal(t.kept_rows, kbuf[slots[nk2]]))


class TestClosedPrefixWatermark(CustomTestCase):
    """The backfill watermark must never run ahead of the caches.

    `_pos_all.shape[0]` is what the close path backfills from. An earlier
    attempt set it eagerly while the projection caches stayed lazy, and the
    first serving close then indexed a 4096-row cache with full-prefix
    indices. build() now fills the caches and the watermark together, so the
    invariant to hold is simply that they agree -- before any close, after the
    first, and after one that adds nothing.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_watermark_matches_the_caches_across_the_first_close(self):
        t, kbuf, slots, keep, D = _mk(6000, 41)
        n = slots.numel()
        for name in ("_pos_all", "_csk_all", "_rho_all"):
            self.assertEqual(
                getattr(t, name).shape[0],
                n,
                f"{name} does not cover the built prefix",
            )

        # a close that adds nothing must add nothing
        self.assertEqual(_backfill(t, kbuf, slots), n)
        self.assertEqual(t._pos_all.shape[0], n, "backfill double-counted")

        # a close that extends the prefix
        g = torch.Generator(device="cuda").manual_seed(3)
        more = torch.randperm(POOL, device="cuda", generator=g)[: n + 2048]
        more[:n] = slots
        _backfill(t, kbuf, more)
        for name in ("_pos_all", "_csk_all", "_rho_all"):
            self.assertEqual(
                getattr(t, name).shape[0],
                n + 2048,
                f"{name} out of step with the watermark after a close",
            )
        self.assertTrue(torch.equal(t._pos_all[:n], slots), "prefix was rewritten")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_archive_selection_matches_a_direct_projection(self):
        """csk/rho as selections must equal projecting the archive directly."""
        t, kbuf, slots, keep, D = _mk(6000, 42)
        arch_idx = (~keep).nonzero().flatten()
        self.assertTrue(torch.equal(t._arch_idx, arch_idx))
        self.assertTrue(
            torch.equal(t.csk, t._csk_all.index_select(0, arch_idx)),
            "csk is not the archive's selection over the closed-prefix cache",
        )
        self.assertTrue(torch.equal(t.rho, t._rho_all.index_select(0, arch_idx)))
        self.assertTrue(torch.equal(t.arch, slots[arch_idx]), "arch must be pool ids")
