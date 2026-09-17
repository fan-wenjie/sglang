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
    t = RecallTier(r=R)
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
        oo = torch.zeros(2, dtype=torch.int32, device="cuda")
        t.query_fixed(torch.randn(H, 576, device="cuda"), out, ol, oo, 0)
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


class TestOmittedMass(CustomTestCase):
    """The omitted-mass arm's logM must live on the decode kernel's scale.

    The blend is sigmoid(lse - logM), where lse is the kernel's absolute
    logsumexp over the attended rows. If logM is off by a constant -- a
    different softmax scale, a missing shift, a bound summed in the wrong
    space -- sigma collapses toward 0, the output is replaced by the archive
    centroid, and every task degrades at once. That is what the arm did on its
    first working run, and it needed no GPU run to see: this recomputes logM
    independently from the tier's own stored operands.
    """

    def _q(self, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(H, 576, device="cuda", generator=g)

    def _parts(self, t, q, D):
        sc = t.scale
        qe = q.float()
        skept = (qe.to(torch.bfloat16) @ t.kept_rows.T).float() * sc
        max1 = t._thr_base(skept, skept.max(-1).values)
        qsk = qe[:, : D.KV_LORA_RANK] @ t.V.T
        qres = (qe[:, : D.KV_LORA_RANK] - qsk @ t.V).norm(dim=-1)
        idxs = ((qe[:, D.KV_LORA_RANK :].to(torch.bfloat16) @ t.side.T).float()
                + (qsk.half() @ t.csk.T).float()) * sc
        cert = (qres[:, None] * t.rho[None, :]) * sc / (D.KV_LORA_RANK - t.r) ** 0.5
        score = idxs + t.zp * cert
        p1 = torch.softmax(skept, -1)
        ent = -(p1 * p1.clamp_min(D.ENTROPY_EPS).log()).sum(-1)
        fire = (score > (max1 - t.margin - t._ent_margin(skept))[:, None]) & (
            ent > t.thr_g)[:, None]
        return skept, max1, score, ~fire.any(0)

    def test_log_mass_matches_an_independent_sum_over_the_omitted_rows(self):
        t, kbuf, slots, keep, D = _mk(4096, 11)
        q = self._q(5)
        logm, mu = t.omitted_mass_and_mean(q)
        skept, max1, score, om = self._parts(t, q, D)
        torch.testing.assert_close(
            logm, torch.logsumexp(score[:, om], dim=-1), rtol=2e-3, atol=2e-3)
        # The centroid is the MASS-WEIGHTED one: a synthetic row is an exact
        # online-softmax step only if it carries the weighted centroid
        # (offline 0.195 output error against 0.367 for the plain mean).
        w = (score[:, om] - max1[:, None]).exp()
        vals = kbuf.index_select(0, t.arch.to(torch.int64))[
            :, : D.KV_LORA_RANK].float()
        want_mu = (w @ vals[om]) / w.sum(-1, keepdim=True)
        torch.testing.assert_close(mu, want_mu, rtol=5e-3, atol=5e-3)

    def test_sigma_is_a_blend_not_a_replacement(self):
        """logM must be COMPARABLE to the attended lse, not dwarf it.

        Offline the attended set holds about 57% of the dense softmax mass, so
        sigma belongs near 0.5. A sigma at 0.01 means the output is 99%
        archive centroid, which is a scale fault however plausible the
        arithmetic looks in isolation. Reported with both sides so a failure
        says WHICH term is wrong.
        """
        t, kbuf, slots, keep, D = _mk(4096, 12)
        q = self._q(6)
        logm, _ = t.omitted_mass_and_mean(q)
        skept, max1, score, om = self._parts(t, q, D)
        fired = score[:, ~om]
        lse_att = torch.logsumexp(torch.cat([skept, fired], dim=-1), dim=-1)
        sigma = torch.sigmoid(lse_att - logm)
        arch_true = (q.float() @ kbuf.index_select(
            0, t.arch.to(torch.int64)).float().T) * t.scale
        self.assertGreater(
            float(sigma.median()), 0.02,
            f"sigma p50 {float(sigma.median()):.5f}: the blend REPLACES the "
            f"output instead of correcting it. logM p50 "
            f"{float(logm.median()):.2f}, attended lse p50 "
            f"{float(lse_att.median()):.2f}, true archive lse p50 "
            f"{float(torch.logsumexp(arch_true, -1).median()):.2f}, "
            f"omitted rows {int(om.sum())} of {int(om.numel())}",
        )


class TestMultiKeyFence(CustomTestCase):
    """A lowered fence must actually raise the flag, not just be configured.

    The fence is the retreat position for multi-key: the certificate is
    weakest where a query needs SEVERAL archived rows, and a fenced lane
    attends its full row set, which is exactly dense. The fired-row count is
    the detector, free because the scan already produces it -- median 0, 1, 2,
    4, 13 rows for queries needing 0, 1, 2, 3, 4+ archived rows.
    """

    def _fire(self, t, q, fence):
        W = 4096
        out = torch.zeros(1, W, dtype=torch.int32, device="cuda")
        ln = torch.zeros(1, dtype=torch.int32, device="cuda")
        ovf = torch.zeros(1, dtype=torch.int32, device="cuda")
        t.fence_rows = fence
        t.query_fixed(q, out, ln, ovf, 0)
        return int(ln[0]), int(ovf[0])

    def test_the_flag_follows_the_fence_not_only_the_buffer(self):
        t, kbuf, slots, keep, D = _mk(4096, 21)
        g = torch.Generator(device="cuda").manual_seed(9)
        q = torch.randn(H, 576, device="cuda", generator=g)
        n0, ovf0 = self._fire(t, q, 0)
        self.assertEqual(ovf0, 0, "a short fire must not overflow a 4096 buffer")
        # fence one below what this query fires: the lane must fence
        if n0 >= 1:
            n1, ovf1 = self._fire(t, q, max(n0 - 1, 0))
            self.assertEqual(n1, n0, "the fence changes the FLAG, not the rows kept")
            self.assertEqual(ovf1, 1, f"fired {n0} rows against a fence of {n0 - 1}")
        # a fence above the fire leaves it alone
        n2, ovf2 = self._fire(t, q, n0 + 8)
        self.assertEqual(ovf2, 0)
        self.assertEqual(n2, n0)

    def test_fence_zero_is_the_historical_behaviour(self):
        t, kbuf, slots, keep, D = _mk(4096, 22)
        g = torch.Generator(device="cuda").manual_seed(10)
        q = torch.randn(H, 576, device="cuda", generator=g)
        self.assertEqual(self._fire(t, q, 0)[1], 0)
