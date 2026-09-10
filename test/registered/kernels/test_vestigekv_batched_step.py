"""The batched tier-2 step must fetch exactly what the per-pair form fetches.

The batching pads every (layer, slot) pair to the widest one, and padding is
where it can silently go wrong: a padded kept row winning max1 with a zero
score, or a padded archive row firing and fetching ARCH's zero-fill. These
tests run heterogeneous pair shapes against the per-pair query_fixed reference
and require identical fetch sets, then drive the padded regions directly.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="base-b-test-1-gpu-small")

H, R, W = 32, 64, 512


def _mk_tier(nk, a, seed, zp, thr_g):
    from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, device="cuda", generator=g)  # noqa: E731
    t = RecallTier(r=R, topj=-1)
    t.r, t.scale, t.zp, t.thr_g, t.built = R, 192**-0.5, zp, thr_g, True
    # storage contract mirrors RecallTier.build: side bf16, csk fp16,
    # kept_rows bf16, rho fp32 -- a fabricated fp32 tier would make the two
    # paths disagree for test-artifact reasons, not algorithmic ones
    # kept_slots is the index and kept_rows the view over it; a tier without
    # the index is not a tier any more (lengths are read off the index, so
    # asking one cannot gather a row table).
    t.kept_slots = torch.arange(nk, dtype=torch.int32, device="cuda")
    t.kept_rows = rnd(nk, 576).to(torch.bfloat16)
    t.V = rnd(R, 512)
    t.side = rnd(a, 64).to(torch.bfloat16)
    t.csk = rnd(a, R).to(torch.float16)
    t.rho = torch.rand(a, device="cuda", generator=g)
    t.arch = torch.randperm(500000, device="cuda")[:a] + 1  # never row 0
    t._scatter_buf = None
    t._qside_t = t._qsk_t = t._hit_buf = t._inf = None
    return t


class TestBatchedStepMatchesPerPair(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_heterogeneous_pairs_bit_identical(self):
        from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack

        torch.manual_seed(0)
        L, max_reqs = 3, 8
        shapes = [(900, 20000), (1100, 26000), (700, 9000)]  # (nk, a) per layer
        slots = [2, 5]
        tiers = {}
        for li in range(L):
            for slot in slots:
                nk, a = shapes[li]
                tiers[(li, slot)] = _mk_tier(
                    nk + slot,
                    a + 100 * slot,
                    seed=li * 10 + slot,
                    zp=float(li),
                    thr_g=-float("inf"),
                )
        qbuf = torch.randn(L, max_reqs, H, 576, device="cuda")
        fetch = torch.zeros(L, max_reqs, W, dtype=torch.int64, device="cuda")
        flen = torch.zeros(L, max_reqs, dtype=torch.int64, device="cuda")

        # per-pair reference
        ref_rows, ref_n = {}, {}
        out = torch.zeros(max_reqs, W, dtype=torch.int64, device="cuda")
        ol = torch.zeros(max_reqs, dtype=torch.int64, device="cuda")
        for (li, slot), t in tiers.items():
            t.query_fixed(qbuf[li, slot], out, ol, slot)
            ref_n[(li, slot)] = int(ol[slot])
            ref_rows[(li, slot)] = out[slot, : int(ol[slot])].clone()

        pairs = list(tiers.keys())
        pack = BatchedScanPack(pairs, [tiers[p] for p in pairs], qbuf, fetch, flen, H)
        pack.run()
        torch.cuda.synchronize()
        for li, slot in pairs:
            n = int(flen[li, slot])
            self.assertEqual(n, ref_n[(li, slot)], f"pair {(li, slot)}")
            self.assertTrue(
                torch.equal(fetch[li, slot, :n], ref_rows[(li, slot)]),
                f"pair {(li, slot)} fetch set differs",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_padded_archive_rows_never_fire(self):
        from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack

        # one tiny pair beside one wide pair; the tiny pair's padded region is
        # driven hard by an always-fire threshold on the wide pair
        torch.manual_seed(1)
        small = _mk_tier(50, 100, seed=1, zp=8.0, thr_g=-float("inf"))
        wide = _mk_tier(60, 30000, seed=2, zp=8.0, thr_g=-float("inf"))
        qbuf = torch.randn(1, 4, H, 576, device="cuda")
        fetch = torch.zeros(1, 4, W, dtype=torch.int64, device="cuda")
        flen = torch.zeros(1, 4, dtype=torch.int64, device="cuda")
        pack = BatchedScanPack([(0, 0), (0, 1)], [small, wide], qbuf, fetch, flen, H)
        pack.run()
        torch.cuda.synchronize()
        n_small = int(flen[0, 0])
        self.assertLessEqual(n_small, 100, "fired past the real archive")
        # arch values were shifted by +1, so a fetched 0 is ARCH padding
        self.assertTrue((fetch[0, 0, :n_small] > 0).all(), "fetched padding")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_padded_kept_rows_do_not_win_the_max(self):
        from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack

        # all real kept scores pushed far negative: with an unmasked zero-score
        # pad, max1 would be 0 and nothing could fire; masked, rows fire
        t_neg = _mk_tier(80, 5000, seed=3, zp=8.0, thr_g=-float("inf"))
        t_neg.kept_rows = t_neg.kept_rows - 100.0
        t_pad = _mk_tier(500, 5000, seed=4, zp=8.0, thr_g=-float("inf"))
        qbuf = torch.randn(1, 2, H, 576, device="cuda")
        fetch = torch.zeros(1, 2, W, dtype=torch.int64, device="cuda")
        flen = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
        pack = BatchedScanPack([(0, 0), (0, 1)], [t_neg, t_pad], qbuf, fetch, flen, H)
        pack.run()
        torch.cuda.synchronize()
        self.assertGreater(int(flen[0, 0]), 0, "pad row won the max: nothing fired")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_update_matches_a_fresh_pack_bit_identically(self):
        from sglang.srt.layers.attention.vestigekv.batched_step import (
            BatchedScanPack,
        )

        torch.manual_seed(5)
        max_reqs = 4
        qbuf = torch.randn(1, max_reqs, H, 576, device="cuda")
        f1 = torch.zeros(1, max_reqs, W, dtype=torch.int64, device="cuda")
        l1 = torch.zeros(1, max_reqs, dtype=torch.int64, device="cuda")
        f2, l2 = f1.clone(), l1.clone()
        # founding request (larger), then a SMALLER successor reusing the pack
        old = [_mk_tier(900, 20000, seed=11, zp=2.0, thr_g=-float("inf"))]
        new = [_mk_tier(700, 15000, seed=12, zp=1.0, thr_g=-float("inf"))]
        pack = BatchedScanPack([(0, 2)], old, qbuf, f1, l1, H)
        pack.run()  # founding contents exercised (also seeds any stale state)
        self.assertTrue(pack.fits([(0, 2)], new))
        pack.update([(0, 2)], new)
        pack.run()
        fresh = BatchedScanPack([(0, 2)], new, qbuf, f2, l2, H)
        fresh.run()
        torch.cuda.synchronize()
        n_u, n_f = int(l1[0, 2]), int(l2[0, 2])
        self.assertEqual(n_u, n_f)
        self.assertTrue(torch.equal(f1[0, 2, :n_u], f2[0, 2, :n_f]))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_update_refuses_nothing_but_fits_reports_overflow(self):
        from sglang.srt.layers.attention.vestigekv.batched_step import (
            BatchedScanPack,
        )

        qbuf = torch.randn(1, 4, H, 576, device="cuda")
        f = torch.zeros(1, 4, W, dtype=torch.int64, device="cuda")
        ln = torch.zeros(1, 4, dtype=torch.int64, device="cuda")
        small = [_mk_tier(100, 5000, seed=21, zp=2.0, thr_g=-float("inf"))]
        big = [_mk_tier(100, 9000, seed=22, zp=2.0, thr_g=-float("inf"))]
        pack = BatchedScanPack([(0, 0)], small, qbuf, f, ln, H)
        self.assertFalse(pack.fits([(0, 0)], big))  # 9000 > 5000*1.05


class TestKeptSetParityWithReference(CustomTestCase):
    """The port's tier-1 selection must equal the reference policy's math:
    per-4096-block sigma (fixed-window rFFT), global top-(rho*closed) plus
    sinks over CLOSED rows only, unclosed tail attended unconditionally. The
    whole-prefix formulation this replaced had an S-dependent cutoff and can
    NOT match; this test is what pins the migration.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_prefill_selection_matches_reference_policy(self):
        from sglang.srt.layers.attention.vestigekv import defaults as D
        from sglang.srt.layers.attention.vestigekv_mla_backend import (
            VestigeKVMLABackend,
        )

        torch.manual_seed(3)
        seq = 2 * D.CLOSE_BLOCK + 777  # two closable blocks + tail
        kbuf = torch.randn(seq + 10, 576, device="cuda")
        row_slots = torch.arange(seq, device="cuda")
        be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
        be.rho = D.RHO
        with unittest.mock.patch.object(
            VestigeKVMLABackend, "_full_arm", return_value=False
        ):
            got = be._arm_aware_kept(row_slots, kbuf, seq, 512)

        # reference policy math, independently written
        closed = 2 * D.CLOSE_BLOCK
        sigmas = []
        for b in range(2):
            side = kbuf[b * D.CLOSE_BLOCK : (b + 1) * D.CLOSE_BLOCK, 512:]
            f = torch.fft.rfft(side.float(), dim=0)
            f[D.LOWPASS_KAPPA :] = 0
            low = torch.fft.irfft(f, n=D.CLOSE_BLOCK, dim=0)
            sigmas.append((side.float() - low).norm(dim=-1))
        sig = torch.cat(sigmas)
        m = max(1, round(D.RHO * closed))
        keep = torch.zeros(closed, dtype=torch.bool, device="cuda")
        keep[sig.topk(m).indices] = True
        keep[: D.SINKS] = True
        expect = torch.cat([row_slots[:closed][keep], row_slots[closed:seq]])
        self.assertTrue(torch.equal(torch.sort(got).values, torch.sort(expect).values))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_short_prefix_attends_everything(self):
        from sglang.srt.layers.attention.vestigekv import defaults as D
        from sglang.srt.layers.attention.vestigekv_mla_backend import (
            VestigeKVMLABackend,
        )

        seq = D.CLOSE_BLOCK - 1
        kbuf = torch.randn(seq, 576, device="cuda")
        row_slots = torch.arange(seq, device="cuda")
        be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
        be.rho = D.RHO
        with unittest.mock.patch.object(
            VestigeKVMLABackend, "_full_arm", return_value=False
        ):
            got = be._arm_aware_kept(row_slots, kbuf, seq, 512)
        self.assertEqual(got.shape[0], seq)


if __name__ == "__main__":
    unittest.main()


class TestBatchedEmptyKept(CustomTestCase):
    """All-empty-kept capture must not crash and must fire the whole archive."""

    @unittest.skipUnless(torch.cuda.is_available(), "batched pack is device-only")
    def test_all_empty_kept_fires_archive(self):
        import sglang.srt.layers.attention.vestigekv.defaults as D
        from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack

        torch.manual_seed(0)
        dev = "cuda"
        H, A = 8, 96

        class T:
            pass

        def mk():
            t = T()
            t.kept_rows = torch.zeros(0, D.LATENT_DIM, device=dev, dtype=torch.bfloat16)
            t.V = torch.linalg.qr(torch.randn(D.KV_LORA_RANK, 16, device=dev))[
                0
            ].T.contiguous()
            t.side = torch.randn(A, D.SIDECAR_DIM, device=dev, dtype=torch.bfloat16)
            t.csk = torch.randn(A, 16, device=dev, dtype=torch.float16)
            t.rho = torch.rand(A, device=dev)
            t.arch = torch.arange(A, device=dev)
            t.a_len = A
            t.thr_g = 0.0
            t.zp = 4.0
            t.scale = 1.0 / (D.LATENT_DIM**0.5)
            t.r = 16
            t.version = 0
            return t

        tiers = [mk(), mk()]
        pairs = [(0, 0), (1, 1)]  # (li, slot) into the per-layer stacks
        # buffers match _alloc_recall_bufs: [n_li, max_slots, ...]
        qbuf = torch.randn(2, 2, H, D.LATENT_DIM, device=dev)
        fetch_buf = torch.zeros(2, 2, A + 8, dtype=torch.int64, device=dev)
        fetch_len = torch.zeros(2, 2, dtype=torch.int64, device=dev)
        pack = BatchedScanPack(pairs, tiers, qbuf, fetch_buf, fetch_len, H)
        pack.run()  # must not raise (was: zero-width kr -> max(-1) crash)
        # empty kept => whole archive eligible; both pairs fire > 0 rows
        self.assertGreater(int(fetch_len.sum()), 0)


class TestArchiveArenaAddressing(CustomTestCase):
    """A tight shared arena must fetch what the per-pair reference fetches.

    The capacity pack keeps every pair's archive rows in ONE table addressed
    by a_off[p], sized by what the KV pool can hold rather than by
    max_bs x max_context. That is purely a change of address arithmetic in two
    Triton kernels, so the check is against query_fixed -- the ABSOLUTE
    reference. Comparing the two layouts against each other would not do: a
    base that is ignored collapses both to offset zero and the two agree while
    both are wrong (that mutation was run, and a differential test passed it).
    Archive lengths are unequal so every pair but the first sits at a non-zero
    offset that no aliasing can reproduce.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_tight_arena_matches_the_per_pair_reference(self):
        from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack

        torch.manual_seed(7)
        lens = [9000, 5000, 7000, 3000]  # unequal on purpose: offsets differ
        n_lids, max_reqs = 2, 2
        P, Am = n_lids * max_reqs, max(lens)
        qbuf = torch.randn(n_lids, max_reqs, H, 576, device="cuda")
        f = torch.zeros(n_lids, max_reqs, W, dtype=torch.int64, device="cuda")
        ln = torch.zeros(n_lids, max_reqs, dtype=torch.int64, device="cuda")

        pairs, tiers = [], []
        for li in range(n_lids):
            for slot in range(max_reqs):
                p = li * max_reqs + slot
                pairs.append((li, slot))
                tiers.append(
                    _mk_tier(800 + p, lens[p], seed=100 + p, zp=1.0, thr_g=-1e30)
                )

        ref_rows, ref_n = [], []
        out = torch.zeros(max_reqs, W, dtype=torch.int64, device="cuda")
        ol = torch.zeros(max_reqs, dtype=torch.int64, device="cuda")
        for (li, slot), t in zip(pairs, tiers):
            t.query_fixed(qbuf[li, slot], out, ol, slot)
            ref_n.append(int(ol[slot]))
            ref_rows.append(out[slot, : int(ol[slot])].clone())

        # arena = the exact sum, i.e. the tightest the pool bound can ever be
        pack = BatchedScanPack.at_capacity(
            P, 900, Am, R, H, qbuf, f, ln, 0, arena=sum(lens)
        )
        self.assertTrue(pack.fits(pairs, tiers))
        pack.update(pairs, tiers)
        pack.run()
        torch.cuda.synchronize()

        self.assertGreater(sum(ref_n), 0, "a reference that fetched nothing")
        for i, (li, slot) in enumerate(pairs):
            n = int(ln[li, slot])
            self.assertEqual(n, ref_n[i], f"pair {(li, slot)} fetch count")
            self.assertTrue(
                torch.equal(f[li, slot, :n], ref_rows[i]),
                f"pair {(li, slot)} fetch set differs from query_fixed",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_arena_smaller_than_the_sum_is_refused_not_corrupted(self):
        from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack

        lens = [9000, 5000, 7000, 3000]
        P, Am = 4, max(lens)
        qbuf = torch.zeros(2, 2, H, 576, device="cuda")
        f = torch.zeros(2, 2, W, dtype=torch.int64, device="cuda")
        ln = torch.zeros(2, 2, dtype=torch.int64, device="cuda")
        pack = BatchedScanPack.at_capacity(
            P, 900, Am, R, H, qbuf, f, ln, 0, arena=sum(lens) - 1
        )
        pairs = [(li, s) for li in range(2) for s in range(2)]
        tiers = [
            _mk_tier(800 + p, lens[p], seed=200 + p, zp=1.0, thr_g=-1e30)
            for p in range(4)
        ]
        # fits() is the gate; update() past it must still refuse rather than
        # write outside the arena, so the overflow can never be silent.
        self.assertFalse(pack.fits(pairs, tiers))
        with self.assertRaises(RuntimeError):
            pack.update(pairs, tiers)


class TestKeptRowsFromPool(CustomTestCase):
    """Scoring kept rows in place in the KV pool must equal scoring a snapshot.

    The pack used to carry its own bf16 copy of every kept row -- 87% of the
    pack at bs16, and every byte a duplicate, because a latent row is written
    once when its token enters the pool and never rewritten. Handing the
    kernel 4 bytes of row id plus the pool's per-layer base pointer has to
    produce the SAME fetch set, so the check is against query_fixed, which
    scores the snapshot.

    The fixture is built by RecallTier.build over a fake per-layer pool, with
    LOW-RANK, heavy-tailed rows. Independent Gaussian rows do not work: every
    archived row is then genuinely competitive with the kept set, the whole
    archive fires for any query, and the fetch set stops depending on the kept
    scores at all -- a test written that way passed a deliberately wrong base
    pointer. Here roughly a fifth of the archive fires, so max1 (and with it
    every kept row the kernel reads) decides the answer.
    """

    FETCH_W = 8192  # above the observed fire, so truncation hides nothing

    def _pool(self, n_lids, pool_rows, seed):
        """Separate per-layer allocations, as the real MLA pool has."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        bufs = []
        for _ in range(n_lids):
            B = torch.randn(24, 576, device="cuda", generator=g)
            A = torch.randn(pool_rows, 24, device="cuda", generator=g)
            mag = torch.rand(pool_rows, 1, device="cuda", generator=g) ** 4 * 8 + 0.2
            bufs.append(((A @ B) / 24**0.5 * mag).to(torch.bfloat16))
        return bufs

    def _tier(self, kbuf, n_tok, pool_rows, seed):
        from sglang.srt.layers.attention.vestigekv import defaults as D
        from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

        g = torch.Generator(device="cuda").manual_seed(seed)
        slots = torch.randperm(pool_rows, device="cuda", generator=g)[:n_tok]
        nkeep = max(1, int(D.RHO * n_tok))
        keep = torch.zeros(n_tok, dtype=torch.bool, device="cuda")
        # tier-1 keeps the high-sigma rows, so the kept set is the large ones
        keep[kbuf[slots].float().norm(dim=-1).topk(nkeep).indices] = True
        qcal = torch.randn(32, H, 576, device="cuda", generator=g)
        qpos = torch.randint(0, n_tok, (32,), device="cuda", generator=g)
        t = RecallTier(r=R, topj=-1)
        t.build(kbuf, slots, keep, qcal, qpos)
        return t

    def _fixture(self):
        n_lids, max_reqs, pool_rows = 2, 2, 8192
        n_toks = [6000, 4000, 5200, 3100]  # unequal, and none a block multiple
        bufs = self._pool(n_lids, pool_rows, seed=31)
        pairs, tiers = [], []
        for li in range(n_lids):
            for slot in range(max_reqs):
                p = li * max_reqs + slot
                pairs.append((li, slot))
                tiers.append(self._tier(bufs[li], n_toks[p], pool_rows, 300 + p))
        qbuf = torch.randn(n_lids, max_reqs, H, 576, device="cuda")
        return bufs, pairs, tiers, qbuf

    def _run(self, pairs, tiers, qbuf, pool_bases, pool_rows=None, mode=None):
        from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack

        n_lids, max_reqs = qbuf.shape[0], qbuf.shape[1]
        WW = self.FETCH_W
        arch_lens = [t.side.shape[0] for t in tiers]
        f = torch.zeros(n_lids, max_reqs, WW, dtype=torch.int64, device="cuda")
        ln = torch.zeros(n_lids, max_reqs, dtype=torch.int64, device="cuda")
        pack = BatchedScanPack.at_capacity(
            n_lids * max_reqs,
            max(t.kept_rows.shape[0] for t in tiers),
            max(arch_lens),
            R,
            H,
            qbuf,
            f,
            ln,
            0,
            arena=sum(arch_lens),
            pool_bases=pool_bases,
            pool_rows=pool_rows,
        )
        if mode is not None:
            pack.pool_mode = mode
        self.assertEqual(pack.kr is None, pool_bases is not None)
        self.assertTrue(pack.fits(pairs, tiers))
        pack.update(pairs, tiers)
        pack.run()
        torch.cuda.synchronize()
        return f, ln

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_from_pool_is_bit_identical_to_the_snapshot(self):
        torch.manual_seed(31)
        bufs, pairs, tiers, qbuf = self._fixture()
        for t in tiers:
            self.assertIsNotNone(t.kept_slots, "build must record the pool row ids")

        # the fixture must SELECT: on a saturated one the fetch set stops
        # depending on the kept scores, and a wrong base pointer goes unseen
        arch_lens = [t.side.shape[0] for t in tiers]
        snap_f, snap_n = self._run(pairs, tiers, qbuf, None)
        for i, (li, slot) in enumerate(pairs):
            n = int(snap_n[li, slot])
            self.assertTrue(
                0 < n < arch_lens[i],
                f"pair {i} fired {n} of {arch_lens[i]} -- fixture is not selective",
            )

        pool_f, pool_n = self._run(
            pairs, tiers, qbuf, [b.data_ptr() for b in bufs], bufs[0].shape[0]
        )
        self.assertTrue(torch.equal(snap_n, pool_n), "fetch counts differ")
        for li, slot in pairs:
            n = int(snap_n[li, slot])
            self.assertTrue(
                torch.equal(snap_f[li, slot, :n], pool_f[li, slot, :n]),
                f"pair {(li, slot)} fetch set differs between snapshot and pool",
            )
        del bufs

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_from_pool_tracks_the_eager_reference(self):
        """Anchor to query_fixed, which the fused path matches only closely.

        The fused kernel and the eager reference disagree on a handful of rows
        at the fire boundary: measured here at 0.00 / 0.10 / 0.28 / 0.48% of
        the fired set. That gap is older than this pool-read -- it reproduces
        identically on the four commits before it -- and was invisible while
        every fixture fired its whole archive. It is recorded in DEFECTS.md.

        The bar is 1% of the fired set: twice the observed worst, and still
        two orders of magnitude below what a misaddressed read costs (a wrong
        base pointer scores different rows entirely, and disagrees on nearly
        all of them).
        """
        torch.manual_seed(31)
        bufs, pairs, tiers, qbuf = self._fixture()
        WW = self.FETCH_W
        out = torch.zeros(qbuf.shape[1], WW, dtype=torch.int64, device="cuda")
        ol = torch.zeros(qbuf.shape[1], dtype=torch.int64, device="cuda")
        ref = []
        for (li, slot), t in zip(pairs, tiers):
            t.query_fixed(qbuf[li, slot], out, ol, slot)
            ref.append(set(out[slot, : int(ol[slot])].tolist()))

        f, ln = self._run(
            pairs, tiers, qbuf, [b.data_ptr() for b in bufs], bufs[0].shape[0]
        )
        for i, (li, slot) in enumerate(pairs):
            got = set(f[li, slot, : int(ln[li, slot])].tolist())
            sym = len(got ^ ref[i])
            self.assertLessEqual(
                sym,
                0.01 * len(ref[i]),
                f"pair {i}: {sym} of {len(ref[i])} rows "
                f"({100 * sym / len(ref[i]):.2f}%) differ from the eager reference",
            )
        del bufs

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_both_pool_read_modes_match_the_snapshot(self):
        """Whichever way the kernel reads the pool, it reads the same rows.

        Kept rows come out of the pool either through a TMA row gather or
        through an indirect load, picked at runtime by whether the gather
        compiles and runs here. Only one of those runs on any given machine,
        so the other would go untested -- both are pinned here, against the
        snapshot the pack used to carry.
        """
        from sglang.srt.layers.attention.vestigekv import fused_prologue as FP

        torch.manual_seed(31)
        bufs, pairs, tiers, qbuf = self._fixture()
        bases, rows = [b.data_ptr() for b in bufs], bufs[0].shape[0]
        snap_f, snap_n = self._run(pairs, tiers, qbuf, None)
        arch_lens = [t.side.shape[0] for t in tiers]
        for i, (li, slot) in enumerate(pairs):
            n = int(snap_n[li, slot])
            self.assertTrue(
                0 < n < arch_lens[i],
                f"pair {i} fired {n} of {arch_lens[i]} -- fixture is not selective",
            )
        available = {1}
        if FP._pool_read_mode(qbuf.device) == 2:
            available.add(2)
        for mode in sorted(available):
            f, ln = self._run(pairs, tiers, qbuf, bases, rows, mode=mode)
            self.assertTrue(torch.equal(snap_n, ln), f"mode {mode}: counts differ")
            for li, slot in pairs:
                n = int(snap_n[li, slot])
                self.assertTrue(
                    torch.equal(snap_f[li, slot, :n], f[li, slot, :n]),
                    f"mode {mode}: pair {(li, slot)} fetch set differs",
                )
        del bufs


class TestProjectionsFromTierCache(CustomTestCase):
    """The pack may read the sketch projections instead of copying them.

    csk is a selection over the tier's closed-prefix cache, so a compacted
    copy in the arena is the same numbers twice -- 128 bytes per archived row.
    The pack can carry the 4-byte row index and read through it, the same
    pattern the kept rows and sidecars use against the KV pool, except the
    cache is reallocated by torch.cat at every block close. That is why its
    address comes from a device table refreshed by update() rather than from
    anything baked into a graph, and why the pack holds a reference to the
    cache while it points at one.

    All three ways of getting csk must produce the same fetch set.
    """

    FETCH_W = 8192

    def _fixture(self, n_lids=2, max_reqs=2, pool_rows=8192):
        from sglang.srt.layers.attention.vestigekv import defaults as D
        from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

        g = torch.Generator(device="cuda").manual_seed(51)
        bufs = []
        for _ in range(n_lids):
            B = torch.randn(24, 576, device="cuda", generator=g)
            A = torch.randn(pool_rows, 24, device="cuda", generator=g)
            mag = torch.rand(pool_rows, 1, device="cuda", generator=g) ** 4 * 8 + 0.2
            bufs.append(((A @ B) / 24**0.5 * mag).to(torch.bfloat16))
        pairs, tiers = [], []
        n_toks = [6000, 4000, 5200, 3100]
        for li in range(n_lids):
            for slot in range(max_reqs):
                p = li * max_reqs + slot
                kb = bufs[li]
                gg = torch.Generator(device="cuda").manual_seed(300 + p)
                slots = torch.randperm(pool_rows, device="cuda", generator=gg)[
                    : n_toks[p]
                ]
                nkeep = max(1, int(D.RHO * n_toks[p]))
                keep = torch.zeros(n_toks[p], dtype=torch.bool, device="cuda")
                keep[kb[slots].float().norm(dim=-1).topk(nkeep).indices] = True
                qcal = torch.randn(32, H, 576, device="cuda", generator=gg)
                qpos = torch.randint(0, n_toks[p], (32,), device="cuda", generator=gg)
                t = RecallTier(r=R, topj=-1)
                t.build(kb, slots, keep, qcal, qpos)
                pairs.append((li, slot))
                tiers.append(t)
        qbuf = torch.randn(n_lids, max_reqs, H, 576, device="cuda")
        return bufs, pairs, tiers, qbuf

    def _run(self, pairs, tiers, qbuf, bufs, from_tier, mode=None, slack=0):
        from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack

        n_lids, max_reqs = qbuf.shape[0], qbuf.shape[1]
        arch_lens = [t.arch.shape[0] for t in tiers]
        f = torch.zeros(
            n_lids, max_reqs, self.FETCH_W, dtype=torch.int64, device="cuda"
        )
        ln = torch.zeros(n_lids, max_reqs, dtype=torch.int64, device="cuda")
        pack = BatchedScanPack.at_capacity(
            n_lids * max_reqs,
            max(t.kept_rows.shape[0] for t in tiers) + slack,
            max(arch_lens),
            R,
            H,
            qbuf,
            f,
            ln,
            0,
            arena=sum(arch_lens) + slack,
            pool_bases=[b.data_ptr() for b in bufs],
            pool_row=576,
            pool_rows=bufs[0].shape[0],
            csk_from_tier=from_tier,
        )
        if mode is not None:
            pack.csk_mode = mode
        self.assertEqual(pack.csk is None, from_tier)
        pack.update(pairs, tiers)
        pack.run()
        torch.cuda.synchronize()
        return pack, f, ln

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_read_through_matches_the_packed_copy(self):
        from sglang.srt.layers.attention.vestigekv import fused_prologue as FP

        torch.manual_seed(51)
        bufs, pairs, tiers, qbuf = self._fixture()
        _, ref_f, ref_n = self._run(pairs, tiers, qbuf, bufs, from_tier=False)
        arch_lens = [t.arch.shape[0] for t in tiers]
        for i, (li, slot) in enumerate(pairs):
            n = int(ref_n[li, slot])
            self.assertTrue(
                0 < n < arch_lens[i],
                f"pair {i} fired {n} of {arch_lens[i]} -- fixture is not selective",
            )
        modes = {1}
        if FP._pool_read_mode(qbuf.device) == 2:
            modes.add(2)
        for m in sorted(modes):
            _, f, ln = self._run(pairs, tiers, qbuf, bufs, from_tier=True, mode=m)
            self.assertTrue(torch.equal(ref_n, ln), f"csk mode {m}: counts differ")
            for li, slot in pairs:
                n = int(ref_n[li, slot])
                self.assertTrue(
                    torch.equal(ref_f[li, slot, :n], f[li, slot, :n]),
                    f"csk mode {m}: pair {(li, slot)} fetch set differs",
                )
        del bufs

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_pack_holds_the_cache_alive_and_tracks_reallocation(self):
        """cbase is a raw address; the pack must keep the cache from being
        freed, and must pick up the new address after a block close."""
        torch.manual_seed(51)
        bufs, pairs, tiers, qbuf = self._fixture()
        # slack for the close below, which grows one pair's archive
        pack, _, _ = self._run(pairs, tiers, qbuf, bufs, from_tier=True, slack=4096)
        self.assertEqual(len(pack._csk_refs), len(tiers), "caches not retained")
        for t, ref in zip(tiers, pack._csk_refs):
            self.assertIs(ref, t._csk_all)
        before = pack.cbase.clone()

        # a close reallocates the cache; update() must refresh cbase
        t0 = tiers[0]
        extra = torch.arange(64, device="cuda", dtype=torch.int64)
        t0.extend_closed(bufs[0][extra], extra)
        self.assertIsNot(t0._csk_all, pack._csk_refs[0], "cat did not reallocate")
        n_closed = t0._pos_all.shape[0]
        k2 = torch.zeros(n_closed, dtype=torch.bool, device="cuda")
        k2[: max(1, n_closed // 32)] = True
        t0.refresh_membership(k2, bufs[0])
        self.assertTrue(pack.fits(pairs, tiers), "capacity check refused the update")
        pack.update(pairs, tiers)
        torch.cuda.synchronize()
        self.assertNotEqual(
            int(before[0]), int(pack.cbase[0]), "cbase kept a stale address"
        )
        self.assertEqual(int(pack.cbase[0]), t0._csk_all.data_ptr())
