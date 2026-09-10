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
