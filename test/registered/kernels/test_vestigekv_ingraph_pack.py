"""The capacity pack (in-graph scan) must equal the legacy pack bit-for-bit.

BatchedScanPack.at_capacity is built before any tier exists and refreshed only
through update(); the decode model graph bakes its addresses. These tests pin
the three properties that make that sound: a partially occupied capacity pack
fetches exactly what a fresh legacy pack fetches; placeholder pairs write only
the trash slot; and a CUDA graph captured over the empty pack, then update()d,
replays the same fetch sets the eager kernels produce.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="base-b-test-1-gpu-small")

H, R, W = 32, 64, 512
L, MAX_REQS = 3, 8
TRASH = MAX_REQS  # buffers carry one extra trash row


def _mk_tier(nk, a, seed, zp, thr_g):
    from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, device="cuda", generator=g)  # noqa: E731
    t = RecallTier(r=R, topj=-1)
    t.r, t.scale, t.zp, t.thr_g, t.built = R, 192**-0.5, zp, thr_g, True
    t.kept_rows = rnd(nk, 576).to(torch.bfloat16)
    t.V = rnd(R, 512)
    t.side = rnd(a, 64).to(torch.bfloat16)
    t.csk = rnd(a, R).to(torch.float16)
    t.rho = torch.rand(a, device="cuda", generator=g)
    t.arch = torch.randperm(500000, device="cuda")[:a] + 1
    t._scatter_buf = None
    t._qside_t = t._qsk_t = t._hit_buf = t._inf = None
    return t


def _mk_state(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qbuf = torch.randn(L, MAX_REQS + 1, H, 576, device="cuda", generator=g)
    qbuf[:, TRASH] = 0
    fetch = torch.zeros(L, MAX_REQS + 1, W, dtype=torch.int64, device="cuda")
    flen = torch.zeros(L, MAX_REQS + 1, dtype=torch.int64, device="cuda")
    tiers = {}
    shapes = [(900, 20000), (1100, 26000), (700, 9000)]
    for li in range(L):
        for slot in (2, 5):
            nk, a = shapes[li]
            tiers[(li, slot)] = _mk_tier(
                nk + slot, a + 100 * slot, seed=li * 10 + slot,
                zp=float(li), thr_g=-float("inf"),
            )
    return qbuf, fetch, flen, tiers


def _capacity_pack(qbuf, fetch, flen, P=None):
    from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack

    return BatchedScanPack.at_capacity(
        P or (L * MAX_REQS), 2048, 32768, R, H, qbuf, fetch, flen, TRASH
    )


class TestCapacityPack(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_partial_update_matches_legacy_pack(self):
        from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack

        qbuf, fetch, flen, tiers = _mk_state()
        pairs = sorted(tiers.keys())
        tl = [tiers[p] for p in pairs]

        ref_fetch, ref_flen = fetch.clone(), flen.clone()
        legacy = BatchedScanPack(pairs, tl, qbuf, ref_fetch, ref_flen, H)
        legacy.run()

        cap = _capacity_pack(qbuf, fetch, flen)
        self.assertTrue(cap.fits(pairs, tl))
        cap.update(pairs, tl)
        cap.run()
        torch.cuda.synchronize()

        for li, slot in pairs:
            n, nr = int(flen[li, slot]), int(ref_flen[li, slot])
            self.assertEqual(n, nr, f"pair {(li, slot)} count")
            self.assertTrue(
                torch.equal(
                    fetch[li, slot, :n].sort().values,
                    ref_fetch[li, slot, :nr].sort().values,
                ),
                f"pair {(li, slot)} fetch set",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_placeholders_write_only_trash(self):
        qbuf, fetch, flen, tiers = _mk_state()
        sentinel = 12345
        flen.fill_(sentinel)
        cap = _capacity_pack(qbuf, fetch, flen)
        cap.run()  # all placeholders
        torch.cuda.synchronize()
        # only the trash row may have been touched
        self.assertTrue((flen[:, :MAX_REQS] == sentinel).all().item())

        # a later empty update() silences a previously occupied pack
        pairs = sorted(tiers.keys())
        cap.update(pairs, [tiers[p] for p in pairs])
        cap.run()
        cap.update([], [])
        flen.fill_(sentinel)
        cap.run()
        torch.cuda.synchronize()
        self.assertTrue((flen[:, :MAX_REQS] == sentinel).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_captured_empty_then_updated_replays_correctly(self):
        from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack

        qbuf, fetch, flen, tiers = _mk_state()
        cap = _capacity_pack(qbuf, fetch, flen)

        # warm up the JIT on the empty pack, then capture it empty -- the
        # exact lifecycle of the model-graph bake (capture precedes tiers)
        for _ in range(2):
            cap.run()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cap.run()

        # reference from a fresh legacy pack on identical inputs
        pairs = sorted(tiers.keys())
        tl = [tiers[p] for p in pairs]
        ref_fetch, ref_flen = fetch.clone(), flen.clone()
        legacy = BatchedScanPack(pairs, tl, qbuf, ref_fetch, ref_flen, H)
        legacy.run()

        cap.update(pairs, tl)
        g.replay()
        torch.cuda.synchronize()
        for li, slot in pairs:
            n, nr = int(flen[li, slot]), int(ref_flen[li, slot])
            self.assertEqual(n, nr, f"pair {(li, slot)} count after replay")
            self.assertTrue(
                torch.equal(
                    fetch[li, slot, :n].sort().values,
                    ref_fetch[li, slot, :nr].sort().values,
                ),
                f"pair {(li, slot)} fetch set after replay",
            )

        # content refresh without recapture: new tiers, same graph
        tiers2 = {
            k: _mk_tier(800 + 7 * k[1], 15000 + 500 * k[1], seed=99 + k[0],
                        zp=1.0, thr_g=-float("inf"))
            for k in pairs
        }
        tl2 = [tiers2[p] for p in pairs]
        ref_fetch.zero_(), ref_flen.zero_()
        legacy2 = BatchedScanPack(pairs, tl2, qbuf, ref_fetch, ref_flen, H)
        legacy2.run()
        cap.update(pairs, tl2)
        g.replay()
        torch.cuda.synchronize()
        for li, slot in pairs:
            n, nr = int(flen[li, slot]), int(ref_flen[li, slot])
            self.assertEqual(n, nr, f"pair {(li, slot)} count after refresh")
            self.assertTrue(
                torch.equal(
                    fetch[li, slot, :n].sort().values,
                    ref_fetch[li, slot, :nr].sort().values,
                ),
                f"pair {(li, slot)} fetch set after refresh",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
