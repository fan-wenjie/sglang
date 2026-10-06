"""sigma's low-pass cutoff as a served parameter.

Before SGLANG_VESTIGEKV_LOWPASS_KAPPA there was no way to set kappa at all: the
cutoff was read from defaults at four sites and the two blockwise_sigma entry
points took no kappa, so the fused kernel got the compiled-in value by default
argument. Threading it created exactly one silent-failure shape -- an entry point
that accepts a kappa and drops it before the projection, which no byte counter,
no fetch percentile and no quality draw can see, because a wrong cutoff produces
a perfectly well-formed sigma over a different frequency band.

1. Both entry points REACH the projection with the kappa they were given, and
   distinct kappas give distinct sigma. The rFFT form is the definition -- zero
   every bin from kappa up, subtract, take the row norm -- so the reference is
   written out here rather than taken from the module under test.
2. The cap the backend validates against is the kernel's real limit. defaults
   owns LOWPASS_KAPPA_MAX and sigma_fused derives its pad from it; if those two
   ever drift, the backend accepts a kappa whose basis cannot be built, and the
   failure lands at first-call compile time, i.e. at serve time.
"""

import unittest

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.eviction import (
    blockwise_sigma,
    blockwise_sigma_from_pool,
)

KAPPAS = (1, 2, 4, 8, 16)


def _reference(side: torch.Tensor, block: int, kappa: int) -> torch.Tensor:
    """sigma by definition: the residual of a kappa-bin low-pass, per block."""
    out = []
    for i in range(side.shape[0] // block):
        b = side[i * block : (i + 1) * block].double()
        f = torch.fft.rfft(b, dim=0)
        f[kappa:] = 0
        out.append((b - torch.fft.irfft(f, n=block, dim=0)).norm(dim=-1))
    return torch.cat(out).float()


class TestLowpassKappaReachesTheProjection(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.block = 64
        self.side = torch.randn(4 * self.block, 32, generator=g)

    def test_blockwise_sigma_honours_the_kappa_it_is_given(self):
        for k in KAPPAS:
            got = blockwise_sigma(self.side, self.block, kappa=k)
            torch.testing.assert_close(
                got, _reference(self.side, self.block, k), rtol=2e-5, atol=2e-5,
                msg=f"kappa={k} did not reach the projection",
            )

    def test_distinct_kappas_give_distinct_sigma(self):
        # A dropped argument looks exactly like this test failing: every kappa
        # returns the default's answer.
        sigs = {k: blockwise_sigma(self.side, self.block, kappa=k) for k in KAPPAS}
        for a, b in zip(KAPPAS, KAPPAS[1:]):
            self.assertFalse(
                torch.allclose(sigs[a], sigs[b], rtol=1e-3),
                f"kappa {a} and {b} returned the same sigma",
            )

    def test_from_pool_honours_it_too(self):
        # The pool entry point addresses a `dim`-wide branch inside a wider row;
        # off CUDA it falls back to the gather, which is the path under test.
        pool = torch.zeros(self.side.shape[0] + 7, 16 + self.side.shape[1])
        slots = torch.arange(self.side.shape[0]) + 7
        pool[slots, 16 : 16 + self.side.shape[1]] = self.side
        for k in KAPPAS:
            got = blockwise_sigma_from_pool(
                pool, slots, self.block, kappa=k, offset=16, dim=self.side.shape[1]
            )
            torch.testing.assert_close(
                got, blockwise_sigma(self.side, self.block, kappa=k),
                rtol=2e-5, atol=2e-5, msg=f"from_pool dropped kappa={k}",
            )


class TestCapIsTheKernelsRealLimit(unittest.TestCase):
    def test_pad_is_derived_from_the_cap(self):
        from sglang.srt.layers.attention.vestigekv import sigma_fused as SF

        self.assertEqual(SF._KB_PAD, 2 * D.LOWPASS_KAPPA_MAX)
        self.assertLessEqual(D.LOWPASS_KAPPA, D.LOWPASS_KAPPA_MAX)

    def test_basis_builds_at_the_cap_and_not_above(self):
        from sglang.srt.layers.attention.vestigekv import sigma_fused as SF

        C = SF._basis(64, D.LOWPASS_KAPPA_MAX, "cpu")
        self.assertEqual(C.shape, (64, 2 * D.LOWPASS_KAPPA_MAX))
        self.assertEqual(
            int((C.abs().sum(0) > 0).sum()), 2 * D.LOWPASS_KAPPA_MAX - 1
        )
        with self.assertRaises(RuntimeError):
            SF._basis(64, D.LOWPASS_KAPPA_MAX + 1, "cpu")


if __name__ == "__main__":
    unittest.main()
