"""The spectrum telemetry names the bin a standing component sits in.

Derived property, not a mirror: the statistic's whole purpose is that a peak
far above tier 1's cutoff means sigma is ranking a component shared by every
row rather than per-row anomaly, and that a broadband residual does not. The
offline study (docs/sidecar-notch-study.md) separates real corpora by the
peak's LOCATION and not its height, so both are pinned here.
"""

import math
import unittest

import torch

from sglang.srt.layers.attention.vestigekv.spectrum import (
    SpectrumTelemetry,
    block_spectrum,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

T, D, KAPPA = 4096, 64, 16


def _noise(seed=0):
    return torch.randn(T, D, generator=torch.Generator().manual_seed(seed))


class TestBlockSpectrum(CustomTestCase):
    def test_a_standing_component_is_reported_at_its_own_bin(self):
        t = torch.arange(T, dtype=torch.float32)
        for bin_ in (71, 200):
            side = torch.sin(2 * math.pi * bin_ * t / T)[:, None] * torch.ones(1, D)
            got, peakiness, share = block_spectrum(side + 0.1 * _noise(), KAPPA)
            self.assertEqual(got, bin_)
            self.assertGreater(peakiness, 1e3)
            self.assertGreater(share, 0.5)

    def test_a_broadband_residual_is_flat(self):
        _, peakiness, _ = block_spectrum(_noise(1), KAPPA)
        self.assertLess(peakiness, 10.0)

    def test_a_smooth_signal_peaks_at_the_first_bin_past_the_cutoff(self):
        # real prose's signature: no rotation, no cadence, just the shoulder a
        # low-pass leaves behind
        ramp = torch.linspace(0, 1, T)[:, None] * torch.ones(1, D)
        got, _, _ = block_spectrum(ramp + 0.1 * _noise(2), KAPPA)
        self.assertEqual(got, KAPPA)

    def test_the_mean_is_removed_so_a_constant_offset_changes_nothing(self):
        # bin 0 would otherwise swamp everything on any signal with a DC term
        side = _noise(3)
        a, b = block_spectrum(side, KAPPA), block_spectrum(side + 7.0, KAPPA)
        self.assertEqual(a[0], b[0])
        self.assertAlmostEqual(a[1], b[1], places=3)  # fp32 subtraction order
        self.assertAlmostEqual(a[2], b[2], places=6)

    def test_the_accumulator_counts_every_whole_block_and_ignores_the_tail(self):
        t = torch.arange(T, dtype=torch.float32)
        one = torch.sin(2 * math.pi * 71 * t / T)[:, None] * torch.ones(1, D)
        tel = SpectrumTelemetry(KAPPA, every=10**9)
        tel.observe(torch.cat([one, one, one[: T // 2]]), T, lid=3)
        tel.observe(one, T, lid=7)
        # the per-block statistics stay on the device until a report is due,
        # so the counts land at the dump and not at observe()
        self.assertEqual(len(tel.rec[3]["pending"]), 2)
        tel.dump(lid=3)
        tel.dump(lid=7)
        self.assertEqual(tel.rec[3]["n"], 2)
        self.assertEqual(tel.rec[3]["bins"], {71: 2})
        # per layer, so one layer's blocks never land in another's record
        self.assertEqual(tel.rec[7]["n"], 1)


if __name__ == "__main__":
    unittest.main()
