"""The top-k threshold telemetry reports the cut a pivot would have to hit.

Derived property, not a mirror: the statistic exists to answer whether the
k-th largest logit is predictable, so what is pinned here is that the value it
reports IS the k-th largest of the LIVE candidates -- padding past the group
length must not reach the selection, and must not be trusted to be -inf
either, since the caller hands over a raw logits buffer.
"""

import unittest

import torch

from sglang.srt.layers.attention.dsa.topk_threshold_telemetry import (
    TopkThresholdTelemetry,
    row_stats,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

K = 8


class TestRowStats(CustomTestCase):
    def test_reports_the_kth_largest_of_the_live_prefix(self):
        row = torch.arange(64, dtype=torch.float32)
        valid = torch.tensor(20)
        kth, mx, halfk, n = row_stats(row, valid, K)
        # live prefix is 0..19, so the largest is 19 and the 8th largest is 12
        self.assertEqual(float(mx), 19.0)
        self.assertEqual(float(kth), 12.0)
        self.assertEqual(float(halfk), 15.0)
        self.assertEqual(float(n), 20.0)

    def test_padding_is_masked_not_assumed_to_be_neg_inf(self):
        # padding that is LARGER than every live value: an unmasked selection
        # would report the padding and the pivot study would be meaningless
        row = torch.cat([torch.arange(20, dtype=torch.float32), torch.full((44,), 999.0)])
        kth, mx, _, _ = row_stats(row, torch.tensor(20), K)
        self.assertEqual(float(mx), 19.0)
        self.assertEqual(float(kth), 12.0)

    def test_fewer_live_candidates_than_k_reports_neg_inf(self):
        row = torch.arange(64, dtype=torch.float32)
        kth, _, _, n = row_stats(row, torch.tensor(K - 1), K)
        self.assertEqual(float(n), float(K - 1))
        self.assertTrue(torch.isinf(kth) and kth < 0)


class TestTelemetry(CustomTestCase):
    def _logits(self, hi):
        return torch.cat(
            [torch.arange(hi, dtype=torch.float32), torch.zeros(64 - hi)]
        ).unsqueeze(0)

    def test_reports_once_per_every_and_keeps_layers_apart(self):
        tel = TopkThresholdTelemetry(every=2)
        with self.assertLogs(
            "sglang.srt.layers.attention.dsa.topk_threshold_telemetry", "INFO"
        ) as cm:
            for lid in (3, 7):
                for hi in (20, 24):
                    tel.observe(self._logits(hi), torch.tensor(hi), K, lid=lid)
        self.assertEqual(len(cm.output), 2)
        self.assertIn("layer=3", cm.output[0])
        self.assertIn("layer=7", cm.output[1])
        for line in cm.output:
            self.assertIn("n=2", line)
            self.assertIn("base=0", line)
        # 8th largest of 0..19 is 12, of 0..23 is 16
        self.assertIn("kth=[12,16]", cm.output[0])

    def test_base_advances_so_samples_stay_alignable_across_layers(self):
        tel = TopkThresholdTelemetry(every=1)
        with self.assertLogs(
            "sglang.srt.layers.attention.dsa.topk_threshold_telemetry", "INFO"
        ) as cm:
            for _ in range(3):
                tel.observe(self._logits(20), torch.tensor(20), K, lid=11)
        self.assertEqual([("base=%d" % i) in l for i, l in enumerate(cm.output)], [True] * 3)

    def test_a_row_narrower_than_k_is_not_recorded(self):
        tel = TopkThresholdTelemetry(every=1)
        tel.observe(torch.zeros(1, K - 1), torch.tensor(K - 1), K, lid=3)
        self.assertEqual(tel.rec, {})


if __name__ == "__main__":
    unittest.main()
