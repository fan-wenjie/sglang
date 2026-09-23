"""The selection-recall telemetry measures headroom, not agreement.

Derived property, not a mirror: the number decides whether a supplement that
only ADDS rows to DSA's selection has anywhere to go, so what is pinned is
that a perfect selection reads 1.0, a disjoint one reads 0.0, and that the
mass figure weights the oracle's own ranking -- missing the top row is not the
same as missing the last one.
"""

import unittest

import torch

from sglang.srt.layers.attention.dsa.select_recall_telemetry import (
    SelectRecallTelemetry,
    step_stats,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

S, D, K = 64, 8, 8


def _rows_and_q():
    """Row i scores exactly i, so the oracle top-K is rows S-K .. S-1."""
    q = torch.zeros(1, D)
    q[0, 0] = 1.0
    rows = torch.zeros(S, D)
    rows[:, 0] = torch.arange(S, dtype=torch.float32)
    return rows, q


class TestStepStats(CustomTestCase):
    def test_a_selection_holding_the_oracle_reads_one(self):
        rows, q = _rows_and_q()
        sel = torch.arange(S - K, S)
        c, m, n = step_stats(rows, q, sel, K)
        self.assertAlmostEqual(float(c), 1.0, places=5)
        self.assertAlmostEqual(float(m), 1.0, places=5)
        self.assertEqual(int(n), K)

    def test_a_disjoint_selection_reads_zero(self):
        rows, q = _rows_and_q()
        c, m, _ = step_stats(rows, q, torch.arange(0, K), K)
        self.assertAlmostEqual(float(c), 0.0, places=5)
        self.assertAlmostEqual(float(m), 0.0, places=5)

    def test_mass_weights_the_oracles_own_ranking(self):
        rows, q = _rows_and_q()
        # miss only the BEST oracle row vs miss only the WORST: same count,
        # different mass, which is the whole reason mass is reported
        miss_best = torch.arange(S - K, S - 1)
        miss_worst = torch.arange(S - K + 1, S)
        c1, m1, _ = step_stats(rows, q, miss_best, K)
        c2, m2, _ = step_stats(rows, q, miss_worst, K)
        self.assertAlmostEqual(float(c1), float(c2), places=5)
        self.assertLess(float(m1), float(m2))

    def test_a_row_scores_by_its_best_head(self):
        rows, q = _rows_and_q()
        q2 = torch.zeros(2, D)
        q2[0, 1] = 1.0  # a head that wants nothing
        q2[1, 0] = 1.0  # the head that ranks rows
        c, _, _ = step_stats(rows, q2, torch.arange(S - K, S), K)
        self.assertAlmostEqual(float(c), 1.0, places=5)


class TestTelemetry(CustomTestCase):
    def test_a_sequence_no_longer_than_the_budget_is_not_recorded(self):
        tel = SelectRecallTelemetry(topk=K, every=1)
        tel.observe(
            rows=torch.zeros(K, D), q=torch.zeros(1, D), selected=torch.arange(2), lid=3
        )
        self.assertEqual(tel.rec, {})

    def test_reports_per_layer_once_per_every(self):
        rows, q = _rows_and_q()
        tel = SelectRecallTelemetry(topk=K, every=2)
        with self.assertLogs(
            "sglang.srt.layers.attention.dsa.select_recall_telemetry", "INFO"
        ) as cm:
            for lid in (3, 7):
                for _ in range(2):
                    tel.observe(rows=rows, q=q, selected=torch.arange(S - K, S), lid=lid)
        self.assertEqual(len(cm.output), 2)
        self.assertIn("layer=3", cm.output[0])
        self.assertIn("count_mean=1.0000", cm.output[0])


if __name__ == "__main__":
    unittest.main()
