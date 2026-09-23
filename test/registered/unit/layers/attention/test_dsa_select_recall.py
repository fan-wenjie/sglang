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


class TestFarRegionStats(CustomTestCase):
    """The far-region comparison must be budget-matched and close-block-free."""

    def test_budget_is_taken_from_dsa_so_tier1_cannot_win_by_spending_more(self):
        from sglang.srt.layers.attention.dsa.select_recall_telemetry import (
            far_region_stats,
        )

        rows, q = _rows_and_q()
        rows[:, 0] = -rows[:, 0]  # oracle now lands in the far region
        n_far = 32
        # DSA picks 4 far rows; tier 1 must be scored at 4, not at its own size
        sel = torch.tensor([0, 1, 2, 3, 40, 50])
        sigma = torch.arange(S, dtype=torch.float32)
        r = far_region_stats(rows, q, sel, sigma, n_far, K)
        d, n_or, budget = r[0], r[12], r[13]
        self.assertEqual(int(budget), 4)
        self.assertEqual(int(n_or), K)
        self.assertAlmostEqual(float(d), 4 / K, places=5)

    def test_only_rows_older_than_the_close_block_are_compared(self):
        from sglang.srt.layers.attention.dsa.select_recall_telemetry import (
            far_region_stats,
        )

        rows, q = _rows_and_q()
        # the oracle top-K is rows 56..63, all inside the close region
        r = far_region_stats(
            rows, q, torch.arange(S - K, S), torch.arange(S, dtype=torch.float32), 32, K
        )
        d, n_or = r[0], r[12]
        self.assertEqual(int(n_or), 0)
        self.assertEqual(float(d), 0.0)

    def test_tier1_reads_high_sigma_as_kept(self):
        from sglang.srt.layers.attention.dsa.select_recall_telemetry import (
            far_region_stats,
        )

        rows, q = _rows_and_q()
        n_far = 32
        # make the far oracle rows 24..31 and give them the highest sigma
        rows2 = rows.clone()
        rows2[:, 0] = 0.0
        rows2[24:32, 0] = torch.arange(1, 9, dtype=torch.float32)
        sigma = torch.zeros(S)
        sigma[24:32] = torch.arange(1, 9, dtype=torch.float32)
        r = far_region_stats(rows2, q, torch.arange(0, 8), sigma, n_far, K)
        t1, budget = r[9], r[13]
        self.assertEqual(int(budget), 8)
        self.assertAlmostEqual(float(t1), 1.0, places=5)

    def test_the_union_separates_complementary_from_redundant_selectors(self):
        from sglang.srt.layers.attention.dsa.select_recall_telemetry import (
            far_region_stats,
        )

        rows, q = _rows_and_q()
        rows[:, 0] = -rows[:, 0]  # oracle is far rows 0..7
        sigma = torch.zeros(S)
        # redundant: tier 1 picks exactly what DSA already has
        sigma[0:4] = torch.arange(4, 0, -1, dtype=torch.float32)
        sel = torch.arange(0, 4)
        r = far_region_stats(rows, q, sel, sigma, 32, K)
        d, u = r[0], r[9]
        self.assertAlmostEqual(float(u), float(d), places=5)
        # complementary: tier 1 picks oracle rows DSA missed
        sigma2 = torch.zeros(S)
        sigma2[4:8] = torch.arange(4, 0, -1, dtype=torch.float32)
        r2 = far_region_stats(rows, q, sel, sigma2, 32, K)
        d2, u2 = r2[0], r2[9]
        # every tier-1 pick is an oracle row DSA missed, so the union gains all 4
        self.assertAlmostEqual(float(u2), float(d2) + 4 / K, places=5)


if __name__ == "__main__":
    unittest.main()
