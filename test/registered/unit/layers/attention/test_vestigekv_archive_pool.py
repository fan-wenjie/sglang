"""Pooling the archive must agree with DSA's grouping and with the projection.

Derived property, not a mirror: the whole reason the archive can be pooled
after it is built is that csk is linear in the latent content, so pooling the
projections has to equal projecting the pooled content -- a rewrite that
replaced the mean with anything else would silently break that identity. And a
fired group has to expand to exactly the rows DSA's grouping says it covers,
or the recalled rows are not the rows that were scored.
"""

import unittest

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.archive_pool import (
    archive_bytes_per_token,
    expand_groups,
    pool_operands,
    pooled_capacity,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

T, DIM, R, POOL = 64, 32, 8, 4


class TestPoolOperands(CustomTestCase):
    def test_pooling_the_projection_equals_projecting_the_pooled_content(self):
        content = torch.randn(T, DIM, dtype=torch.float64)
        V = torch.randn(R, DIM, dtype=torch.float64)
        csk = content @ V.T
        rho = content.norm(dim=1)
        pooled, _, _ = pool_operands(csk, rho, POOL)
        direct = content.view(-1, POOL, DIM).mean(1) @ V.T
        self.assertTrue(torch.allclose(pooled, direct, atol=1e-9))

    def test_the_norm_pools_by_max_so_it_still_bounds_the_group(self):
        csk = torch.zeros(T, R)
        rho = torch.arange(T, dtype=torch.float32)
        _, pr, _ = pool_operands(csk, rho, POOL)
        # group g holds rows 4g..4g+3, so its bound is the last of them
        self.assertTrue(torch.equal(pr, torch.arange(3, T, POOL, dtype=torch.float32)))

    def test_a_trailing_partial_group_is_dropped_not_short_averaged(self):
        n = T + POOL - 1
        csk = torch.ones(n, R)
        rho = torch.ones(n)
        pooled, pr, _ = pool_operands(csk, rho, POOL)
        self.assertEqual(pooled.shape[0], n // POOL)
        self.assertEqual(pr.shape[0], n // POOL)

    def test_pool_size_one_is_the_identity(self):
        csk = torch.randn(T, R)
        rho = torch.randn(T)
        pooled, pr, _ = pool_operands(csk, rho, 1)
        self.assertIs(pooled, csk)
        self.assertIs(pr, rho)

    def test_fewer_rows_than_one_group_yields_no_entries(self):
        pooled, pr, _ = pool_operands(torch.ones(POOL - 1, R), torch.ones(POOL - 1), POOL)
        self.assertEqual(pooled.shape[0], 0)
        self.assertEqual(pr.shape[0], 0)


class TestExpandGroups(CustomTestCase):
    def test_a_group_expands_to_the_rows_dsa_grouping_gives_it(self):
        got = expand_groups(torch.tensor([0, 3]), POOL)
        self.assertEqual(got.tolist(), [0, 1, 2, 3, 12, 13, 14, 15])

    def test_expansion_round_trips_with_pooling(self):
        csk = torch.arange(T * R, dtype=torch.float32).view(T, R)
        pooled, _, _ = pool_operands(csk, torch.zeros(T), POOL)
        g = torch.tensor([2])
        rows = expand_groups(g, POOL)
        self.assertTrue(torch.allclose(csk[rows].mean(0), pooled[2]))


class TestBudgetAndTraffic(CustomTestCase):
    def test_the_budget_scales_with_the_ratio_because_the_measurement_says_so(self):
        # pooled at the unpooled budget recalls LESS; the change is only a win
        # with the budget raised, so the two must not drift apart
        self.assertEqual(pooled_capacity(2048, 4), 4096)
        self.assertEqual(pooled_capacity(2048, 1), 2048)

    def test_pooling_puts_the_archive_at_dsas_own_bytes_per_token(self):
        # DSA's pooled index cache is 132 B per 4 tokens = 33 B/token/layer
        self.assertAlmostEqual(archive_bytes_per_token(D.INDEX_RANK, 4), 34.0)
        self.assertAlmostEqual(archive_bytes_per_token(D.INDEX_RANK, 1), 132.0)


class TestSpreadBound(CustomTestCase):
    """The spread is what keeps the certificate sound once the archive pools."""

    def test_the_spread_bounds_every_rows_gap_from_the_pooled_score(self):
        torch.manual_seed(0)
        csk = torch.randn(T, R, dtype=torch.float64)
        rho = torch.zeros(T, dtype=torch.float64)
        mean, _, spread = pool_operands(csk, rho, POOL)
        q = torch.randn(R, dtype=torch.float64)
        per_row = (csk @ q).view(-1, POOL)
        pooled = mean @ q
        gap = (per_row - pooled[:, None]).abs().amax(1)
        # Cauchy-Schwarz: the inflation the scan applies must cover every row
        self.assertTrue(bool((gap <= spread * q.norm() + 1e-9).all()))

    def test_identical_rows_have_no_spread_so_pooling_costs_no_inflation(self):
        row = torch.randn(1, R, dtype=torch.float64)
        csk = row.repeat(T, 1)
        _, _, spread = pool_operands(csk, torch.zeros(T, dtype=torch.float64), POOL)
        self.assertLess(float(spread.max()), 1e-9)

    def test_an_unpooled_archive_reports_no_spread(self):
        _, _, spread = pool_operands(torch.randn(T, R), torch.ones(T), 1)
        self.assertEqual(spread.shape[0], T)
        self.assertEqual(float(spread.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
