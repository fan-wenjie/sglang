"""kept_rows drops a device sync without dropping any row select_kept keeps.

The prefill hook runs per MLA layer per chunk, and the mask form ends in a
nonzero whose output shape the caller then reads, which is a sync each time --
704 of them for a 32k prompt at a 512-token chunk. kept_rows gets the count
back onto the host by taking the sinks out of the ranking instead of unioning
them into it, and what has to hold for that to be a free trade is a property,
not an implementation detail: the row set only ever grows, so no row the
reference policy keeps can go missing, and the count is a host-side
expression rather than a readback.

Order is part of the contract too. The mask form yields ascending rows
(nonzero does), and the kept table, the CSR pack and the row-invariant check
all read it that way.
"""

import unittest

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.eviction import kept_rows, select_kept
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RHO = 1.0 / 32
SHAPES = ((4096, 37), (8192, 0), (256, 255), (16, 3))


def _reference(sigma, rows, closed, sinks):
    mask = select_kept(sigma, rho=RHO, closed=closed, sinks=sinks)
    return torch.cat([rows[:closed][mask.nonzero(as_tuple=True)[0]], rows[closed:]])


class TestKeptRows(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_keeps_every_row_the_reference_keeps(self):
        for closed, tail in SHAPES:
            with self.subTest(closed=closed):
                sigma = torch.randn(closed)
                rows = torch.arange(closed + tail)
                got = kept_rows(sigma, rows, rho=RHO, closed=closed, sinks=D.SINKS)
                ref = _reference(sigma, rows, closed, D.SINKS)
                self.assertTrue(bool(torch.isin(ref, got).all()))

    def test_count_is_a_host_side_expression(self):
        # What the sync was paying for: sinks + m + tail, known before the
        # kernel runs. The reference's count is at most this and varies with
        # the data, which is exactly why it had to be read back.
        for closed, tail in SHAPES:
            with self.subTest(closed=closed):
                sigma = torch.randn(closed)
                rows = torch.arange(closed + tail)
                n_sink = min(D.SINKS, closed)
                m = min(max(1, round(RHO * closed)), closed - n_sink)
                got = kept_rows(sigma, rows, rho=RHO, closed=closed, sinks=D.SINKS)
                self.assertEqual(got.numel(), n_sink + m + tail)
                self.assertLessEqual(
                    _reference(sigma, rows, closed, D.SINKS).numel(), got.numel()
                )

    def test_rows_stay_ascending(self):
        for closed, tail in SHAPES:
            with self.subTest(closed=closed):
                got = kept_rows(
                    torch.randn(closed),
                    torch.arange(closed + tail),
                    rho=RHO,
                    closed=closed,
                    sinks=D.SINKS,
                )
                self.assertTrue(bool((got[:-1] <= got[1:]).all()))

    def test_sinks_are_kept_when_their_sigma_is_the_lowest(self):
        # The negative branch: sinks win no ranking place now, so the only
        # thing keeping them is the unconditional prefix.
        closed = 4096
        sigma = torch.rand(closed) + 1.0
        sigma[: D.SINKS] = 0.0
        got = kept_rows(
            sigma, torch.arange(closed), rho=RHO, closed=closed, sinks=D.SINKS
        )
        self.assertTrue(bool((got[: D.SINKS] == torch.arange(D.SINKS)).all()))


if __name__ == "__main__":
    unittest.main()
