"""m_fixed: a constant kept budget instead of a proportional one.

Derived properties, both of which the proportional path already relies on and
a fixed budget must not break:

1. The COUNT is a host-side expression. kept_rows exists so the caller can size
   the table write and the CSR bound without a device sync, which it can only do
   if the count is `n_sink + min(m, closed - n_sink) + tail` with no data
   dependence. A fixed m keeps that; a fixed m larger than the closed prefix
   must clamp rather than overrun.
2. kept_rows' set is a SUPERSET of select_kept's. kept_rows takes the sinks out
   of the ranking instead of unioning them in, so the two differ only where a
   sink would have won a top-m place -- and then kept_rows keeps the displaced
   non-sink row as well. Whichever way m is chosen, that relation must hold, or
   a consumer reading one and sizing by the other under-reads.
"""

import unittest

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.eviction import kept_rows, select_kept


def _case(closed=4096, tail=300, seed=0, dev="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    sigma = torch.rand(closed, generator=g).to(dev)
    row_slots = torch.arange(closed + tail, device=dev) + 1000
    return sigma, row_slots


class TestFixedKeptBudget(unittest.TestCase):
    def test_count_is_the_host_side_expression(self):
        sigma, row_slots = _case()
        closed, tail = 4096, 300
        for M in (1, 32, 128, 1024):
            rows = kept_rows(sigma, row_slots, rho=1 / 128, closed=closed, m_fixed=M)
            n_sink = min(D.SINKS, closed)
            want = n_sink + min(M, closed - n_sink) + tail
            self.assertEqual(rows.numel(), want, f"m_fixed={M}")

    def test_a_budget_past_the_prefix_clamps(self):
        # 8 closed rows, a budget of 4096: must clamp, not overrun or raise.
        sigma, row_slots = _case(closed=8, tail=3)
        rows = kept_rows(sigma, row_slots, rho=1 / 128, closed=8, m_fixed=4096)
        self.assertEqual(rows.numel(), 8 + 3)

    def test_kept_rows_is_a_superset_of_select_kept(self):
        sigma, row_slots = _case()
        closed = 4096
        for M in (1, 32, 128):
            _, pos = kept_rows(
                sigma, row_slots, rho=1 / 128, closed=closed, m_fixed=M, return_pos=True
            )
            mask = select_kept(sigma, rho=1 / 128, closed=closed, m_fixed=M)
            got = set(pos[pos < closed].tolist())
            want = set(mask.nonzero().flatten().tolist())
            self.assertTrue(want <= got, f"m_fixed={M}: {len(want - got)} positions lost")
            # and it is tight: at most `sinks` extra, which is the displacement
            self.assertLessEqual(len(got - want), D.SINKS, f"m_fixed={M}")

    def test_fixed_does_not_track_context(self):
        # the whole point: the count is flat in `closed` where rho's grows
        n = {}
        for closed in (4096, 32768, 131072):
            sigma, row_slots = _case(closed=closed, tail=0)
            f = kept_rows(sigma, row_slots, rho=1 / 128, closed=closed, m_fixed=128)
            p = kept_rows(sigma, row_slots, rho=1 / 128, closed=closed)
            n[closed] = (f.numel(), p.numel())
        self.assertEqual({v[0] for v in n.values()}, {128 + min(D.SINKS, 4096)})
        self.assertEqual(len({v[1] for v in n.values()}), 3)  # proportional grows


if __name__ == "__main__":
    unittest.main(verbosity=2)
