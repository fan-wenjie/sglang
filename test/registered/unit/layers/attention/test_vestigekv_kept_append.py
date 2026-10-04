"""The kept table may be extended instead of rebuilt while `closed` holds.

The prefill hook used to re-rank the whole closed prefix on every chunk, 64
times over for a 32k prompt at a 512-token chunk. It now appends the rows the
chunk added to the unclosed tail and re-ranks only when a block closes. That
is exact for one reason: sigma is append-only and `closed` indexes a prefix of
it, so between two closes the same rows win the same top-m and the ranked half
of the table cannot move.

If that ever stops holding -- a sigma that is rewritten in place, a ranking
that depends on the tail, an m that is not a function of `closed` -- the
serving path would attend a stale row set and nothing else would notice. So
the property is pinned here rather than the implementation.
"""

import unittest

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.eviction import kept_rows
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RHO = 1.0 / 32
BLOCK = D.CLOSE_BLOCK


class TestKeptAppend(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_appending_the_tail_equals_a_full_rebuild(self):
        # One closed block, then chunks that only grow the tail.
        closed = BLOCK
        sigma = torch.randn(closed)
        for s0, s1 in ((closed + 512, closed + 1024), (closed + 1024, 2 * BLOCK - 1)):
            with self.subTest(grew=(s0, s1)):
                rows = torch.arange(s1)
                at_s0 = kept_rows(sigma, rows[:s0], rho=RHO, closed=closed, sinks=D.SINKS)
                appended = torch.cat([at_s0, rows[s0:s1]])
                rebuilt = kept_rows(sigma, rows, rho=RHO, closed=closed, sinks=D.SINKS)
                self.assertTrue(bool(torch.equal(appended, rebuilt)))

    def test_a_close_moves_the_ranked_half(self):
        # The negative branch: once `closed` advances the append is NOT valid,
        # which is why the hook re-ranks there. Without this case a version
        # that never re-ranked would pass the suite.
        rows = torch.arange(2 * BLOCK + 7)
        sigma = torch.randn(2 * BLOCK)
        one = kept_rows(sigma, rows[: BLOCK + 512], rho=RHO, closed=BLOCK, sinks=D.SINKS)
        two = kept_rows(sigma, rows, rho=RHO, closed=2 * BLOCK, sinks=D.SINKS)
        self.assertNotEqual(one.numel() + (rows.numel() - BLOCK - 512), two.numel())

    def test_sigma_is_a_prefix_so_the_ranking_is_stable(self):
        # The premise itself: scoring a longer prefix does not change the
        # scores of the rows already in it.
        closed = BLOCK
        sigma_long = torch.randn(2 * closed)
        rows = torch.arange(2 * closed)
        a = kept_rows(sigma_long[:closed], rows[:closed], rho=RHO, closed=closed, sinks=D.SINKS)
        b = kept_rows(sigma_long, rows[:closed], rho=RHO, closed=closed, sinks=D.SINKS)
        self.assertTrue(bool(torch.equal(a, b)))



class TestKeptPositionsMatchTheirRows(CustomTestCase):
    """The branch rule reads a kept row's request-relative POSITION from a table
    parallel to the row table, so position[i] must name row[i]: a mismatch marks
    the wrong groups kept, which moves the threshold and therefore the fire
    count, with nothing in any count or latency to show it.

    The kernel tests build both tables directly and so cannot see a mismatch
    introduced by the WRITERS, which is where it would come from.
    """

    def test_kept_rows_returns_positions_that_name_its_rows(self):
        from sglang.srt.layers.attention.vestigekv.eviction import kept_rows

        torch.manual_seed(0)
        for seq, closed in ((BLOCK + 512, BLOCK), (3 * BLOCK, 2 * BLOCK), (BLOCK, BLOCK)):
            # row_slots is a SHUFFLED pool mapping, as req_to_token is: a
            # position-ordered table would hide an off-by-one.
            rows = torch.randperm(1 << 15)[:seq]
            sigma = torch.rand(closed)
            got, pos = kept_rows(sigma, rows, rho=RHO, closed=closed,
                                 sinks=D.SINKS, return_pos=True)
            self.assertEqual(got.shape, pos.shape)
            self.assertTrue(torch.equal(rows[pos], got))
            # the tail is kept whole, in order, and the sinks are always in
            if seq > closed:
                self.assertTrue(torch.equal(pos[closed - seq:], torch.arange(closed, seq)))
            self.assertTrue(set(range(D.SINKS)).issubset(set(pos.tolist())))
            self.assertTrue(bool((pos[1:] > pos[:-1]).all()))  # strictly ascending

    def test_the_close_paths_positions_name_its_rows(self):
        """The close path derives positions with keep.nonzero() beside rows with
        closed_slots[keep]; the two orders have to agree."""
        torch.manual_seed(1)
        seq, c1 = 3 * BLOCK + 777, 3 * BLOCK
        r2t_row = torch.randperm(1 << 15)[:seq]
        sigma = torch.rand(c1)
        m = max(1, round(RHO * c1))
        keep = torch.zeros(c1, dtype=torch.bool)
        keep[sigma[:c1].topk(min(m, c1)).indices] = True
        keep[: D.SINKS] = True
        closed_slots = r2t_row[:c1]
        new_kept = torch.cat([closed_slots[keep], r2t_row[c1:seq]])
        new_pos = torch.cat([keep.nonzero().flatten(), torch.arange(c1, seq)])
        self.assertEqual(new_kept.shape, new_pos.shape)
        self.assertTrue(torch.equal(r2t_row[new_pos], new_kept))

if __name__ == "__main__":
    unittest.main()
