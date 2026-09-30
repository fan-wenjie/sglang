"""The width controller's semantics, on CPU: no GPU needed to pin the logic.

Three of these are load-bearing rather than incidental.

LATCHING is the whole stability argument. The fire fraction rises when width
rises and falls when width falls, so a per-step loop on it diverges in
whichever direction it is pointed. This controller observes at the width the
build chose, decides once, and then holds -- so the observation is never a
function of its own output and there is no feedback path to be unstable.

ALLOCATION-FREEDOM is what lets it run where the pack is replayed from a
captured graph: torch.where and Tensor.add(scalar) both allocate, and both were
in the first draft.

RESET-ON-PREFILL is what stops one request's width decision deciding the next
occupant of the same slot.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.attention.vestigekv.width_control import (
    CertWidthController,
)


class TestCertWidthController(unittest.TestCase):
    def _ctl(self, P=4, tau=0.5, hi=2.0, lo=0.5, warmup=4):
        c = CertWidthController(P, torch.device("cpu"), tau=tau, gain_hi=hi,
                                gain_lo=lo, warmup=warmup)
        c.set_base(torch.full((P,), 0.5))
        return c

    def _run(self, c, fracs, steps, P=4, alen=100.0):
        cc = torch.zeros(P)
        for _ in range(steps):
            c.apply_(torch.tensor([int(f * alen) for f in fracs],
                                  dtype=torch.int32),
                     torch.tensor([alen] * P), cc)
        return cc

    def test_holds_build_width_until_warmup_completes(self):
        """Before the latch the served width must be exactly the build's, or the
        observation is already contaminated by the thing it is deciding."""
        c = self._ctl(warmup=4)
        cc = self._run(c, [1.0, 1.0, 0.0, 0.0], steps=3)
        self.assertEqual(cc.tolist(), [0.5] * 4, "width moved before latching")
        self.assertEqual(c.latched.tolist(), [False] * 4)

    def test_latches_high_and_low_by_the_observed_mean(self):
        c = self._ctl(warmup=4, hi=2.0, lo=0.5)
        cc = self._run(c, [1.0, 0.75, 0.25, 0.0], steps=4)
        self.assertEqual(c.latched.tolist(), [True] * 4)
        # tau 0.5: lanes 0,1 widen to 2x, lanes 2,3 narrow to 0.5x
        self.assertEqual([round(v, 4) for v in c.gain.tolist()],
                         [2.0, 2.0, 0.5, 0.5])
        self.assertEqual([round(v, 4) for v in cc.tolist()],
                         [1.0, 1.0, 0.25, 0.25])

    def test_latch_holds_when_the_signal_reverses(self):
        """The decision must survive the fire fraction collapsing afterwards --
        which it will, because widening changed it. That is the feedback path
        this design exists to cut."""
        c = self._ctl(warmup=4)
        self._run(c, [1.0, 1.0, 1.0, 1.0], steps=4)
        self.assertEqual(c.gain.tolist(), [2.0] * 4)
        self._run(c, [0.0, 0.0, 0.0, 0.0], steps=20)
        self.assertEqual(c.gain.tolist(), [2.0] * 4, "latch moved after deciding")

    def test_mean_not_last_value_decides(self):
        """One freak step must not decide a request: three steps at 0.0 and one
        at 1.0 average 0.25, below tau, so the lane narrows."""
        c = self._ctl(warmup=4, tau=0.5)
        cc = torch.zeros(4)
        for f in (1.0, 0.0, 0.0, 0.0):
            c.apply_(torch.tensor([int(f * 100)] * 4, dtype=torch.int32),
                     torch.tensor([100.0] * 4), cc)
        self.assertAlmostEqual(c.ff[0].item(), 0.25, places=5)
        self.assertEqual(c.gain.tolist(), [0.5] * 4)

    def test_reset_clears_the_latch(self):
        c = self._ctl(warmup=2)
        self._run(c, [1.0] * 4, steps=2)
        self.assertEqual(c.gain.tolist(), [2.0] * 4)
        c.reset(torch.tensor([1, 3]))
        self.assertEqual(c.gain.tolist(), [2.0, 1.0, 2.0, 1.0])
        self.assertEqual(c.latched.tolist(), [True, False, True, False])
        self.assertEqual(c.seen.tolist(), [2.0, 0.0, 2.0, 0.0])
        c.reset()
        self.assertEqual(c.gain.tolist(), [1.0] * 4)
        self.assertEqual(c.latched.tolist(), [False] * 4)

    def test_apply_allocates_nothing(self):
        c = self._ctl(warmup=3)
        cc = torch.zeros(4)
        names = ["base", "gain", "ff", "seen", "latched",
                 "_x", "_w", "_n", "_open", "_ready", "_now", "_hi"]
        before = {n: getattr(c, n).data_ptr() for n in names}
        before["cc"] = cc.data_ptr()
        for k in range(8):
            c.apply_(torch.tensor([k * 12] * 4, dtype=torch.int32),
                     torch.tensor([100.0] * 4), cc)
        for n in names:
            self.assertEqual(before[n], getattr(c, n).data_ptr(), f"{n} reallocated")
        self.assertEqual(before["cc"], cc.data_ptr(), "cc_out reallocated")

    def test_unit_gains_are_an_exact_no_op(self):
        """The off switch has to be exact: with both gains 1 the served width is
        the build's own, bit for bit, whatever the fire fraction does -- so the
        arm can be enabled in a run without moving any number."""
        c = self._ctl(warmup=2, hi=1.0, lo=1.0)
        cc = self._run(c, [1.0, 0.9, 0.1, 0.0], steps=10)
        self.assertEqual(cc.tolist(), [0.5] * 4)
        self.assertEqual(c.gain.tolist(), [1.0] * 4)

    def test_empty_archive_does_not_divide_by_zero(self):
        c = self._ctl(warmup=2)
        cc = torch.zeros(4)
        for _ in range(3):
            c.apply_(torch.zeros(4, dtype=torch.int32), torch.zeros(4), cc)
        self.assertTrue(torch.isfinite(c.ff).all(), c.ff)
        self.assertTrue(torch.isfinite(cc).all(), cc)


if __name__ == "__main__":
    unittest.main()


class TestArmedControllerIsBitIdenticalWhenNeutral(unittest.TestCase):
    """The armed-but-neutral seam must reproduce the unarmed cc exactly.

    An end-to-end score cannot establish this: four identical served-arm MRCR
    runs span 0.414-0.436, so a real regression and a no-op are
    indistinguishable there. The tensor comparison is decisive and costs no GPU
    hours, which is the right order to do these in -- this test was written
    after a validation run came back at 0.229 with no way to tell whether the
    seam or the workload had moved it.
    """

    def test_neutral_gains_reproduce_the_base_exactly(self):
        P = 7  # one slot per MLA layer, which is what the server reports
        torch.manual_seed(0)
        base = torch.rand(P).double().float()
        c = CertWidthController(P, torch.device("cpu"), tau=0.5,
                                gain_hi=1.0, gain_lo=1.0, warmup=4)
        c.set_base(base)
        cc = torch.zeros(P)
        for step in range(40):
            fired = torch.randint(0, 101, (P,), dtype=torch.int32)
            c.apply_(fired, torch.full((P,), 100.0), cc)
            self.assertTrue(
                torch.equal(cc, base),
                f"step {step}: cc drifted from base by "
                f"{(cc - base).abs().max().item():.3e}",
            )

    def test_gain_multiplies_base_exactly_once_latched(self):
        """And when it IS armed, cc is base * gain to the bit -- no accumulated
        drift from the running mean's in-place arithmetic."""
        P = 7
        base = torch.full((P,), 0.3)
        c = CertWidthController(P, torch.device("cpu"), tau=0.5,
                                gain_hi=2.0, gain_lo=0.5, warmup=2)
        c.set_base(base)
        cc = torch.zeros(P)
        for _ in range(10):
            c.apply_(torch.tensor([100, 100, 100, 0, 0, 0, 0], dtype=torch.int32),
                     torch.full((P,), 100.0), cc)
        self.assertTrue(torch.equal(cc, base * c.gain), (cc, base * c.gain))
        self.assertEqual(c.gain.tolist(), [2.0] * 3 + [0.5] * 4)
