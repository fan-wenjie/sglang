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


class TestDerivedWidthController(unittest.TestCase):
    """Width with no tuned constant but the spec.

    The claim being pinned is that rho is DERIVED -- from delta, the measured
    binding rate h, and the conjunction length -- so these check the arithmetic
    against hand-computed values rather than just checking it moves.
    """

    def _ctl(self, P=3, n_cal=64, delta=0.01):
        from sglang.srt.layers.attention.vestigekv.width_control import (
            DerivedWidthController,
        )
        c = DerivedWidthController(P, n_cal, torch.device("cpu"), delta=delta)
        # calibration sample z_(1..64) = 1..64, so z_(k) reads back as k
        c.set_calibration(torch.arange(1, n_cal + 1).float().expand(P, n_cal)
                          .contiguous(),
                          torch.full((P,), float(n_cal)))
        self._fac = torch.full((P,), 1.0 / (512 ** 0.5))
        return c

    def test_no_evidence_starts_at_maximum_width(self):
        """With nothing observed the Wilson bound is ~1, so the layer assumes
        the archive binds every step and serves the widest certificate. Width is
        earned, not granted -- which is why no warmup constant is needed."""
        c = self._ctl()
        cc = torch.zeros(3)
        c.apply_(torch.tensor([1, 1, 1], dtype=torch.int32), self._fac, cc)
        self.assertGreater(c.hbar.min().item(), 0.5)
        self.assertGreater(c.rho.min().item(), 0.99)

    def test_the_64_sample_cannot_express_an_answer_level_target(self):
        """Deriving rho from a spec exposes a hard limit of the calibration.

        A conformal order statistic over n_cal samples can only express targets
        up to n_cal/(n_cal+1) = 64/65 = 0.9846. But an ANSWER-level spec over a
        few hundred steps needs per-step recall far tighter than that: 200 steps
        at h=0.03 is about 6 binding steps, and delta=0.01 over them is
        rho = 1 - 0.005/6.4 = 0.99922, which needs k = 65 on a 64-sample.

        So the target is not reachable and the layer correctly reports
        infeasible and serves the sample maximum. This is the quantitative form
        of the per-step-versus-conjunction problem: 0.93 per step is 6e-20 over
        600, and a 64-sample quantile tops out three orders of magnitude short of
        what the conjunction demands. Going further needs the PARAMETRIC fit,
        which is the one thing an order statistic cannot do -- it cannot exceed
        its own sample maximum.
        """
        c = self._ctl(delta=0.01)
        cc = torch.zeros(3)
        for _ in range(200):
            c.apply_(torch.zeros(3, dtype=torch.int32), self._fac, cc)
        # the evidence DID accumulate: h is bounded well away from 1
        self.assertLess(c.hbar.max().item(), 0.05,
                        "200 non-binding steps did not shrink the bound")
        # but the derived target still runs off the sample
        self.assertGreaterEqual(c.rho.min().item(), 64.0 / 65.0 - 1e-9)
        self.assertTrue(bool(c.infeasible.all()),
                        "should report the spec unreachable at this sample size")
        self.assertAlmostEqual(cc.max().item(), 64.0 / (512 ** 0.5), places=5)

    def test_the_saving_is_across_layers_at_equal_T_not_over_time(self):
        """A never-binding layer gets a looser target than an always-binding one.

        Not a narrowing over TIME: rho = 1 - delta/(2*hbar*T) and hbar*T is the
        expected number of binding steps, which only grows as the answer
        lengthens -- correctly, because the conjunction it has to survive is
        getting longer. What the derivation buys is the gap BETWEEN layers at
        the same T, and that gap is the whole saving.
        """
        c = self._ctl(P=2, delta=0.5)
        cc = torch.zeros(2)
        for _ in range(400):
            # lane 0 binds every step, lane 1 never does
            c.apply_(torch.tensor([3, 0], dtype=torch.int32), self._fac, cc)
        self.assertGreater(c.hbar[0].item(), 0.9)
        self.assertLess(c.hbar[1].item(), 0.02)
        self.assertGreater(c.rho[0].item(), c.rho[1].item(),
                           "binding layer should demand the tighter target")
        self.assertGreater(cc[0].item(), cc[1].item(),
                           "binding layer should serve the wider certificate")
        self.assertTrue(bool(c.infeasible[0]), "always-binding must be infeasible")
        self.assertFalse(bool(c.infeasible[1]), "never-binding should be reachable")

    def test_a_layer_that_always_fires_stays_wide(self):
        c = self._ctl()
        cc = torch.zeros(3)
        for _ in range(200):
            c.apply_(torch.tensor([5, 5, 5], dtype=torch.int32), self._fac, cc)
        self.assertGreater(c.hbar.min().item(), 0.9)
        # rho = 1 - delta/(2*hbar*T) with T=200, hbar~1, delta=0.01 -> ~0.999975
        self.assertGreater(c.rho.min().item(), 0.9999)
        self.assertTrue(bool(c.infeasible.all()),
                        "k should run off a 64-sample at rho > 1 - 1/65")

    def test_infeasible_is_flagged_not_silently_clamped(self):
        """When the sample cannot certify the spec the layer says so and serves
        the sample maximum. That replaces Z_MAX-as-a-constant with a diagnostic."""
        c = self._ctl(n_cal=64)
        cc = torch.zeros(3)
        for _ in range(50):
            c.apply_(torch.tensor([1, 1, 1], dtype=torch.int32), self._fac, cc)
        self.assertTrue(bool(c.infeasible.all()))
        # grid top is z_(64)=64; cc = z * scale / sqrt(kv_lora - r)
        expect = 64.0 / (512 ** 0.5)
        for v in cc.tolist():
            self.assertAlmostEqual(v, expect, places=5,
                                   msg="not serving the sample max")

    def test_rho_matches_the_union_bound_by_hand(self):
        """rho = 1 - delta/(2*hbar*T), checked against a hand computation."""
        c = self._ctl(P=1, delta=0.02)
        cc = torch.zeros(1)
        for _ in range(100):
            c.apply_(torch.zeros(1, dtype=torch.int32), self._fac, cc)
        h = c.hbar[0].item()
        expected = 1.0 - 0.01 / max(h * 100.0, 1e-9)
        self.assertAlmostEqual(c.rho[0].item(), max(0.0, min(1.0, expected)),
                               places=6)

    def test_delta_is_the_only_knob_that_moves_width(self):
        """Tightening the spec must widen the certificate, monotonically."""
        widths = []
        for d in (0.2, 0.05, 0.01):
            c = self._ctl(P=1, delta=d)
            cc = torch.zeros(1)
            for _ in range(100):
                c.apply_(torch.zeros(1, dtype=torch.int32), self._fac, cc)
            widths.append(cc[0].item())
        self.assertEqual(widths, sorted(widths),
                         f"width not monotone in the spec: {widths}")

    def test_apply_allocates_nothing(self):
        c = self._ctl()
        cc = torch.zeros(3)
        names = ["z_sorted", "n_cal", "n", "a", "rho", "hbar", "infeasible",
                 "_t", "_u", "_k", "_g", "_b"]
        before = {n: getattr(c, n).data_ptr() for n in names}
        before["cc"] = cc.data_ptr()
        for k in range(20):
            c.apply_(torch.tensor([k % 2] * 3, dtype=torch.int32), self._fac, cc)
        for n in names:
            self.assertEqual(before[n], getattr(c, n).data_ptr(), f"{n} realloc")
        self.assertEqual(before["cc"], cc.data_ptr())


class TestBothControllersAnswerOneEntryPoint(unittest.TestCase):
    """The pack calls write_cc and nothing else.

    It used to branch on which controller was installed; an edit dropped one
    branch and the derived arm died at serve time with an AttributeError, after
    every unit test passed -- because the unit tests drive the controllers
    directly and never exercised the pack's dispatch. This pins the property
    that made the branch unnecessary.
    """

    def _kwargs(self, P, cc_out):
        from sglang.srt.layers.attention.vestigekv import defaults as D
        return dict(
            fired=torch.zeros(P, dtype=torch.int32),
            cc_host=torch.full((P,), 0.5),
            a_len=torch.full((P,), 100.0),
            z=torch.arange(1, D.N_CAL_MAX + 1).float().expand(
                P, D.N_CAL_MAX).contiguous(),
            n_cal=torch.full((P,), float(D.N_CAL_MAX)),
            fac_host=torch.full((P,), 1.0 / (512 ** 0.5)),
            fac=torch.zeros(P),
            cc_out=cc_out,
        )

    def test_every_controller_accepts_the_same_call(self):
        from sglang.srt.layers.attention.vestigekv import defaults as D
        from sglang.srt.layers.attention.vestigekv.width_control import (
            CertWidthController,
            DerivedWidthController,
            GeometryWidthController,
        )
        P = 3
        made = [
            CertWidthController(P, torch.device("cpu"), tau=0.5, warmup=2),
            DerivedWidthController(P, D.N_CAL_MAX, torch.device("cpu"),
                                   delta=0.05),
            GeometryWidthController(P, 4, torch.device("cpu"), level=0.7),
        ]
        for c in made:
            cc = torch.zeros(P)
            kw = self._kwargs(P, cc)
            for _ in range(4):
                c.write_cc(**kw)
                c.after_prologue(qrel=torch.zeros(P, 4),
                                 relthr_out=torch.zeros(P), **kw)
            self.assertTrue(torch.isfinite(cc).all(),
                            f"{type(c).__name__} wrote non-finite cc: {cc}")

    def test_the_pack_calls_write_cc_and_no_variant_of_it(self):
        """A second entry point is how the dispatch crept back in last time."""
        import pathlib
        src = pathlib.Path(
            "python/sglang/srt/layers/attention/vestigekv/batched_step.py"
        ).read_text()
        self.assertIn("self._width.write_cc(", src)
        for gone in ("self._width.set_base(", "self._width.apply_(",
                     "self._width.set_calibration("):
            self.assertNotIn(gone, src, f"pack still calls {gone} directly")


class TestGeometryWidthController(unittest.TestCase):
    """The threshold the kernel compares qperp_rel against, per layer.

    The controller no longer writes cc -- scaling cc moves only the certificate
    term, and on the served arm firing is not certificate-driven. The actuator
    is the margin, applied per head inside the prologue that computes
    qperp_rel, so all this does is keep that comparison threshold current.

    Pinned here: a cold layer runs WIDE (threshold 0, every head above it), the
    threshold converges to the layer's own quantile, each layer tracks its own
    offset (0.339 to 0.698 across the seven, so a shared absolute level is
    exactly what fails), and nothing allocates after construction.
    """

    def _ctl(self, P=3, H=4, level=0.7, memory=64):
        from sglang.srt.layers.attention.vestigekv.width_control import (
            GeometryWidthController,
        )
        return GeometryWidthController(P, H, torch.device("cpu"), level=level,
                                       memory=memory)

    def _step(self, c, rel, P=3, H=4):
        out = torch.zeros(P)
        qrel = torch.tensor(rel, dtype=torch.float32).view(P, 1).expand(P, H)
        c.after_prologue(qrel=qrel.contiguous(), relthr_out=out)
        return out

    def test_a_cold_layer_starts_at_zero_so_every_head_runs_wide(self):
        c = self._ctl()
        self.assertEqual(c.thr.tolist(), [0.0] * 3)
        out = self._step(c, [0.1, 0.5, 0.9])
        self.assertTrue(bool((out >= 0).all()), out)

    def test_the_exceedance_rate_converges_to_one_minus_level(self):
        """The operational property. Asserting on thr itself is the wrong
        target: Robbins-Monro oscillates around its fixed point with amplitude
        set by the step size, so one final reading is a draw from that
        oscillation rather than the estimate."""
        import random
        c = self._ctl(P=1, H=2, level=0.7, memory=512)
        rng = random.Random(0)
        above = 0
        N = 8000
        for i in range(N):
            v = rng.random()
            t = self._step(c, [v], P=1, H=2)
            if i >= N // 2 and v > t[0].item():
                above += 1
        rate = above / (N - N // 2)
        self.assertAlmostEqual(rate, 0.30, delta=0.04,
                               msg=f"exceedance {rate:.3f}, thr={c.thr[0]:.4f}")

    def test_each_layer_tracks_its_own_offset(self):
        import random
        c = self._ctl(P=2, H=2, level=0.7, memory=64)
        rng = random.Random(1)
        for _ in range(6000):
            self._step(c, [0.2 * rng.random(), 0.5 + 0.5 * rng.random()],
                       P=2, H=2)
        self.assertLess(c.thr[0].item(), 0.25, f"lo thr={c.thr[0].item():.3f}")
        self.assertGreater(c.thr[1].item(), 0.7, f"hi thr={c.thr[1].item():.3f}")

    def test_allocates_nothing_after_construction(self):
        c = self._ctl()
        names = ["thr", "rel", "seen", "_x", "_w", "_b", "_hb"]
        before = {n: getattr(c, n).data_ptr() for n in names}
        for k in range(10):
            self._step(c, [0.1 * k, 0.5, 0.9])
        for n in names:
            self.assertEqual(before[n], getattr(c, n).data_ptr(), f"{n} realloc")

    def test_reset_clears_the_threshold(self):
        c = self._ctl()
        for _ in range(100):
            self._step(c, [0.9, 0.9, 0.9])
        self.assertGreater(c.thr.max().item(), 0.0)
        c.reset(torch.tensor([1]))
        self.assertEqual(c.thr[1].item(), 0.0)
        self.assertGreater(c.thr[0].item(), 0.0)


class TestBothControllersAnswerOneEntryPoint(unittest.TestCase):
    """The pack calls write_cc and nothing else.

    It used to branch on which controller was installed; an edit dropped one
    branch and the derived arm died at serve time with an AttributeError, after
    every unit test passed -- because the unit tests drive the controllers
    directly and never exercised the pack's dispatch. This pins the property
    that made the branch unnecessary.
    """

    def _kwargs(self, P, cc_out):
        from sglang.srt.layers.attention.vestigekv import defaults as D
        return dict(
            fired=torch.zeros(P, dtype=torch.int32),
            cc_host=torch.full((P,), 0.5),
            a_len=torch.full((P,), 100.0),
            z=torch.arange(1, D.N_CAL_MAX + 1).float().expand(
                P, D.N_CAL_MAX).contiguous(),
            n_cal=torch.full((P,), float(D.N_CAL_MAX)),
            fac_host=torch.full((P,), 1.0 / (512 ** 0.5)),
            fac=torch.zeros(P),
            cc_out=cc_out,
        )

    def test_every_controller_accepts_the_same_call(self):
        from sglang.srt.layers.attention.vestigekv import defaults as D
        from sglang.srt.layers.attention.vestigekv.width_control import (
            CertWidthController,
            DerivedWidthController,
            GeometryWidthController,
        )
        P = 3
        made = [
            CertWidthController(P, torch.device("cpu"), tau=0.5, warmup=2),
            DerivedWidthController(P, D.N_CAL_MAX, torch.device("cpu"),
                                   delta=0.05),
            GeometryWidthController(P, 4, torch.device("cpu"), level=0.7),
        ]
        for c in made:
            cc = torch.zeros(P)
            kw = self._kwargs(P, cc)
            for _ in range(4):
                c.write_cc(**kw)
                c.after_prologue(qrel=torch.zeros(P, 4),
                                 relthr_out=torch.zeros(P), **kw)
            self.assertTrue(torch.isfinite(cc).all(),
                            f"{type(c).__name__} wrote non-finite cc: {cc}")

    def test_the_pack_calls_write_cc_and_no_variant_of_it(self):
        """A second entry point is how the dispatch crept back in last time."""
        import pathlib
        src = pathlib.Path(
            "python/sglang/srt/layers/attention/vestigekv/batched_step.py"
        ).read_text()
        self.assertIn("self._width.write_cc(", src)
        for gone in ("self._width.set_base(", "self._width.apply_(",
                     "self._width.set_calibration("):
            self.assertNotIn(gone, src, f"pack still calls {gone} directly")


