r"""Per-(lane, layer) certificate width, adapted from the previous step.

The scan's width enters as one number per pack slot: cc = zp * sc /
sqrt(kv_lora - r), already a [P] fp32 device tensor that `update()` refreshes
every step. So adapting width needs no new plumbing and no kernel change -- it
needs cc written from an observable instead of from the build's fitted zp.

WHICH observable is the whole question, and it is settled by measurement, not
taste. Against the step dump's own label (an archived row truly beat the kept
maximum), matched max-over-heads on both sides, on 14196 adjacent-step pairs:

                          same-step AUC   stale-by-one
  fire fraction |F|/|A|
    branch r=0                  0.952         0.937
    served r=64                 0.856         0.761
  certified omitted mass
    served r=64                 0.843         0.804
    branch r=0                  0.156         0.210   <- ANTI-predictive

Two facts make this table the design.

First, the label is strongly autocorrelated across adjacent steps in the same
layer (phi = +0.46 served, +0.56 branch), so acting stale-by-one costs the
served arm 0.039 of AUC. A controller that sees only the previous step is
almost as good as one that sees this one, which is what makes this feasible at
all: max1 is computed before the archive is scanned, so stale-by-one is the
shape the architecture already forces.

Second, the two observables are the two halves of one partition, and they swap
roles. O = A \ F, so the omitted mass measures the complement's WEIGHT and the
fire fraction measures F's SIZE. Where F stays small the complement's mass is
informative; where F saturates, |F| is. On the branch arm r=0 makes q_res the
whole content query and rho_u the whole content norm, so cert_u is large, F ->
A, O -> empty, and leak_cert -> 0 exactly on the steps that needed the archive.
That is the inversion above, and it is a logical consequence of the gate's own
response, not noise.

STABILITY is why this file LATCHES instead of running a loop.

  leak_cert: widen -> more rows fire -> O shrinks -> leak_cert falls. Negative
  feedback; widening consumes the signal that asked for it. Safe to close a
  loop on.

  fire fraction: widen -> more rows fire -> fraction RISES. Positive feedback;
  a loop closed on it runs away to F = A.

Narrowing has the same defect with the sign flipped: narrow -> fewer rows fire
-> fraction falls -> narrow again, a ratchet toward zero width. So the fire
fraction supports NEITHER direction as a per-step loop, and one-way escalation
is not a fix either -- on the branch arm zp is already pinned at Z_MAX on the
workloads that matter, so there is no width left to escalate INTO and the whole
value there is narrowing.

What breaks the loop is to stop making the observation a function of its own
output. This controller observes for a warmup window of `warmup` steps at the
width the BUILD chose, latches one decision per (lane, layer) at the end of it,
and then never moves again. With no feedback path there is no stability
question, either direction is legitimate, and the result is a per-request
static per-layer width -- which is independently where the GLM port's own
analysis landed.

It therefore does not invent a control law; it replaces `archive_bound`'s
SENSOR. hard_rate is measured on the prompt's last calibration queries, out of
distribution for answer steps (0.1% against 22.85% miss) and measured
non-discriminative (0.047 / 0.031 / 0.008). The fire fraction is measured at
decode, on real answer steps, at AUC 0.952 same-step.

A genuine two-sided per-step loop needs the certified omitted mass, whose sign
permits one (widening lowers it) -- and that needs an accumulator the scan does
not store yet (see `needs_mass_accumulator`).

Everything here is in-place on preallocated buffers: the pack is captured in a
CUDA graph and `update()` revalidates it by refreshing tensor CONTENTS, so an
op that allocates or reads a device value to the host would either break
capture or stall the step it is meant to speed up.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D


class _WidthController:
    """Two hooks, both no-ops by default, so the pack never dispatches.

    A controller acts at one of two moments and they are not interchangeable:
    `write_cc` runs in update(), before the prologue, where only the previous
    step's outcome is known; `after_prologue` runs between the prologue and the
    scan, where this step's query geometry is available. Defaulting both to
    no-ops lets the pack call both unconditionally -- the alternative is an
    if-chain on the controller type, which is exactly what silently lost a
    branch once already.
    """

    def write_cc(self, **_kw):
        return

    def after_prologue(self, **_kw):
        return


class CertWidthController(_WidthController):
    """One-way width escalation per pack slot, driven on device.

    Owns `base` (the width the build fitted) and `gain` (>= 1, monotone within
    a request). The scan reads `base * gain`.
    """

    def __init__(self, P: int, device, tau: float = 0.5, gain_hi: float = 1.0,
                 gain_lo: float = 1.0, warmup: int = 16):
        # tau on the fire FRACTION, not a row count: |A| grows with context and
        # a count threshold would drift into always-on. 0.5 sits between the
        # measured medians -- 1.0000 when the archive was needed, 0.0000 when
        # not -- so it is not tuned against the same data that chose it.
        self.tau = float(tau)
        # Multiplicative on cc, hence on zp. 1.0 for both is an exact no-op, so
        # the arm can be enabled without moving a served number.
        self.gain_hi = float(gain_hi)   # applied where the fraction crossed tau
        self.gain_lo = float(gain_lo)   # applied where it did not
        self.warmup = int(warmup)
        self.base = torch.zeros(P, device=device)
        self.gain = torch.ones(P, device=device)
        self.ff = torch.zeros(P, device=device)     # running mean over warmup
        self.seen = torch.zeros(P, device=device)   # steps observed, unlatched
        self.latched = torch.zeros(P, device=device, dtype=torch.bool)
        # Scratch. Every op below writes through one of these: the pack is
        # replayed from a captured graph, so an op that allocates is not
        # replayable, and torch.where / Tensor.add(scalar) both allocate.
        self._x = torch.zeros(P, device=device)
        self._w = torch.zeros(P, device=device)
        self._n = torch.zeros(P, device=device)
        self._open = torch.zeros(P, device=device, dtype=torch.bool)
        self._ready = torch.zeros(P, device=device, dtype=torch.bool)
        self._now = torch.zeros(P, device=device, dtype=torch.bool)
        self._hi = torch.zeros(P, device=device, dtype=torch.bool)

    def set_base(self, cc_host: torch.Tensor) -> None:
        """Adopt the build's fitted widths, vectorised.

        The caller holds them as one [P] host tensor rather than assigning
        cc[i] per slot: that store is a scalar write into a CUDA tensor, which
        stages its own copy and stalls the host, once per lane per LAYER per
        step.
        """
        self.base.copy_(cc_host, non_blocking=True)

    def reset(self, lanes: torch.Tensor | None = None) -> None:
        """Drop the latch, so a slot's new occupant inherits no width decision."""
        if lanes is None:
            self.gain.fill_(1.0)
            self.ff.zero_()
            self.seen.zero_()
            self.latched.fill_(False)
        else:
            self.gain.index_fill_(0, lanes, 1.0)
            self.ff.index_fill_(0, lanes, 0.0)
            self.seen.index_fill_(0, lanes, 0.0)
            self.latched.index_fill_(0, lanes, False)

    def write_cc(self, *, fired, cc_host, a_len, cc_out, **_ignored):
        """Single entry point the pack calls on whichever controller it holds.

        The pack used to branch on which controller was installed and pick the
        matching call; one of those branches went missing in an edit and the
        arm died at serve time with an AttributeError, after the unit tests --
        which exercise the controllers directly -- all passed. One name that
        every controller answers to removes the branch and the bug class with
        it. Extra keywords are accepted and ignored so the pack can pass
        everything either controller might want without knowing which it has.
        """
        self.set_base(cc_host)
        self.apply_(fired, a_len, cc_out)

    @torch.inference_mode()
    def apply_(self, fired: torch.Tensor, a_len: torch.Tensor,
               cc_out: torch.Tensor) -> None:
        """Observe, latch once per lane, then hold. In place, allocation-free.

        `fired` [P] is the PREVIOUS step's fired-row count -- update() runs
        before the scan -- and a_len [P] the archive length. Both are device
        tensors the pack already refreshes, so nothing is read back to the host.
        """
        # x = fire fraction this step
        self._n.copy_(a_len)
        self._n.clamp_min_(1.0)
        self._x.copy_(fired)
        self._x.div_(self._n)

        # running mean on lanes that have not latched: ff += (x - ff)/(seen+1).
        # A mean, not the latest value, so one freak step cannot decide a request.
        torch.logical_not(self.latched, out=self._open)
        self._w.copy_(self._open)
        self._n.copy_(self.seen)
        self._n.add_(1.0)
        self._x.sub_(self.ff)
        self._x.div_(self._n)
        self._x.mul_(self._w)
        self.ff.add_(self._x)
        self.seen.add_(self._w)

        # lanes latching on THIS step: enough observations, not yet latched
        torch.ge(self.seen, float(self.warmup), out=self._ready)
        torch.logical_and(self._ready, self._open, out=self._now)
        self._w.copy_(self._now)

        # target = gain_hi where the observed mean crossed tau, else gain_lo
        torch.gt(self.ff, self.tau, out=self._hi)
        self._x.copy_(self._hi)
        self._x.mul_(self.gain_hi - self.gain_lo)
        self._x.add_(self.gain_lo)

        # gain += w * (target - gain): moves only the latching lanes
        self._x.sub_(self.gain)
        self._x.mul_(self._w)
        self.gain.add_(self._x)
        self.latched.logical_or_(self._now)

        torch.mul(self.base, self.gain, out=cc_out)


def needs_mass_accumulator() -> str:
    """Why two-sided control is not here yet.

    De-escalation needs the certified omitted mass M = sum over non-fired
    archived rows of exp(b_u - max1), whose sign permits a loop (widening
    lowers it). The scan computes every b_u already and compares it to the
    threshold, so M costs one fp32 accumulator and one exp per row with no
    extra memory traffic -- but it is not stored today, and adding it changes a
    decode-path Triton kernel, which must be disassembled and checked for
    spills before it is timed (a spilling build slows every step, including the
    ones whose feature never fires).
    """
    return "scan must store sum(exp(b_u - max1)) over non-fired rows"


class DerivedWidthController(_WidthController):
    r"""Per-layer certificate width with NO tuned constant but the spec itself.

    CertWidthController above works, but it buys adaptivity with four new
    constants -- tau, gain_hi, gain_lo, warmup -- which is a bad trade: a method
    that needs four numbers chosen by hand is not obviously better than one
    number chosen by hand. This class removes all four by deriving the width
    instead of scaling it.

    The chain is short because the pieces already exist.

    1. zp is ALREADY a pure function of one target. build() computes
       zp = z_(k) with k = ceil((n_cal+1) * rho), the conformal order statistic
       of the calibration sample. Nothing is tuned there; rho is the whole
       input. So the right thing to adapt is rho, not a multiplier on cc -- a
       gain on cc moves the width off the order statistic and forfeits the
       distribution-free guarantee that made it meaningful.

    2. rho follows from the conjunction, not from taste. An answer is a
       conjunction over the steps that read the archive, and the paper's own
       arithmetic is that a 0.929 per-step recall is 6.4e-20 over 600 steps. So
       fix a SPEC: the whole generation should fail with probability at most
       delta. Only the steps where the archive holds the winner can fail; if
       that happens at rate h, there are about h*T of them, and a union bound
       gives

           h * T * (1 - rho)  <=  delta      =>      rho >= 1 - delta/(h*T)

       which is the per-layer target. A layer whose archive never binds has
       h -> 0 and no constraint, so it narrows. A layer that binds every step
       gets rho = 1 - delta/T and stays wide. Continuous in h, so the tau
       THRESHOLD disappears, and gain_hi/gain_lo disappear with it because the
       width is now read off the order statistic rather than scaled.

    3. h is measured, and measured in the safe direction. A row fires exactly
       when its certified bound beats the threshold, so "at least one row
       fired" is implied by "an archived row truly beat the kept max" at the
       certificate's own confidence. Counting steps with |F| > 0 therefore
       OVER-estimates h, which over-tightens rho, which widens. Wrong in the
       direction that costs work rather than answers.

    4. warmup disappears because the estimate carries its own uncertainty. Take
       the Wilson upper bound on h rather than the sample rate: with no
       evidence it is 1, so the layer starts at maximum width and NARROWS as it
       earns the right to. There is no window to choose and no moment at which
       a decision is taken -- the width simply tracks the evidence.

    What remains is delta, and delta is a specification rather than a knob: it
    is the answer-level failure rate the deployment will accept. Z_MAX stops
    being a constant to pick as well, and becomes a diagnostic: if the required
    k exceeds n_cal, the calibration sample CANNOT certify the spec for that
    layer, and the honest response is to serve the sample maximum and say so,
    not to silently clamp.

    Not removed, and worth naming: T. Here it is the number of decode steps
    taken so far, so the bound covers the prefix generated so far and tightens
    as the answer lengthens. Using a planned max_new_tokens instead would make
    it a fixed budget spent up front. That is a modelling choice, not a tuned
    value, but it is a choice.
    """

    def __init__(self, P: int, n_cal_max: int, device, delta: float = 0.01):
        # The only number, and it is the spec: acceptable probability that the
        # generated answer is wrong BECAUSE a needed archived row was missed.
        # Split once, half to the certificate's per-step miss and half to the
        # uncertainty in h, so a single delta covers both.
        self.delta = float(delta)
        self.n_grid = int(n_cal_max)
        # z for the Wilson bound at 1 - delta/2, computed once on the host.
        self.zc = _normal_quantile(1.0 - 0.5 * self.delta)
        self.z_sorted = torch.zeros(P, n_cal_max, device=device)  # quantile grid
        # Per slot: a build can stop short of N_CAL_MAX (the window doubles
        # until MIN_HARD hard samples exist), so the reachable target
        # n_cal/(n_cal+1) differs per layer and k must be formed against each
        # slot's own sample size.
        self.n_cal = torch.zeros(P, device=device)
        self.n = torch.zeros(P, device=device)      # steps observed
        self.a = torch.zeros(P, device=device)      # steps with |F| > 0
        self.rho = torch.zeros(P, device=device)    # derived target, for logs
        self.hbar = torch.ones(P, device=device)    # Wilson upper bound on h
        self.infeasible = torch.zeros(P, device=device, dtype=torch.bool)
        self._t = torch.zeros(P, device=device)
        self._u = torch.zeros(P, device=device)
        self._k = torch.zeros(P, 1, device=device, dtype=torch.int64)
        self._g = torch.zeros(P, 1, device=device)
        self._b = torch.zeros(P, device=device, dtype=torch.bool)

    def set_calibration(self, z_sorted: torch.Tensor,
                        n_cal: torch.Tensor) -> None:
        """Adopt the build's SORTED calibration samples, [P, N_CAL_MAX], and
        each slot's true sample size.

        Kept because the order statistic has to be re-read at a new target, and
        re-deriving it needs the sample, not just the one quantile build chose.
        Rows are padded with their own sample maximum, so an index past n_cal
        reads the max -- which is what "this sample cannot certify that target"
        should serve.
        """
        self.z_sorted.copy_(z_sorted, non_blocking=True)
        self.n_cal.copy_(n_cal, non_blocking=True)

    def reset(self, lanes: torch.Tensor | None = None) -> None:
        if lanes is None:
            self.n.zero_()
            self.a.zero_()
            self.hbar.fill_(1.0)
            self.infeasible.fill_(False)
        else:
            self.n.index_fill_(0, lanes, 0.0)
            self.a.index_fill_(0, lanes, 0.0)
            self.hbar.index_fill_(0, lanes, 1.0)
            self.infeasible.index_fill_(0, lanes, False)

    def write_cc(self, *, fired, z, n_cal, fac_host, fac, cc_out, **_ignored):
        """See CertWidthController.write_cc: one name, no dispatch in the pack."""
        fac.copy_(fac_host, non_blocking=True)
        self.set_calibration(z, n_cal)
        self.apply_(fired, fac, cc_out)

    @torch.inference_mode()
    def apply_(self, fired: torch.Tensor, cc_factor: torch.Tensor,
               cc_out: torch.Tensor) -> None:
        """Width for the next step, derived. In place; allocates nothing.

        cc_factor [P] is scale / sqrt(kv_lora - r) per slot, which the caller
        already knows per tier; keeping it per slot rather than scalar means a
        pack holding tiers of different rank still gets each one's own factor.
        """
        # a += [ |F| > 0 ]; n += 1
        torch.gt(fired, 0, out=self._b)
        self._t.copy_(self._b)
        self.a.add_(self._t)
        self.n.add_(1.0)

        # Wilson upper bound on h at 1 - delta/2. n = 0 leaves hbar at 1.
        z2 = self.zc * self.zc
        self._t.copy_(self.a)                      # a
        self._u.copy_(self.n)
        self._u.sub_(self.a)                       # n - a
        self._t.mul_(self._u)
        self._t.div_(self.n.clamp_min(1.0))        # a(n-a)/n
        self._t.add_(0.25 * z2)
        self._t.sqrt_()
        self._t.mul_(self.zc)
        self._t.add_(self.a)
        self._t.add_(0.5 * z2)
        self._u.copy_(self.n)
        self._u.add_(z2)
        self._t.div_(self._u)
        self._t.clamp_(0.0, 1.0)
        # keep hbar = 1 until a step has been seen
        torch.gt(self.n, 0.0, out=self._b)
        self._u.copy_(self._b)
        self._t.mul_(self._u)
        self._u.mul_(-1.0)
        self._u.add_(1.0)
        self._t.add_(self._u)
        self.hbar.copy_(self._t)

        # rho = 1 - delta / (2 * hbar * T), T = steps so far
        self._t.copy_(self.hbar)
        self._t.mul_(self.n)
        self._t.clamp_min_(1e-9)
        self._u.fill_(0.5 * self.delta)
        self._u.div_(self._t)
        self._u.mul_(-1.0)
        self._u.add_(1.0)
        self._u.clamp_(0.0, 1.0)
        self.rho.copy_(self._u)

        # Conformal rank for this target, k = ceil((n+1) * rho), then the grid
        # is addressed by the LEVEL k/n rather than by rho directly. Going
        # through k matters at the endpoint: rho = n/(n+1) is the tightest
        # reachable target and must land on the sample maximum, which indexing
        # by rho misses by one grid point.
        self._t.copy_(self.n_cal)
        self._t.add_(1.0)
        self._u.mul_(self._t)
        self._u.ceil_()
        # A sample of n points cannot certify past k = n; flag, do not pretend.
        torch.gt(self._u, self.n_cal, out=self.infeasible)
        self._t.copy_(self.n_cal)
        self._t.clamp_min_(1.0)
        torch.minimum(self._u, self._t, out=self._u)
        self._u.div_(self._t)                   # level = k/n in (0, 1]
        self._u.mul_(self.n_grid - 1)
        self._u.round_()
        self._u.clamp_(0.0, float(self.n_grid - 1))
        self._k.copy_(self._u.unsqueeze(1).to(torch.int64))

        # zp = z_(k); cc = zp * scale / sqrt(kv_lora - r)
        torch.gather(self.z_sorted, 1, self._k, out=self._g)
        cc_out.copy_(self._g.squeeze(1))
        cc_out.mul_(cc_factor)


def _normal_quantile(p: float) -> float:
    """Inverse standard normal, Acklam's rational approximation.

    Inlined rather than pulled from scipy: this runs once per process on a
    scalar, and the engine does not depend on scipy.
    """
    import math

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    rr = q * q
    return (((((a[0]*rr+a[1])*rr+a[2])*rr+a[3])*rr+a[4])*rr+a[5])*q / \
           (((((b[0]*rr+b[1])*rr+b[2])*rr+b[3])*rr+b[4])*rr+1)


class GeometryWidthController(_WidthController):
    r"""Width from the query's own geometry, decided before the scan.

    Scored against the right label -- the top-1 row actually LOST, not the
    proxy "an archived row beat the kept maximum" -- qperp_rel = ||q_res|| /
    ||q_c|| separates per layer at AUC 0.823 / 0.866 / 0.871 / 0.860 / 0.922 on
    the five sig-cont layers that ever lose a top-1. At each layer's own p70
    quantile it catches 91.5% of losses while widening 21.5% of steps; p60
    catches 100% at 28.6%.

    PER LAYER is not decoration. Pooled, the same quantity reads only 0.740,
    because it carries a strong per-layer offset -- median 0.339 to 0.698 across
    the seven layers -- and pooling mixes baselines that are not comparable.
    That is Simpson's paradox, and it is the shape that hid this signal until
    the layers were separated.

    Three properties decide it against the alternatives here.

      No feedback. It is a property of the QUERY, not of the fire set, so it
      does not move when the width moves. The fire fraction rises when width
      rises (a loop on it diverges); the certified omitted mass falls (a loop is
      safe but needs an accumulator the scan does not store).

      No staleness. The fused prologue computes ||q_res|| before the scan, so
      the decision uses this step's own geometry. Every other candidate was
      stale by a step or by a layer.

      Free. fused_prologue line 69 is qres2 = qnorm2 - sum(qsk*qsk): both norms
      are already in registers there. Until that store is added -- a decode-path
      Triton change, so it needs a disassembly pass first -- ||q_c|| is recovered
      with one small reduction over the query buffer.

    It is dead on synthetic retrieval (sig-mrcr AUC 0.514, 0.384-0.582 per
    layer), which is precisely where cross-layer prediction works (phi +0.41,
    previous layer's signal at AUC 0.78). Neither covers both regimes; together
    they do.

    The threshold is each layer's own running quantile, so no absolute level is
    ever chosen -- the quantity's per-layer offset is exactly what a fixed
    threshold would get wrong. It is tracked by the Robbins-Monro update

        thr <- thr + eta * (1[x > thr] - (1 - level))

    whose fixed point is the level-quantile, because it is elementwise and
    therefore capturable: a ring buffer's write index changes every step and
    would be baked in at capture. eta is the reciprocal of the memory length,
    not a free parameter -- it says how many steps the estimate averages over.
    """

    def __init__(self, P: int, H: int, device, level: float = 0.70,
                 memory: int = 256):
        self.level = float(level)
        self.eta = 1.0 / float(memory)
        self.thr = torch.zeros(P, device=device)
        self.rel = torch.zeros(P, device=device)
        self.seen = torch.zeros(P, device=device)
        self._x = torch.zeros(P, device=device)
        self._w = torch.zeros(P, device=device)
        self._b = torch.zeros(P, device=device, dtype=torch.bool)
        # Per-head scratch, allocated here and not on first use: a lazy
        # allocation lands inside graph capture, which is the bug this class's
        # two predecessors each had a version of.
        self._hb = torch.zeros(P, H, device=device)

    def reset(self, lanes: torch.Tensor | None = None) -> None:
        """A new occupant of a slot inherits no threshold."""
        if lanes is None:
            self.thr.zero_()
            self.seen.zero_()
        else:
            self.thr.index_fill_(0, lanes, 0.0)
            self.seen.index_fill_(0, lanes, 0.0)

    @torch.inference_mode()
    def after_prologue(self, *, qrel, relthr_out, **_ignored):
        """Advance this layer's qperp_rel threshold, in place, from this step.

        It no longer writes cc. Scaling cc moves only the certificate term, and
        on the served arm firing is not certificate-driven -- Spearman between
        qperp_rel and the fire fraction is -0.292 on the layer carrying 82% of
        the fetch. The actuator is the MARGIN, and the prologue applies it per
        head in the same kernel that computes qperp_rel, so all this has to do
        is keep the threshold that kernel compares against.

        qrel [P, H] is what the prologue just emitted. The threshold is per
        slot, so the heads are reduced by median -- the aggregation the offline
        separation was measured at.
        """
        self._hb.copy_(qrel)
        self.rel.copy_(self._hb.median(dim=1).values)

        # Robbins-Monro step toward the level-quantile. Elementwise, hence
        # capturable; a ring buffer's write index moves every step and would be
        # baked in at capture. eta is the reciprocal of the memory length.
        torch.gt(self.rel, self.thr, out=self._b)
        self._w.copy_(self._b)
        self._w.sub_(1.0 - self.level)
        self._w.mul_(self.eta)
        self.thr.add_(self._w)
        self.seen.add_(1.0)
        relthr_out.copy_(self.thr)
