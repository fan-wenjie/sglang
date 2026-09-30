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


class CertWidthController:
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
