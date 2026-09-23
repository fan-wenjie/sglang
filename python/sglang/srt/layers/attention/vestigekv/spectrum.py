# SPDX-License-Identifier: Apache-2.0
"""Telemetry: where a closing block's salience branch puts its above-cutoff energy.

Tier 1 keeps the rows whose branch departs from the block's low-pass trend, so
the residual is meant to hold per-row anomaly. When the input carries a
standing periodic component above the cutoff, the residual holds that instead,
the component is shared by every row, and sigma ranks phase rather than
distinctiveness. This module reports when that is happening. It changes
nothing: no row's fate depends on anything here.

What to read (offline study, docs/sidecar-notch-study.md, 35 request-layer
snapshots per corpus at ~64k): the peak's LOCATION separates the corpora and
its height does not. RULER's synthetic haystack peaks at bin 71-72 of a
4096-row block (a period near 57 tokens) in 35 of 35 snapshots; real prose
peaks at bin 16-17, the first bin past the cutoff, where any smooth signal
leaves a shoulder, in 35 of 35. Peakiness overlaps between the two (RULER's
minimum 50 sits under prose's maximum 93), so a magnitude threshold has no
margin and a detector should key on the bin.

Acting on it is a separate decision this module does not take, and the same
study argues against the two obvious actions: notching the component moves
sigma's selection AWAY from the rows the request's own decode queries score
highest (4.7% -> 4.3% overlap on RULER), and falling back to the exact path
would make every benchmark that trips the detector stop being evidence about
the sparse path.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def block_spectrum(side: torch.Tensor, kappa: int) -> tuple[int, float, float]:
    """(peak bin, peakiness, above-cutoff share) of one closed block.

    side: [T, D] the block's salience branch, T the close granularity. The
    mean is removed so bin 0 does not dominate; energy is summed over the D
    channels, which is what sigma's norm does. Peakiness is the largest
    above-cutoff bin against the median one: near 1 is broadband (a residual
    holding per-row anomaly), large is a standing component shared by rows.
    """
    x = side.float()
    e = torch.fft.rfft(x - x.mean(0, keepdim=True), dim=0).abs().pow(2).sum(1)
    above = e[kappa:]
    if above.numel() == 0:
        return -1, float("nan"), 0.0
    med = above.median()
    peakiness = float(above.max() / med) if float(med) > 0 else float("inf")
    total = float(e.sum())
    share = float(above.sum()) / total if total > 0 else 0.0
    return int(above.argmax()) + kappa, peakiness, share


class SpectrumTelemetry:
    """Per-layer running record of the closing blocks' above-cutoff peaks.

    One rFFT per closing block per layer, i.e. once per CLOSE_BLOCK tokens,
    off the per-step path entirely; the accumulator is host-side ints and
    floats, kept per layer so one layer's cadence cannot label another's
    numbers. Enabled by SGLANG_DEBUG_VESTIGEKV_SPECTRUM (blocks between
    reports), which is 0 in production: this is an operator's window into
    whether tier 1's signal is degenerate on the traffic it is being fed, not
    an input to any decision.
    """

    def __init__(self, kappa: int, every: int = 8):
        self.kappa, self.every = kappa, max(1, every)
        self.rec: dict[int, dict] = {}

    def _layer(self, lid: int) -> dict:
        return self.rec.setdefault(
            lid, {"n": 0, "since": 0, "bins": {}, "peak": 0.0, "share": 0.0}
        )

    def observe(self, side: torch.Tensor, block: int, *, lid: int) -> None:
        """Record every whole block of `side` ([N, D], N a multiple of block)."""
        r = self._layer(lid)
        for b in range(side.shape[0] // block):
            bin_, peakiness, share = block_spectrum(
                side[b * block : (b + 1) * block], self.kappa
            )
            if bin_ < 0:
                continue
            r["n"] += 1
            r["since"] += 1
            r["bins"][bin_] = r["bins"].get(bin_, 0) + 1
            r["peak"] += peakiness
            r["share"] += share
        if r["since"] >= self.every:
            r["since"] = 0
            self.dump(lid=lid)

    def dump(self, *, lid: int) -> None:
        r = self.rec.get(lid)
        if not r or not r["n"]:
            return
        top = sorted(r["bins"].items(), key=lambda kv: -kv[1])[:3]
        logger.info(
            "VKSPECTRUM layer=%d blocks=%d peak_bins=%s peakiness_mean=%.0f "
            "above_cutoff_share_mean=%.3f (telemetry only; a peak far above the "
            "cutoff means tier 1 is ranking a standing component)",
            lid,
            r["n"],
            ",".join(f"{b}:{c}" for b, c in top),
            r["peak"] / r["n"],
            r["share"] / r["n"],
        )
