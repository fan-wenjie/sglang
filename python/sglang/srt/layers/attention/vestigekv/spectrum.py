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


def block_stats(side: torch.Tensor, kappa: int) -> torch.Tensor:
    """[4] on `side`'s device: peak bin, its energy, the median above-cutoff
    bin, the whole block's energy. Stays on the device on purpose -- reading
    any of it here would stall the stream at every block close, which measured
    195 us per block per layer against 12 for the arithmetic.

    Energy is summed over the D channels, which is what sigma's norm does.
    Bin 0 carries the block's mean and is dropped rather than subtracted: the
    cutoff is at bin kappa, so the mean never reaches the statistics anyway,
    and the subtraction was a launch (39 of the 156 us this used to cost).
    """
    e = torch.fft.rfft(side.float(), dim=0).abs().pow(2).sum(1)[1:]
    above = e[kappa - 1 :]
    if above.numel() == 0:
        return torch.zeros(4, device=e.device)
    return torch.stack(
        [
            above.argmax().float() + kappa,
            above.max(),
            above.median(),
            e.sum(),
        ]
    )


def read_stats(row) -> tuple[int, float, float]:
    """(peak bin, peakiness, above-cutoff share) from one row of block_stats.

    Peakiness is the largest above-cutoff bin against the median one: near 1
    is broadband (a residual holding per-row anomaly), large is a standing
    component shared by rows. The share is that peak against the block's
    energy outside bin 0.
    """
    bin_, mx, med, total = (float(v) for v in row)
    peakiness = mx / med if med > 0 else float("inf")
    # the above-cutoff share needs the summed above-cutoff energy, which the
    # median alone cannot give; the block's own total bounds it and is what
    # the operator reads for "how much of the residual is up there at all"
    share = mx / total if total > 0 else 0.0
    return int(bin_), peakiness, share


def block_spectrum(side: torch.Tensor, kappa: int) -> tuple[int, float, float]:
    """(peak bin, peakiness, peak share of block energy), reading the device.

    The offline path and the tests use this; the serving path uses
    block_stats + read_stats so the read happens once per report instead of
    once per block.
    """
    st = block_stats(side, kappa)
    if float(st[2]) == 0.0 and float(st[1]) == 0.0:
        return -1, float("nan"), 0.0
    return read_stats(st)


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
            lid, {"n": 0, "pending": [], "bins": {}, "peak": 0.0, "share": 0.0}
        )

    def observe(self, side: torch.Tensor, block: int, *, lid: int) -> None:
        """Record every whole block of `side` ([N, D], N a multiple of block).

        Nothing is read back here: the per-block statistics stay on the device
        and are transferred in one go when a report is due, so a block close
        costs the arithmetic and not a stream stall.
        """
        r = self._layer(lid)
        for b in range(side.shape[0] // block):
            r["pending"].append(
                block_stats(side[b * block : (b + 1) * block], self.kappa)
            )
        if len(r["pending"]) >= self.every:
            self.dump(lid=lid)

    def dump(self, *, lid: int) -> None:
        r = self.rec.get(lid)
        if not r:
            return
        if r["pending"]:
            for row in torch.stack(r["pending"]).cpu():  # the one read
                bin_, peakiness, share = read_stats(row)
                if bin_ < 0:
                    continue
                r["n"] += 1
                r["bins"][bin_] = r["bins"].get(bin_, 0) + 1
                r["peak"] += peakiness
                r["share"] += share
            r["pending"] = []
        if not r["n"]:
            return
        top = sorted(r["bins"].items(), key=lambda kv: -kv[1])[:3]
        logger.info(
            "VKSPECTRUM layer=%d blocks=%d peak_bins=%s peakiness_mean=%.0f "
            "peak_share_mean=%.3f (telemetry only; a peak far above the cutoff "
            "means tier 1 is ranking a standing component)",
            lid,
            r["n"],
            ",".join(f"{b}:{c}" for b, c in top),
            r["peak"] / r["n"],
            r["share"] / r["n"],
        )
