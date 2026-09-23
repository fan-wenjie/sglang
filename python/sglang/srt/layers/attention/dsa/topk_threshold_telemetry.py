# SPDX-License-Identifier: Apache-2.0
"""Telemetry: how stable the indexer's top-k cut is across steps and layers.

The indexer's top-k is the fastest-growing decode term (69.3 us at 4k to 118.8
us at 32k, +71% for 8x the context), and the only way to make it cheaper
without changing what it selects is to hand it a pivot: a value close to the
k-th largest logit, so the selection becomes a count plus a compact instead of
a full selection over S/pool candidates.

A pivot needs the THRESHOLD, not the selected set. Those are different
quantities and they can behave differently -- tier 1's kept set agrees only
0.296 between adjacent DSA layers on prose (0.465 on RULER), while the cut
value itself may still be smooth. This module records the cut so that question
can be answered from data instead of assumed.

It records nothing in production: SGLANG_DEBUG_DSA_TOPK_THRESHOLD is 0, and
nothing downstream reads what it collects.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def row_stats(row: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """[4] on `row`'s device: the k-th largest logit, the largest, the k/2-th,
    and the live candidate count.

    `valid` is a 0-d tensor and is applied as a device-side mask: converting it
    to a Python int here would sync the stream once per layer per step, which
    is the mistake the spectrum telemetry paid 195 us per block for before its
    read was deferred to one per report.

    The k/2-th value stands in for a median. A median over the padded row is
    -inf whenever padding exceeds half the row, and masking it out exactly
    would cost a second pass; the k/2-th comes free from the same selection and
    says more about the tail the pivot has to land in.
    """
    idx = torch.arange(row.shape[0], device=row.device)
    masked = torch.where(idx < valid, row.float(), row.new_full((), float("-inf")))
    top = torch.topk(masked, k, largest=True, sorted=True).values
    return torch.stack([top[-1], top[0], top[k // 2], valid.float()])


class TopkThresholdTelemetry:
    """Per-layer record of the indexer's top-k cut, one sample per decode step.

    Samples row 0 only: the question is how the cut moves over steps and
    layers, and one lane answers it without making the telemetry itself a
    per-batch cost. Device-resident until a report is due, then one transfer.
    """

    def __init__(self, every: int = 64):
        self.every = max(1, every)
        self.rec: dict[int, dict] = {}

    def _layer(self, lid: int) -> dict:
        return self.rec.setdefault(lid, {"pending": [], "base": 0})

    def observe(
        self, logits: torch.Tensor, valid: torch.Tensor, k: int, *, lid: int
    ) -> None:
        """Record the cut for one decode row.

        `valid` is the row's live candidate count as a device tensor; entries
        past it are padding and are masked out rather than trusted to be -inf.
        A row with fewer than k live candidates has no cut -- the selection
        takes everything -- and reaches the log as -inf, which the offline
        study drops. Filtering it here would need the count on the host.
        """
        if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < k:
            return
        r = self._layer(lid)
        r["pending"].append(row_stats(logits[0], valid, k))
        if len(r["pending"]) >= self.every:
            self.dump(lid=lid)

    def dump(self, *, lid: int) -> None:
        r = self.rec.get(lid)
        if not r or not r["pending"]:
            return
        rows = torch.stack(r["pending"]).cpu()  # the one read
        n = rows.shape[0]
        base, r["base"], r["pending"] = r["base"], r["base"] + n, []
        fmt = lambda col: ",".join(f"{float(v):.5g}" for v in rows[:, col])
        logger.info(
            "VKTOPKTH layer=%d base=%d n=%d kth=[%s] max=[%s] halfk=[%s] valid=[%s]",
            lid,
            base,
            n,
            fmt(0),
            fmt(1),
            fmt(2),
            fmt(3),
        )
