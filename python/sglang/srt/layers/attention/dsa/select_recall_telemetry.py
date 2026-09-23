# SPDX-License-Identifier: Apache-2.0
"""Telemetry: how much of what attention actually wants does DSA's top-k get?

A parasitic VestigeKV that only ever ADDS rows to DSA's selection can raise
quality and cannot lower it, but the size of that headroom is set by what DSA
already captures. If its top-2048 holds essentially everything the decode
query scores highly, there is nothing for a supplement to add; if it misses a
real share, a small certified supplement has somewhere to go.

Both numbers are reported because they answer different questions. The COUNT
overlap says how many of the oracle rows DSA holds; the MASS says how much of
the oracle's score it holds, and missing the 2000th row is not the same as
missing the 1st.

This is a study flag: computing the oracle is dense scoring over the whole
sequence, which is exactly the work the sparse path exists to avoid, so
SGLANG_DEBUG_DSA_SELECT_RECALL is 0 in production and nothing downstream reads
what this collects.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def step_stats(
    rows: torch.Tensor, q: torch.Tensor, selected: torch.Tensor, topk: int
) -> torch.Tensor:
    """[3] on the rows' device: oracle count overlap, oracle mass overlap, and
    the oracle size actually used.

    `rows` is [S, D] latent KV for one request, `q` the absorbed decode query
    [H, D], `selected` the row ordinals DSA chose. A row's score is its best
    over heads: a row any head wants is a row the layer wants.

    Stays on the device. The one host read happens per report, the lesson the
    spectrum telemetry paid 195 us per block to learn.
    """
    score = (rows @ q.T).amax(dim=1)
    k = min(topk, score.shape[0])
    top = score.topk(k)
    hit = torch.zeros(score.shape[0], dtype=torch.bool, device=score.device)
    hit[selected] = True
    got = hit[top.indices]
    # scores are pre-softmax logits and can be negative; shift by the oracle's
    # own floor so the mass ratio stays in [0, 1] and is not dominated by sign
    w = top.values - top.values[-1]
    total = w.sum()
    return torch.stack(
        [
            got.float().mean(),
            (w * got).sum() / total if total > 0 else got.float().mean(),
            torch.tensor(float(k), device=score.device),
        ]
    )


class SelectRecallTelemetry:
    """Per-layer running record of DSA's selection against the oracle."""

    def __init__(self, topk: int, every: int = 32):
        self.topk, self.every = topk, max(1, every)
        self.rec: dict[int, dict] = {}

    def _layer(self, lid: int) -> dict:
        return self.rec.setdefault(lid, {"pending": [], "base": 0})

    def observe(
        self,
        *,
        rows: torch.Tensor,
        q: torch.Tensor,
        selected: torch.Tensor,
        lid: int,
    ) -> None:
        if rows.ndim != 2 or rows.shape[0] <= self.topk:
            # a sequence no longer than the budget has nothing to select
            return
        r = self._layer(lid)
        r["pending"].append(step_stats(rows, q, selected, self.topk))
        if len(r["pending"]) >= self.every:
            self.dump(lid=lid)

    def dump(self, *, lid: int) -> None:
        r = self.rec.get(lid)
        if not r or not r["pending"]:
            return
        got = torch.stack(r["pending"]).cpu()  # the one read
        n = got.shape[0]
        base, r["base"], r["pending"] = r["base"], r["base"] + n, []
        logger.info(
            "VKSELREC layer=%d base=%d n=%d count_mean=%.4f count_min=%.4f "
            "mass_mean=%.4f mass_min=%.4f oracle=%d "
            "(headroom for a supplement is 1 - count_mean)",
            lid,
            base,
            n,
            float(got[:, 0].mean()),
            float(got[:, 0].min()),
            float(got[:, 1].mean()),
            float(got[:, 1].min()),
            int(got[0, 2]),
        )
