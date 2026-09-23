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


def far_region_stats(
    rows: torch.Tensor,
    q: torch.Tensor,
    selected: torch.Tensor,
    sigma: torch.Tensor,
    n_far: int,
    topk: int,
) -> torch.Tensor:
    """[4]: DSA's and tier 1's recall of the oracle in the far region, the
    oracle's far size, and the shared budget.

    The comparison is confined to rows older than the close block, and
    budget-matched, because neither of those is optional. Both selectors force
    the tail -- DSA by index_kpool_always_select_tail, tier 1 by keeping close
    blocks whole -- so the whole-sequence number mostly measures an agreement
    that was never in question. And tier 1 keeps far more rows than DSA's
    fixed budget, so an unmatched comparison rewards it for spending more.

    Tier 1 keeps HIGH sigma, so its picks are the top of the score.
    """
    score = (rows @ q.T).amax(dim=1)
    k = min(topk, score.shape[0])
    oracle = score.topk(k).indices
    far_oracle = oracle[oracle < n_far]
    sel_far = selected[selected < n_far]
    budget = min(int(sel_far.numel()), n_far)
    if far_oracle.numel() == 0 or budget == 0:
        return torch.zeros(14, device=score.device)
    hit_dsa = torch.zeros(n_far, dtype=torch.bool, device=score.device)
    hit_dsa[sel_far] = True
    # Tier 1's picks in ITS OWN rank order, so a prefix of them is what a
    # budget-Delta supplement would actually attend.
    order = sigma[:n_far].topk(budget).indices
    hit_dsa_o = hit_dsa[far_oracle]
    # oracle mass, shifted by the oracle's floor so the ratio stays in [0, 1]
    w = score[far_oracle]
    w = w - w.min()
    tot_w = w.sum().clamp_min(1e-9)
    in_oracle = torch.zeros(n_far, dtype=torch.bool, device=score.device)
    in_oracle[far_oracle] = True
    # rank of each oracle row inside tier 1's order, or budget if absent
    rank = torch.full((n_far,), budget, dtype=torch.long, device=score.device)
    rank[order] = torch.arange(budget, device=score.device)
    r_o = rank[far_oracle]
    out = [hit_dsa_o.float().mean()]
    for delta in (64, 128, 256, 512, budget):
        got = hit_dsa_o | (r_o < delta)
        out.append(got.float().mean())
        out.append((w * got).sum() / tot_w)
    out.append((w * hit_dsa_o).sum() / tot_w)
    out.append(torch.tensor(float(far_oracle.numel()), device=score.device))
    out.append(torch.tensor(float(budget), device=score.device))
    return torch.stack(out)


class SelectRecallTelemetry:
    """Per-layer running record of DSA's selection against the oracle."""

    def __init__(self, topk: int, every: int = 32):
        self.topk, self.every = topk, max(1, every)
        self.rec: dict[int, dict] = {}

    def _layer(self, lid: int) -> dict:
        return self.rec.setdefault(lid, {"pending": [], "far": [], "base": 0})

    def observe(
        self,
        *,
        rows: torch.Tensor,
        q: torch.Tensor,
        selected: torch.Tensor,
        lid: int,
        sigma: torch.Tensor = None,
        n_far: int = 0,
    ) -> None:
        if rows.ndim != 2 or rows.shape[0] <= self.topk:
            # a sequence no longer than the budget has nothing to select
            return
        r = self._layer(lid)
        r["pending"].append(step_stats(rows, q, selected, self.topk))
        if sigma is not None and n_far > 0:
            r["far"].append(
                far_region_stats(rows, q, selected, sigma, n_far, self.topk)
            )
        if len(r["pending"]) >= self.every:
            self.dump(lid=lid)

    def dump(self, *, lid: int) -> None:
        r = self.rec.get(lid)
        if not r or not r["pending"]:
            return
        got = torch.stack(r["pending"]).cpu()  # the one read
        n = got.shape[0]
        base, r["base"], r["pending"] = r["base"], r["base"] + n, []
        if r["far"]:
            f = torch.stack(r["far"]).cpu()
            r["far"] = []
            logger.info(
                "VKSELFAR layer=%d base=%d n=%d dsa=%.4f dsa_mass=%.4f "
                "d64=%.4f m64=%.4f d128=%.4f m128=%.4f d256=%.4f m256=%.4f "
                "d512=%.4f m512=%.4f dfull=%.4f mfull=%.4f "
                "oracle_far=%.0f budget=%.0f (union of DSA with tier 1's top-Delta; "
                "d=count, m=oracle mass)",
                lid, base, f.shape[0],
                float(f[:, 0].mean()), float(f[:, 11].mean()),
                *[float(f[:, i].mean()) for i in range(1, 11)],
                float(f[:, 12].mean()), float(f[:, 13].mean()),
            )
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
