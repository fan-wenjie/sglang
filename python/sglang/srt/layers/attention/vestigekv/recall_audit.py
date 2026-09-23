# SPDX-License-Identifier: Apache-2.0
"""Telemetry: the recall the certificate ACHIEVES, not the one it targets.

VKSTATS reports overflow, fetch percentiles and the fence rate. All three are
cost: they say how much the certificate fired and what that cost, and none of
them says whether it fired on the rows that mattered. The quality side --
achieved tier-2 recall on the live decode query -- has had no served
instrument at all, and the two are not interchangeable.

The failure that makes the gap matter: zp is refitted at every build against
the calibration queries, so it keeps hitting its target ON THOSE. If the live
query drifts away from them the fit still succeeds, the fire count barely
moves, the fence rate stays flat, and the recall actually delivered falls.
Every number the serving path already collects would look unchanged.

So this scores the archive densely with the step's own query, takes the rows
whose true score beats the tier-1 threshold, and asks what fraction of them
the certificate fired on. That dense pass is exactly the work the sparse path
exists to avoid, which is why SGLANG_DEBUG_VESTIGEKV_RECALL_AUDIT is 0 in
production and nothing downstream reads what this collects.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def step_recall(
    arch: torch.Tensor, q: torch.Tensor, thr: torch.Tensor, fired: torch.Tensor
) -> torch.Tensor:
    """[3] on the archive's device: achieved recall, the true count, the fired
    count.

    `arch` is [A, D] archived content, `q` the decode query [H, D], `thr` the
    per-head tier-1 threshold the scan compares against, `fired` the boolean
    the certificate produced. A row is TRUE when any head's exact score beats
    that head's threshold -- the same any-head rule the scan fires on, so the
    two sets are comparable and a disagreement is the certificate's, not the
    reduction's.

    Recall is 1.0 when nothing was above threshold: the certificate cannot
    miss what does not exist, and reporting 0 there would drag the mean down
    on exactly the steps that were easiest.
    """
    true = ((arch @ q.T) > thr[None, :]).any(dim=1)
    n_true = true.sum()
    hit = (true & fired).sum()
    rec = torch.where(
        n_true > 0, hit.float() / n_true.clamp_min(1).float(), torch.ones((), device=arch.device)
    )
    return torch.stack([rec, n_true.float(), fired.sum().float()])


class RecallAudit:
    """Per-layer running record of achieved recall against the target."""

    def __init__(self, target: float, every: int = 32):
        self.target, self.every = target, max(1, every)
        self.rec: dict[int, dict] = {}

    def _layer(self, lid: int) -> dict:
        return self.rec.setdefault(lid, {"pending": [], "base": 0})

    def observe(self, *, arch, q, thr, fired, lid: int) -> None:
        if arch.ndim != 2 or arch.shape[0] == 0:
            return
        r = self._layer(lid)
        r["pending"].append(step_recall(arch, q, thr, fired))
        if len(r["pending"]) >= self.every:
            self.dump(lid=lid)

    def dump(self, *, lid: int) -> None:
        r = self.rec.get(lid)
        if not r or not r["pending"]:
            return
        got = torch.stack(r["pending"]).cpu()  # the one read
        n = got.shape[0]
        base, r["base"], r["pending"] = r["base"], r["base"] + n, []
        rec = got[:, 0]
        logger.info(
            "VKRECALL layer=%d base=%d n=%d achieved_mean=%.4f achieved_min=%.4f "
            "below_target=%.3f target=%.2f true_mean=%.1f fired_mean=%.1f "
            "(achieved recall on the live query; VKSTATS reports only cost)",
            lid,
            base,
            n,
            float(rec.mean()),
            float(rec.min()),
            float((rec < self.target).float().mean()),
            self.target,
            float(got[:, 1].mean()),
            float(got[:, 2].mean()),
        )
