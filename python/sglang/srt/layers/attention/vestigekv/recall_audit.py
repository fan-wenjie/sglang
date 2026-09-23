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

WHERE IT RUNS, AND WHY NOT ON THE PRODUCTION PATH YET. The per-step work is
all device ops into a device ring under a device counter, so it would capture
and replay; the host reads the ring once per report. But the production scan
is the batched in-graph kernel, and its pairs and tiers are decided at block
close while the graph is captured at init on a dummy batch when no tier
exists -- a Python loop over tiers at capture time sees none and captures
nothing. A first version hung off that loop and produced zero reports. Making
it capture-correct means addressing every pair's rows through the pack's
device tables, and the one thing that is a Python object per pair, the layer's
KV buffer, cannot be indexed by a device layer id; it would take a Triton
kernel reading through the pack's kbase pointers. Not built.

So the hook sits in RecallTier.query_fixed, the per-tier scan the eager path
runs (SGLANG_ENABLE_VESTIGEKV_INGRAPH_SCAN=0 with CUDA graphs off), and it is
only as meaningful as that path's certificate. On first use that path fired
the whole archive on every step (fired ~31110 of ~31500 at 32k against ~2000
truly above threshold), so its recall of 0.9999 measured nothing; the
threshold itself was finite, which points the blame at the bound's inflation
term rather than the gate.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def step_recall(
    arch: torch.Tensor, q: torch.Tensor, thr: torch.Tensor, fired: torch.Tensor,
    valid: torch.Tensor = None,
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
    if valid is not None:
        # a captured caller reads a fixed-width arena window per pair, and
        # the rows past that pair's live length are whatever the arena held
        # last; they must count as neither true nor fired
        true = true & valid
        fired = fired & valid
    n_true = true.sum()
    hit = (true & fired).sum()
    rec = torch.where(
        n_true > 0, hit.float() / n_true.clamp_min(1).float(), torch.ones((), device=arch.device)
    )
    return torch.stack([rec, n_true.float(), fired.sum().float()])


class RecallAudit:
    """Per-tier record of achieved recall, kept on the device between reports."""

    def __init__(self, target: float, every: int = 32):
        self.target, self.every = target, max(1, every)
        self.ring = None  # [every, 3] fp32, lazily on the archive's device
        self.count = None  # [] int64 device counter: steps observed so far
        self.reported = 0  # host: steps already folded into a report

    def observe(self, *, arch, q, thr, fired, lid: int = 0, valid=None) -> None:
        if arch.ndim != 2 or arch.shape[0] == 0:
            return
        if self.ring is None:
            self.ring = torch.zeros(self.every, 3, device=arch.device)
            self.count = torch.zeros((), dtype=torch.int64, device=arch.device)
        # every op below is a device op on fixed addresses: it captures once
        # and replays with the step, which is the whole point
        slot = self.count % self.every
        self.ring.index_copy_(0, slot.view(1), step_recall(arch, q, thr, fired, valid).view(1, 3))
        self.count += 1

    def dump(self, *, lid: int) -> None:
        """Host side, outside the graph: fold whatever the ring holds since the
        last report into one log line. The one device read is here."""
        if self.ring is None:
            return
        n_total = int(self.count)  # the one sync
        n_new = min(n_total - self.reported, self.every)
        if n_new <= 0:
            return
        got = self.ring.cpu()
        # the ring holds the last `every` steps in slot order; take the newest
        end = n_total % self.every
        idx = [(end - n_new + i) % self.every for i in range(n_new)]
        got = got[idx]
        self.reported = n_total
        rec = got[:, 0]
        logger.info(
            "VKRECALL layer=%d base=%d n=%d achieved_mean=%.4f achieved_min=%.4f "
            "below_target=%.3f target=%.2f true_mean=%.1f fired_mean=%.1f "
            "(achieved recall on the live query; VKSTATS reports only cost)",
            lid,
            n_total - n_new,
            n_new,
            float(rec.mean()),
            float(rec.min()),
            float((rec < self.target).float().mean()),
            self.target,
            float(got[:, 1].mean()),
            float(got[:, 2].mean()),
        )
