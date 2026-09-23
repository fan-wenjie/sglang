# SPDX-License-Identifier: Apache-2.0
"""Pool the tier-2 archive at DSA's own ratio, the parity rule for anything
sized by the sequence.

DSA stores one index entry per token and scans it 4:1 pooled, so it reads 33 B
per token of context per layer. The archive stored one entry per token and
scanned every one of them, 132 B per token, which is where VestigeKV's decode
slope above DSA came from: 0.241 ms per 100k tokens against DSA's 0.105.
Pooling at the same ratio, over the same groups, puts the two on the same
footing.

Measured off the calibration dumps at ~10.6k, 11 DSA layers, recall of the
dense top-2048 by a live decode query:

    unpooled, attend 2048   0.738 prose / 0.679 RULER   scan 1x
    pooled,   attend 2048   0.509        / 0.545        scan 1/4
    pooled,   attend 4096   0.755        / 0.756        scan 1/4

so the pooled archive at twice the attended budget BEATS the unpooled one at
a quarter of the scan traffic. Pooling loses per-row precision and the freed
bandwidth buys back more than it costs, which is why the budget rises with the
ratio rather than staying put.

The rule is the plain mean. Weighting by the stored row norm was measured too
(prose 0.5126 against 0.5090, RULER 0.5168 against 0.5454 -- better on one
corpus, worse on the other) and a max-norm delegate is clearly worse (0.479 /
0.429). DSA's own pooling is a softmax gate over a per-channel score the model
was trained with; nothing in the stored archive reconstructs that, so an
unweighted mean is both the simplest rule and as good as any available.
"""

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D


def pool_operands(
    csk: torch.Tensor, rho: torch.Tensor, pool_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool [T, r] scores and [T] norms into [G, r], [G], [G].

    The third return is the pooling SPREAD, max_i ||c_i - m|| over the group.
    A group is scanned by its mean, so a row whose score sits far above the
    mean would be skipped silently; |c_i . q - m . q| <= ||c_i - m|| ||q||
    makes that gap bounded rather than hoped-for, and the scan inflates the
    pooled score by it exactly as it already inflates by the rank-truncation
    residual. Without this term the certificate is unsound the moment the
    archive is pooled -- it would still be calibrated, and it would still be
    wrong.

    The mean is exact under the projection: csk is content @ V.T, which is
    linear in content, so pooling the projections equals projecting the pooled
    content. That identity is why the archive can be pooled after the fact
    without rebuilding it from the latent rows.

    A trailing partial group is dropped rather than averaged over fewer rows:
    the tail is inside the close block, which is attended whole and never
    reaches the archive scan, and a short group would carry a different
    denominator into a score the certificate compares against a threshold.
    """
    if pool_size <= 1:
        return csk, rho, torch.zeros_like(rho)
    g = csk.shape[0] // pool_size
    if g == 0:
        return csk[:0], rho[:0], rho[:0]
    n = g * pool_size
    grp = csk[:n].view(g, pool_size, csk.shape[1]).float()
    mean = grp.mean(dim=1)
    spread = (grp - mean.unsqueeze(1)).norm(dim=2).amax(dim=1)
    # the norm pools by max: it bounds the group's largest row, which is what a
    # conservative certificate needs, while the mean would understate it
    return (
        mean.to(csk.dtype),
        rho[:n].view(g, pool_size).amax(dim=1),
        spread.to(rho.dtype),
    )


def expand_groups(group_ids: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Group ids -> the row ids they cover, DSA's grouping: group g is rows
    g*pool_size .. g*pool_size+pool_size-1.

    Same shape as dsa/kpool_fp8_index.expand_pooled_groups_to_topk, kept
    separate because that one also carries a page table and a tail rule this
    path has no use for.
    """
    if pool_size <= 1:
        return group_ids
    off = torch.arange(pool_size, device=group_ids.device, dtype=group_ids.dtype)
    return (group_ids[:, None] * pool_size + off).reshape(-1)


def pooled_capacity(capacity: int, pool_size: int) -> int:
    """Rows to attend once the archive is pooled.

    The measurement above is the whole argument: at the unpooled budget the
    pooled archive recalls less, and at pool_size//2 times it recalls more,
    for a quarter of the scan. Scaling the budget is part of the change, not a
    separate tuning knob.
    """
    if pool_size <= 1:
        return capacity
    return capacity * max(1, pool_size // 2)


def archive_bytes_per_token(rank: int, pool_size: int) -> float:
    """Scan traffic the archive costs per token of context per layer.

    fp16 scores, one fp32 norm and, when pooled, one fp32 spread per entry,
    spread over the tokens the entry covers. At rank 64, pool 4 this is 34
    B/token against DSA's 33 for its pooled index cache -- the point of the
    parity rule, with the soundness term the pooling itself makes necessary.
    """
    if pool_size <= 1:
        return float(rank * 2 + 4)
    return (rank * 2 + 4 + 4) / pool_size
