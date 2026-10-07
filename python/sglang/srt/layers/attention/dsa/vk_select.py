"""A reference-set threshold in place of DSA's rank top-k.

DSA picks the bin holding the K-th largest group score. That is a RANK
threshold: it cannot be known until every score has been histogrammed, a
histogram needs one block, and so kpool_topk_transform runs ONE CTA over every
group -- 127 kB through a single SM at 124k, measured 14.6 us/layer-step.

This rule takes the threshold from a REFERENCE SET instead: the q-quantile of
the groups the PREVIOUS layer-step selected. That is known before any score is
read, so the comparison has no cross-group dependency and the pass takes the
whole device. Measured 6.4 us/layer-step at the served fire rate, 2.1x, and it
holds through bs=8.

Why the previous selection is a sound reference set: it was the top-K by
construction, so it is a high-scoring sample, and while the selection itself
turns over ~75% per step the score DISTRIBUTION is stable -- the same property
that lets the fired-set band use the previous step's maximum. What is stale is
the set, not the quantile.

Bootstrap: with no previous selection the threshold is -inf, every group becomes
a candidate, the staging area spills, and the spill path runs DSA's own radix
over the full row. The first step of a request is therefore exactly DSA.
"""

import torch
import triton
import triton.language as tl

NBIN = 256


@triton.jit
def _prev_threshold_kernel(
    sel_ptr,       # [B, SG] int32: groups the previous step selected
    seln_ptr,      # [B] int32: how many
    logits_ptr,    # [B, G] fp32: THIS step's scores
    thr_ptr,       # [B] fp32 out
    q,
    G,
    SG: tl.constexpr, NB: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.load(seln_ptr + b)
    j = tl.arange(0, SG)
    m = (j < n) & (n > 0)
    g = tl.load(sel_ptr + b * SG + j, mask=m, other=0)
    ok = m & (g >= 0) & (g < G)
    s = tl.load(logits_ptr + b * G + g, mask=ok, other=0.0)
    lo = tl.min(tl.where(ok, s, float("inf")), 0)
    hi = tl.max(tl.where(ok, s, float("-inf")), 0)
    cnt = tl.sum(ok.to(tl.int32), 0)
    width = tl.maximum(hi - lo, 1e-6)
    bin_ = tl.minimum(tl.maximum(((s - lo) / width * NB).to(tl.int32), 0), NB - 1)
    h = tl.histogram(tl.where(ok, bin_, 0), NB)
    nz = tl.sum((~ok).to(tl.int32), 0)
    bb = tl.arange(0, NB)
    h = h - tl.where(bb == 0, nz, 0)
    r = tl.floor((1.0 - q) * cnt.to(tl.float32)).to(tl.int32)
    above = cnt - tl.cumsum(h, 0)
    is_bin = (above <= r) & (above + h > r)
    sel = tl.max(tl.where(is_bin, bb, 0), 0)
    # No previous selection -> -inf, which fires everything, spills the staging
    # area and routes the step to DSA's own radix. That is the bootstrap, and it
    # is exact rather than an approximation of DSA.
    thr = tl.where(cnt > 0, lo + sel.to(tl.float32) / NB * width, float("-inf"))
    tl.store(thr_ptr + b, thr)


def prev_threshold(*, logits, sel_groups, sel_n, thr, q):
    """q-quantile of the previous selection's scores under THIS step's query.

    NOT on the hot path any more. The emit kernel writes the next threshold
    itself from scores it already holds; doing it here costs a scattered gather
    of up to group_budget elements out of the global score row, every
    layer-step, and reading those scores more than once a step is the trap on
    this path. Kept because it expresses the rule for a general q, which the
    offline q-sweep that found the ratchet needs.
    """
    B, G = logits.shape
    SG = sel_groups.shape[1]
    assert SG == triton.next_power_of_2(SG), f"sel width {SG} must be a power of 2"
    _prev_threshold_kernel[(B,)](
        sel_groups, sel_n, logits, thr, float(q), G, SG=SG, NB=NBIN, num_warps=4,
    )
    return thr


def select_state(*, bs, group_budget, dev, cand_max=2048):
    """Fixed-address scratch. Nothing is allocated inside a captured step."""
    return {
        "cand_score": torch.zeros(bs, cand_max, dtype=torch.float32, device=dev),
        "cand_idx": torch.zeros(bs, cand_max, dtype=torch.int32, device=dev),
        "ncand": torch.zeros(bs, dtype=torch.int32, device=dev),
        # -inf, not zero: the first step must fire everything and route itself
        # through DSA's radix. Zero would silently fire "everything above 0",
        # which is a different and wrong rule.
        "thr": torch.full((bs,), float("-inf"), dtype=torch.float32, device=dev),
        # The reference set for the NEXT step, and its count. Zeroed here so the
        # first step of a fresh server bootstraps through DSA's radix.
        "sel_groups": torch.full((bs, group_budget), -1, dtype=torch.int32, device=dev),
        "sel_n": torch.zeros(bs, dtype=torch.int32, device=dev),
        "ovf": torch.zeros(bs, dtype=torch.int32, device=dev),
    }
