"""D1: the threshold, computed from the kept rows alone.

Replaces _branch_kept_kernel + _branch_hist_kernel + the threshold half of
_branch_hit_kernel with one kernel, one CTA per lane, and no sweep over the
candidate groups at all.

Why no sweep is needed. The quantile is over the DISTINCT kept GROUPS -- several
kept rows can name one group, and the old histogram deduplicated them by
masking a pass over all G groups on `km`. But eviction.kept_rows emits the
positions SORTED, so g = pos // POOL is non-decreasing and a group boundary is
exactly g[j] != g[j-1]. One pass over the <=1156 kept rows therefore sees each
group once, which is all the histogram ever needed. At 124k that is a sweep over
32768 groups removed outright.

Semantics are deliberately unchanged from _branch_hit_kernel: NBIN bins over the
kept groups' range, the q-quantile's bin, threshold at that bin's LOWER edge, so
the rule fires at least what the exact quantile would. tl.sort exists in this
Triton and would give the quantile exactly, but that is a different rule and
needs its own quality draw; D1 is a cost change and nothing else.
"""

import torch
import triton
import triton.language as tl

NBIN = 256


@triton.jit
def _kept_threshold_kernel(
    keptpos_ptr,   # [NSLOT, CAPK] int32: request-relative position of each kept row
    klen_ptr,      # [NSLOT] int32
    slot_ptr,      # [B] pool slot per lane
    logits_ptr,    # [B, G] fp32
    km_ptr,        # [B, G] int8 out (zero on entry): 1 = group holds a kept row
    kg_ptr,        # [B, KG] int32 out: the distinct kept groups, for the seed
    kgn_ptr,       # [B] int32 out: how many of them
    thr_ptr,       # [B] fp32 out
    q,
    CAPK, G,
    POOL: tl.constexpr, BK: tl.constexpr, NB: tl.constexpr, KG: tl.constexpr,
):
    b = tl.program_id(0)
    slot = tl.load(slot_ptr + b).to(tl.int64)
    n = tl.load(klen_ptr + slot)
    j = tl.arange(0, BK)
    mj = j < n
    pos = tl.load(keptpos_ptr + slot * CAPK + j, mask=mj, other=0).to(tl.int64)
    g = tl.minimum(pos // POOL, G - 1)
    tl.store(km_ptr + b * G + g, tl.full([BK], 1, tl.int8), mask=mj)

    # First-of-group, off the sorted order. j == 0 is always first; otherwise
    # compare with the previous row's group. The guard is `mj`, so a lane past
    # the kept length is never first.
    prev = tl.load(keptpos_ptr + slot * CAPK + j - 1, mask=mj & (j > 0), other=-1)
    gp = tl.minimum(prev.to(tl.int64) // POOL, G - 1)
    first = mj & ((j == 0) | (prev < 0) | (g != gp))

    s = tl.load(logits_ptr + b * G + g, mask=first, other=0.0)
    lo = tl.min(tl.where(first, s, float("inf")), 0)
    hi = tl.max(tl.where(first, s, float("-inf")), 0)
    nkg = tl.sum(first.to(tl.int32), 0)
    width = tl.maximum(hi - lo, 1e-6)

    # Compact the distinct groups for the kernel's seed list, in order.
    slots_ = tl.cumsum(first.to(tl.int32), 0) - 1
    tl.store(kg_ptr + b * KG + slots_, g.to(tl.int32), mask=first & (slots_ < KG))
    tl.store(kgn_ptr + b, tl.minimum(nkg, KG))

    bin_ = tl.minimum(
        tl.maximum(((s - lo) / width * NB).to(tl.int32), 0), NB - 1
    )
    h = tl.histogram(tl.where(first, bin_, 0), NB)
    # Lane 0's bin collects the masked-off entries; take them back out.
    nz = tl.sum((~first).to(tl.int32), 0)
    bb = tl.arange(0, NB)
    h = h - tl.where(bb == 0, nz, 0)

    r = tl.floor((1.0 - q) * nkg.to(tl.float32)).to(tl.int32)
    above = nkg - tl.cumsum(h, 0)
    is_bin = (above <= r) & (above + h > r)
    sel = tl.max(tl.where(is_bin, bb, 0), 0)
    # No kept group means no resident set to rank against: everything fires,
    # which overflows and hands the lane to DSA's exact top-k. That is the right
    # degenerate answer, and it is the same one the old kernel gave.
    thr = tl.where(nkg > 0, lo + sel.to(tl.float32) / NB * width, float("-inf"))
    tl.store(thr_ptr + b, thr)


def kept_threshold(*, logits, kept_pos, klen, slots, q, pool, scratch, kg_max):
    """Per-lane threshold, km mask and distinct-kept-group seed list."""
    B, G = logits.shape
    CAPK = kept_pos.shape[1]
    BK = triton.next_power_of_2(max(1, int(kg_max) * pool))
    assert BK <= 4096, f"kept bound {kg_max} groups needs BK={BK}, one CTA's limit"
    _kept_threshold_kernel[(B,)](
        kept_pos, klen, slots, logits,
        scratch["km"], scratch["kg"], scratch["kgn"], scratch["thr"],
        float(q), CAPK, G,
        POOL=pool, BK=BK, NB=NBIN, KG=scratch["kg"].shape[1], num_warps=8,
    )
    return scratch["thr"]


def select_scratch(*, bs, G, kg_max, dev, cand_max=2048):
    return {
        # Candidates staged in GLOBAL memory for the wide collect pass, which
        # cannot use shared memory across CTAs. kCandMax in the .cuh must match.
        "cand_score": torch.zeros(bs, cand_max, dtype=torch.float32, device=dev),
        "cand_idx": torch.zeros(bs, cand_max, dtype=torch.int32, device=dev),
        "ncand": torch.zeros(bs, dtype=torch.int32, device=dev),
        "km": torch.zeros(bs, G, dtype=torch.int8, device=dev),
        "kg": torch.zeros(bs, int(kg_max), dtype=torch.int32, device=dev),
        "kgn": torch.zeros(bs, dtype=torch.int32, device=dev),
        "thr": torch.zeros(bs, dtype=torch.float32, device=dev),
        "ovf": torch.zeros(bs, dtype=torch.int32, device=dev),
    }
