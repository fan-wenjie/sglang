# SPDX-License-Identifier: Apache-2.0
"""The GLM branch rule's operators: threshold over the kept groups, fire mask,
compaction. Forked in shape from the Kimi line's scan (batched_step.py:
per-bucket fused counts into compact_fired) rather than written from the
torch chain, because the torch chain was 85 kernels per layer-step and its
cost was the node count, not any one op (profile 2026-10-03: 136 us of GPU
time behind 283 us of graph replay per layer).

Five launches per layer-step plus compact_fired's two (and three torch nodes):

  _branch_kept_kernel   one program per lane: every kept row -> its group
                        (through a page-table inverse built by one scatter),
                        km[g] = 1, and the range of the kept groups' logits.
  _branch_hist_kernel   grid over group buckets: the kept groups' logits binned.
  _branch_thr_kernel    one program per lane: the q-quantile off the histogram.
  _branch_hit_kernel    one program per (bucket, lane): hit[g] = valid &
                        ~km[g] & (logit[g] > thr), with the bucket's count
                        stored for compact_fired's prefix.
  compact_fired         (fused_prologue.py) group ids -> a prefix, fetch_len
                        in groups, the overflow flag.
  _branch_rows_kernel   one program per lane: fired groups x pool -> request
                        positions -> physical rows through req_to_token, into
                        the fetch buffer; fetch_len = 0 and fetch_ovf = 1 on
                        an overflowed lane (the pack fences it to DSA).

The threshold is read off a histogram of the kept groups' logits, so it is
the q-quantile to within one bin (NBIN bins over the kept scores' range),
always rounded DOWN (the bin's lower edge): the rule fires at least what the
exact quantile would. The quantile is a knob (SGLANG_VESTIGEKV_BRANCH_Q), and
a knob known to one part in NBIN is still the knob.
"""
import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.vestigekv import defaults as D

NBIN = 256


@triton.jit
def _branch_kept_kernel(
    kept_ptr,      # [NSLOT, CAPK] int32 physical rows
    klen_ptr,      # [NSLOT] int32
    slot_ptr,      # [B] int32 pool slot per lane
    inv_ptr,       # [B, NP] int64: physical page -> request-relative page (torch scatter)
    logits_ptr,    # [B, G] fp32
    km_ptr,        # [B, G] int8 out (pre-zeroed): 1 = group holds a kept row
    range_ptr,     # [B, 2] fp32 out: (lo, hi) of the kept groups' logits
    CAPK, NP, G,
    PAGE: tl.constexpr, POOL: tl.constexpr, BK: tl.constexpr,
):
    # One program per lane over its kept ROWS (<= capK, ~1-4k live): each row
    # -> its group, km[g] = 1, and the range of the groups' logits (the min and
    # max over rows equal those over the groups the rows name).
    b = tl.program_id(0)
    slot = tl.load(slot_ptr + b)
    n = tl.load(klen_ptr + slot)
    jj = tl.arange(0, BK)
    lo = float("inf")
    hi = float("-inf")
    for j0 in range(0, n, BK):
        j = j0 + jj
        mj = j < n
        row = tl.load(kept_ptr + slot * CAPK + j, mask=mj, other=0).to(tl.int64)
        rel = tl.load(inv_ptr + b * NP + tl.minimum(row // PAGE, NP - 1), mask=mj, other=0)
        g = tl.minimum((rel * PAGE + row % PAGE) // POOL, G - 1)
        tl.store(km_ptr + b * G + g, tl.full([BK], 1, tl.int8), mask=mj)
        s = tl.load(logits_ptr + b * G + g, mask=mj, other=0.0)
        lo = tl.minimum(lo, tl.min(tl.where(mj, s, float("inf")), 0))
        hi = tl.maximum(hi, tl.max(tl.where(mj, s, float("-inf")), 0))
    tl.store(range_ptr + b * 2, lo)
    tl.store(range_ptr + b * 2 + 1, hi)


@triton.jit
def _branch_hist_kernel(
    logits_ptr, km_ptr, range_ptr,
    hist_ptr,      # [B, NB] int32 (zero on entry)
    G,
    BLOCK: tl.constexpr, NB: tl.constexpr,
):
    # One program per (bucket, lane): the kept groups in this bucket, binned.
    bk = tl.program_id(0)
    b = tl.program_id(1)
    offs = bk * BLOCK + tl.arange(0, BLOCK)
    m = offs < G
    km = tl.load(km_ptr + b * G + offs, mask=m, other=0) != 0
    s = tl.load(logits_ptr + b * G + offs, mask=km, other=0.0)
    lo = tl.load(range_ptr + b * 2)
    hi = tl.load(range_ptr + b * 2 + 1)
    width = tl.maximum(hi - lo, 1e-6)
    bin_ = tl.minimum(((s - lo) / width * NB).to(tl.int32), NB - 1)
    tl.atomic_add(hist_ptr + b * NB + bin_, 1, mask=km)


@triton.jit
def _branch_thr_kernel(
    hist_ptr, range_ptr, thr_ptr, q,
    NB: tl.constexpr,
):
    # One program per lane: the (r+1)-th largest kept-group logit, r =
    # floor((1-q) n), read off the histogram and rounded DOWN to its bin's
    # lower edge; the histogram is reset for the next step.
    b = tl.program_id(0)
    bb = tl.arange(0, NB)
    h = tl.load(hist_ptr + b * NB + bb)
    n = tl.sum(h, 0)
    r = tl.floor((1.0 - q) * n.to(tl.float32)).to(tl.int32)
    above = n - tl.cumsum(h, 0)  # entries in bins strictly above
    is_bin = (above <= r) & (above + h > r)
    sel = tl.max(tl.where(is_bin, bb, 0), 0)
    lo = tl.load(range_ptr + b * 2)
    hi = tl.load(range_ptr + b * 2 + 1)
    width = tl.maximum(hi - lo, 1e-6)
    tl.store(thr_ptr + b, lo + sel.to(tl.float32) / NB * width)
    tl.store(hist_ptr + b * NB + bb, tl.zeros([NB], dtype=tl.int32))


@triton.jit
def _branch_hit_kernel(
    logits_ptr, km_ptr, thr_ptr, plen_ptr,
    hit_ptr,       # [B, GP] int8 out
    counts_ptr,    # [B, NBK] int32 out (per bucket)
    G, GP,
    BLOCK: tl.constexpr,
):
    bk = tl.program_id(0)
    b = tl.program_id(1)
    offs = bk * BLOCK + tl.arange(0, BLOCK)
    m = offs < G
    plen = tl.load(plen_ptr + b)
    thr = tl.load(thr_ptr + b)
    s = tl.load(logits_ptr + b * G + offs, mask=m, other=float("-inf"))
    km = tl.load(km_ptr + b * G + offs, mask=m, other=1)
    fired = m & (offs < plen) & (km == 0) & (s > thr)
    tl.store(hit_ptr + b * GP + offs, fired.to(tl.int8), mask=offs < GP)
    nbk = (GP + BLOCK - 1) // BLOCK
    tl.store(counts_ptr + b * nbk + bk, tl.sum(fired.to(tl.int32), 0))


@triton.jit
def _branch_rows_kernel(
    gbuf_ptr,      # [NSLOT, WF] int32 compacted group ids (compact_fired out, li=0)
    glen_ptr,      # [NSLOT] int32 fired groups (<= WF)
    govf_ptr,      # [NSLOT] int32 overflow flag
    slot_ptr,      # [B] int32
    r2t_ptr,       # [NSLOT, MAXCTX] int64
    fetch_ptr,     # [NSLOT, W] int32 out (this layer's slice)
    flen_ptr,      # [NSLOT] int32 out
    fovf_ptr,      # [NSLOT] int32 out
    WF, W, MAXCTX,
    POOL: tl.constexpr, BW: tl.constexpr,
):
    b = tl.program_id(0)
    slot = tl.load(slot_ptr + b)
    ng = tl.load(glen_ptr + slot)
    ovf = tl.load(govf_ptr + slot)
    k = tl.arange(0, BW)  # fetch row index
    for k0 in range(0, W, BW):
        kk = k0 + k
        gi = kk // POOL
        mk = (kk < W) & (gi < ng)
        g = tl.load(gbuf_ptr + slot * WF + tl.minimum(gi, WF - 1), mask=mk, other=0).to(tl.int64)
        pos = tl.minimum(g * POOL + (kk % POOL), MAXCTX - 1)
        row = tl.load(r2t_ptr + slot.to(tl.int64) * MAXCTX + pos, mask=mk, other=0)
        tl.store(fetch_ptr + slot * W + kk, row.to(tl.int32), mask=kk < W)
    n_rows = tl.where(ovf > 0, 0, ng * POOL)
    tl.store(flen_ptr + slot, n_rows)
    tl.store(fovf_ptr + slot, ovf)


def branch_fire(
    *, logits, pool_lens, page_table, slots, kept, klen, q, scratch, fetch_buf,
    fetch_len, fetch_ovf, r2t, pool, page_size, fence_groups,
):
    """One layer-step of the branch rule. scratch: dict from branch_scratch().
    fetch_buf [NSLOT, W], fetch_len/fetch_ovf [NSLOT] are this layer's slices."""
    from sglang.srt.layers.attention.vestigekv.fused_prologue import compact_fired

    B, G = logits.shape
    NSLOT, CAPK = kept.shape
    P = page_table.shape[1]
    sc = scratch
    GP = sc["hit"].shape[1]
    sc["km"].zero_()
    sc["slot"].copy_(slots.to(torch.int32))
    inv = sc["inv"]
    inv.zero_()
    # Scatter in REVERSE column order so that, where a physical page id
    # repeats, the LOWEST request-relative index wins: the page table is
    # padded (with zeros) past the request's pages, and a plain scatter let
    # the padding overwrite physical page 0's true index whenever page 0 was
    # in use (caught by test_vestigekv_dsa_branch, not by the ad-hoc check).
    pt = page_table[:B].to(torch.int64).clamp_(min=0, max=inv.shape[1] - 1).flip(1)
    inv.scatter_(1, pt, torch.arange(P - 1, -1, -1, device=inv.device)[None, :].expand(B, -1))
    _branch_kept_kernel[(B,)](
        kept, klen, sc["slot"], inv, logits, sc["km"], sc["range"],
        CAPK, inv.shape[1], G, PAGE=page_size, POOL=pool, BK=256, num_warps=4,
    )
    nbk = GP // D.SCAN_BUCKET
    _branch_hist_kernel[(nbk, B)](
        logits, sc["km"], sc["range"], sc["hist"], G, BLOCK=D.SCAN_BUCKET, NB=NBIN, num_warps=4,
    )
    _branch_thr_kernel[(B,)](sc["hist"], sc["range"], sc["thr"], float(q), NB=NBIN, num_warps=1)
    nbk = GP // D.SCAN_BUCKET
    _branch_hit_kernel[(nbk, B)](
        logits, sc["km"], sc["thr"], pool_lens.to(torch.int32), sc["hit"], sc["counts"],
        G, GP, BLOCK=D.SCAN_BUCKET, num_warps=4,
    )
    compact_fired(
        sc["hit"].view(-1), sc["ids"].view(-1), sc["a_len"], sc["a_off"], sc["li"], sc["slot"],
        sc["gbuf"], sc["glen"], sc["govf"], sc["ovf_count"],
        (sc["counts"], sc["offsets"], sc["total"]), GP,
    )
    W = fetch_buf.shape[1]
    _branch_rows_kernel[(B,)](
        sc["gbuf"][0], sc["glen"][0], sc["govf"][0], sc["slot"], r2t,
        fetch_buf, fetch_len, fetch_ovf,
        sc["gbuf"].shape[-1], W, r2t.shape[1], POOL=pool, BW=512, num_warps=4,
    )


def branch_scratch(*, bs, G, nslot, fetch_w, pool, fence_groups, dev, n_pages):
    """Fixed-address scratch for one (batch width, logits width)."""
    WG = fetch_w // pool
    WF = WG if fence_groups is None else max(1, min(WG, int(fence_groups)))
    GP = (G + D.SCAN_BUCKET - 1) // D.SCAN_BUCKET * D.SCAN_BUCKET
    NBK = GP // D.SCAN_BUCKET
    return {
        "inv": torch.zeros(bs, n_pages, dtype=torch.int64, device=dev),
        "km": torch.zeros(bs, G, dtype=torch.int8, device=dev),
        "hist": torch.zeros(bs, NBIN, dtype=torch.int32, device=dev),
        "thr": torch.zeros(bs, dtype=torch.float32, device=dev),
        "range": torch.zeros(bs, 2, dtype=torch.float32, device=dev),
        "slot": torch.zeros(bs, dtype=torch.int32, device=dev),
        "hit": torch.zeros(bs, GP, dtype=torch.int8, device=dev),
        "ids": torch.arange(GP, dtype=torch.int32, device=dev).repeat(bs, 1),
        "counts": torch.zeros(bs, NBK, dtype=torch.int32, device=dev),
        "offsets": torch.zeros(bs, NBK, dtype=torch.int32, device=dev),
        "total": torch.zeros(bs, dtype=torch.int32, device=dev),
        "a_len": torch.full((bs,), G, dtype=torch.int32, device=dev),
        "a_off": (torch.arange(bs, device=dev) * GP).to(torch.int64),
        "li": torch.zeros(bs, dtype=torch.int32, device=dev),
        "gbuf": torch.zeros(1, nslot, WF, dtype=torch.int32, device=dev),
        "glen": torch.zeros(1, nslot, dtype=torch.int32, device=dev),
        "govf": torch.zeros(1, nslot, dtype=torch.int32, device=dev),
        "ovf_count": torch.zeros(1, dtype=torch.int32, device=dev),
    }
