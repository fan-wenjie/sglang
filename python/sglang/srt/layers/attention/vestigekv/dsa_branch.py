# SPDX-License-Identifier: Apache-2.0
"""The GLM branch rule's operators: threshold over the kept groups, fire mask,
compaction, and the fallback -- five kernels, no host-side glue.

Forked in shape from the Kimi line's scan (batched_step.py: per-bucket fused
counts feeding fused_prologue.compact_fired) rather than written from the torch
chain, because the torch chain was 85 kernels per layer-step and its cost was
the node count, not any one op. Node-level nsys of the chain version
(2026-10-03, 11 layers, one device): +498 us/step over DSA = 198 us of Triton
+ 137 us of torch glue + ~127 us of graph-node floor for +178 nodes.

The fallback and the recall are TWO ARMS OF ONE KERNEL, as in the MLA version
(_compact_write_kernel publishes out_len and out_ovf from the same program):
_branch_commit_kernel decides per lane whether the fire exceeded the trigger,
publishes (fetch_len, fetch_ovf) either way from the same store addresses, and
runs the row expansion only on the recall arm. Splitting the decision from the
act was what the chain version did -- the count in one kernel, the fence flag
in a second, the rows in a third -- and it cost two launches and a reread of
the same counter.

Per layer-step:

  _branch_kept_kernel   grid over kept ROWS: each kept row -> its group, from
                        the POSITION tier 1 recorded (no page-table inverse),
                        km[g] = 1, and the range of the kept groups' logits.
  _branch_hist_kernel   grid over groups: the kept groups' logits, binned.
  _branch_hit_kernel    grid over groups: the q-quantile off the histogram
                        (recomputed per CTA -- 256 loads and a scan, cheaper
                        than the launch a separate kernel costs), then
                        hit[g] = valid & ~km[g] & (logit[g] > thr) with the
                        bucket's count for the prefix. Resets km on its way out.
  _compact_scan_kernel  (fused_prologue, unchanged) exclusive bucket prefix.
  _branch_commit_kernel the two arms above, plus the compaction and the
                        expansion of each fired group to its POOL contiguous
                        rows through req_to_token. Resets the quantile scratch
                        for the next layer.

Nothing is zeroed from the host: km is reset by its last reader
(_branch_hit_kernel), the histogram and the range by _branch_commit_kernel for
the NEXT call, and the bucket counts by _compact_scan_kernel (its own rule).
branch_scratch seeds them once, outside any graph.

Upstream considered for the threshold, and why it is not called: DSA's own
`kpool_topk_transform.cuh` selects by exactly this algorithm -- a radix
histogram, a threshold bin, then a compact (`s_histogram`,
`s_threshold_bin_id`) -- and `dsa/topk_threshold_telemetry.py` already states
the design, that handing the selection a pivot turns it into "a count plus a
compact". It cannot be called here: its K is a compile-time constant
(`SGL_GROUP_TOPK`) over the whole candidate row, while this rule needs the
r-th largest of the KEPT subset with r = floor((1-q) n_kept) varying per lane
per step. So the histogram phase is re-expressed in Triton over a masked
subset with a runtime rank; the selection phase is not duplicated -- the
compaction is the Kimi line's `compact_fired`, reused.

The threshold is read off a histogram of the kept groups' logits, so it is the
q-quantile to within one bin (NBIN bins over the kept scores' range), always
rounded DOWN to the bin's lower edge: the rule fires at least what the exact
quantile would. The quantile is a knob (SGLANG_VESTIGEKV_BRANCH_Q), and a knob
known to one part in NBIN is still the knob.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.vestigekv import defaults as D

NBIN = 256


@triton.jit
def _branch_kept_kernel(
    keptpos_ptr,   # [NSLOT, CAPK] int32: request-relative position of each kept row
    klen_ptr,      # [NSLOT] int32
    slot_ptr,      # [B] int pool slot per lane
    logits_ptr,    # [B, G] fp32
    km_ptr,        # [B, G] int8 out (zero on entry): 1 = group holds a kept row
    range_ptr,     # [B, 2] fp32 in/out: (+inf, -inf) on entry, (lo, hi) after
    CAPK, G,
    POOL: tl.constexpr, BK: tl.constexpr,
):
    # Grid over the lane's kept ROWS, not one CTA per lane: a single CTA costs
    # 74.7 us/layer-step at 128k and grows with the kept set, which is what
    # makes the tax scale with context instead of staying flat.
    #
    # A kept row's group is its POSITION // POOL. Tier 1 records the position
    # when it writes the table (eviction.kept_rows produces it), so there is no
    # page-table inverse here: the physical row is not even read. The inverse
    # cost a torch scatter per step and one served crash (the page table is
    # sized by max_context_len and the inverse by the KV pool, and on this
    # geometry the pool is the smaller).
    b = tl.program_id(1)
    pid = tl.program_id(0)
    slot = tl.load(slot_ptr + b).to(tl.int64)
    n = tl.load(klen_ptr + slot)
    j = pid * BK + tl.arange(0, BK)
    mj = j < n
    pos = tl.load(keptpos_ptr + slot * CAPK + j, mask=mj, other=0).to(tl.int64)
    g = tl.minimum(pos // POOL, G - 1)
    tl.store(km_ptr + b * G + g, tl.full([BK], 1, tl.int8), mask=mj)
    # The range of the kept GROUPS' logits: the min and max over the rows equal
    # those over the groups the rows name, so one pass serves both.
    s = tl.load(logits_ptr + b * G + g, mask=mj, other=0.0)
    tl.atomic_min(range_ptr + b * 2, tl.min(tl.where(mj, s, float("inf")), 0))
    tl.atomic_max(range_ptr + b * 2 + 1, tl.max(tl.where(mj, s, float("-inf")), 0))


@triton.jit
def _branch_hist_kernel(
    logits_ptr, km_ptr, range_ptr,
    hist_ptr,      # [B, NB] int32 (zero on entry)
    G,
    BLOCK: tl.constexpr, NB: tl.constexpr,
):
    b = tl.program_id(1)
    blk = tl.program_id(0)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    m = offs < G
    km = tl.load(km_ptr + b * G + offs, mask=m, other=0) != 0
    s = tl.load(logits_ptr + b * G + offs, mask=km, other=0.0)
    lo = tl.load(range_ptr + b * 2)
    hi = tl.load(range_ptr + b * 2 + 1)
    width = tl.maximum(hi - lo, 1e-6)
    bin_ = tl.minimum(
        tl.maximum(((s - lo) / width * NB).to(tl.int32), 0), NB - 1
    )
    tl.atomic_add(hist_ptr + b * NB + bin_, 1, mask=km)


@triton.jit
def _branch_hit_kernel(
    logits_ptr, km_ptr, hist_ptr, range_ptr, plen_ptr,
    hit_ptr,       # [B, GP] int8 out
    counts_ptr,    # [B, NBK] int32 out, one per bucket
    thr_ptr,       # [B] fp32 out: the resolved threshold (observability/tests)
    q,
    G, GP,
    BLOCK: tl.constexpr, NB: tl.constexpr,
):
    b = tl.program_id(1)
    blk = tl.program_id(0)
    nb = GP // BLOCK
    # The threshold, per CTA, off the 256-bin histogram: the (r+1)-th largest
    # kept-group logit, r = floor((1-q) n), rounded DOWN to its bin's lower
    # edge. 256 loads and a scan per CTA is cheaper than the launch a separate
    # threshold kernel costs (11.9 us/layer-step as its own kernel).
    bb = tl.arange(0, NB)
    h = tl.load(hist_ptr + b * NB + bb)
    nkg = tl.sum(h, 0)
    r = tl.floor((1.0 - q) * nkg.to(tl.float32)).to(tl.int32)
    above = nkg - tl.cumsum(h, 0)  # entries in bins strictly above
    is_bin = (above <= r) & (above + h > r)
    sel = tl.max(tl.where(is_bin, bb, 0), 0)
    lo = tl.load(range_ptr + b * 2)
    hi = tl.load(range_ptr + b * 2 + 1)
    width = tl.maximum(hi - lo, 1e-6)
    # A lane with no kept group has no threshold to stand on: everything fires,
    # which overflows and fences the lane to DSA. That is the right degenerate
    # answer -- there is no resident set to rank against.
    thr = tl.where(nkg > 0, lo + sel.to(tl.float32) / NB * width, float("-inf"))
    if blk == 0:
        tl.store(thr_ptr + b, thr)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    m = offs < G
    plen = tl.load(plen_ptr + b)
    s = tl.load(logits_ptr + b * G + offs, mask=m, other=float("-inf"))
    km = tl.load(km_ptr + b * G + offs, mask=m, other=1)
    fired = m & (offs < plen) & (km == 0) & (s > thr)
    # hit covers the padded width so the compaction's tail buckets read zeros.
    tl.store(hit_ptr + b * GP + offs, fired.to(tl.int8), mask=offs < GP)
    tl.store(counts_ptr + b * nb + blk, tl.sum(fired.to(tl.int32), 0))
    # Reset km for the next (layer, step): this kernel is its last reader, so
    # the reset rides on a store that is already in flight instead of a memset.
    tl.store(km_ptr + b * G + offs, tl.zeros([BLOCK], dtype=tl.int8), mask=m)


@triton.jit
def _branch_commit_kernel(
    hit_ptr,       # [B, GP] int8 (hit kernel)
    offsets_ptr,   # [B, NBK] int32 exclusive bucket prefix (compact scan)
    total_ptr,     # [B] int32 fired group count (compact scan)
    slot_ptr,      # [B] int pool slot per lane
    r2t_ptr,       # [NSLOT, MAXCTX] int64
    fetch_ptr,     # [NSLOT, W] int32 out
    flen_ptr,      # [NSLOT] int32 out
    fovf_ptr,      # [NSLOT] int32 out
    ovf_count_ptr, # [1] int32 running fence tally
    hist_ptr,      # [B, NB] int32: reset here for the NEXT call
    range_ptr,     # [B, 2] fp32: reset here for the NEXT call
    G, GP, WG, WF, MAXCTX,
    POOL: tl.constexpr, BLOCK: tl.constexpr, NB: tl.constexpr,
):
    """The fallback and the recall, one kernel, two arms (see the module
    docstring). Fork of fused_prologue._compact_write_kernel: the arena
    indirection is gone (a group's id IS its index), and a compacted entry
    becomes POOL contiguous rows instead of one."""
    b = tl.program_id(1)
    pid = tl.program_id(0)
    NPROG = tl.num_programs(0)  # capped grid; live buckets stride over it
    nb = GP // BLOCK
    slot = tl.load(slot_ptr + b).to(tl.int64)
    ng = tl.load(total_ptr + b)
    fenced = ng > WF
    W = WG * POOL

    if pid == 0:
        # Both arms publish the same two words at the same addresses, so the
        # pack reads one shape and the lanes of a batch may disagree: the
        # fenced lane gets length 0 and the flag the pack fences on (the fork
        # then attends the indexer's own top-k and NOT the kept rows), the
        # recalling lane gets its row count. fetch_len is consumed every step
        # even by a lane that fires nothing, so this write cannot ride on
        # bucket 0 existing in the (capped) grid.
        tl.store(flen_ptr + slot, tl.where(fenced, 0, ng * POOL))
        tl.store(fovf_ptr + slot, fenced.to(tl.int32))
        if fenced:
            tl.atomic_add(ovf_count_ptr, 1)
        # The quantile scratch for the next (layer, step). Nothing reads it
        # after this kernel; resetting here costs no launch where a standalone
        # memset would cost one per layer. branch_scratch seeds the first call.
        bb = tl.arange(0, NB)
        tl.store(hist_ptr + b * NB + bb, tl.zeros([NB], dtype=tl.int32))
        tl.store(range_ptr + b * 2, float("inf"))
        tl.store(range_ptr + b * 2 + 1, float("-inf"))

    if not fenced:
        # The recall arm. The fenced arm does NONE of this -- the point of a
        # fallback is to delete work -- so the branch sits outside the loop
        # rather than predicating the loads off: both arms' address tensors
        # would otherwise stay live across the loop and the allocator spills
        # (.claude/rules/disassemble-check-for-spills.md). The fenced arm is
        # empty, so no loop body is duplicated.
        j = tl.arange(0, POOL)
        trips = (tl.maximum(nb - pid, 0) + NPROG - 1) // NPROG
        for i in range(trips):
            blk = pid + i * NPROG
            offs = blk * BLOCK + tl.arange(0, BLOCK)
            m = offs < G
            h = tl.load(hit_ptr + b * GP + offs, mask=m, other=0) != 0
            base = tl.load(offsets_ptr + b * nb + blk)
            pos = base + tl.cumsum(h.to(tl.int32), 0) - 1
            ok = h & (pos < WG)
            # a fired group's POOL token positions -> this request's physical
            # rows, landing at pos*POOL in the fetch buffer
            tpos = tl.minimum(offs[:, None] * POOL + j[None, :], MAXCTX - 1)
            row = tl.load(
                r2t_ptr + slot * MAXCTX + tpos, mask=ok[:, None], other=0
            )
            dst = pos[:, None] * POOL + j[None, :]
            tl.store(
                fetch_ptr + slot * W + dst, row.to(tl.int32), mask=ok[:, None]
            )


def branch_fire(
    *, logits, pool_lens, slots, kept_pos, klen, q, scratch, fetch_buf,
    fetch_len, fetch_ovf, r2t, pool, fence_groups,
):
    """One layer-step of the branch rule. `scratch` comes from branch_scratch;
    fetch_buf [NSLOT, W], fetch_len/fetch_ovf [NSLOT] are this layer's slices.
    `kept_pos` is tier 1's position table, parallel to its row table."""
    from sglang.srt.layers.attention.vestigekv.fused_prologue import (
        _compact_scan_kernel,
    )

    B, G = logits.shape
    NSLOT, CAPK = kept_pos.shape
    sc = scratch
    GP = sc["hit"].shape[1]
    NBK = GP // D.SCAN_BUCKET
    WG = fetch_buf.shape[1] // pool
    WF = WG if fence_groups is None else max(1, min(WG, int(fence_groups)))

    _branch_kept_kernel[(triton.cdiv(CAPK, 256), B)](
        kept_pos, klen, slots, logits, sc["km"], sc["range"],
        CAPK, G, POOL=pool, BK=256, num_warps=4,
    )
    _branch_hist_kernel[(NBK, B)](
        logits, sc["km"], sc["range"], sc["hist"], G,
        BLOCK=D.SCAN_BUCKET, NB=NBIN, num_warps=4,
    )
    _branch_hit_kernel[(NBK, B)](
        logits, sc["km"], sc["hist"], sc["range"], pool_lens,
        sc["hit"], sc["counts"], sc["thr"], float(q), G, GP,
        BLOCK=D.SCAN_BUCKET, NB=NBIN, num_warps=4,
    )
    _compact_scan_kernel[(B,)](
        sc["counts"], sc["offsets"], sc["total"], NBK,
        NB2=triton.next_power_of_2(NBK),
    )
    _branch_commit_kernel[(min(NBK, D.SCAN_GRID_CAP), B)](
        sc["hit"], sc["offsets"], sc["total"], slots, r2t,
        fetch_buf, fetch_len, fetch_ovf, sc["ovf_count"], sc["hist"], sc["range"],
        G, GP, WG, WF, r2t.shape[1],
        POOL=pool, BLOCK=D.SCAN_BUCKET, NB=NBIN, num_warps=4,
    )


def branch_scratch(*, bs, G, nslot, fetch_w, pool, fence_groups, dev):
    """Fixed-address scratch for one (batch width, logits width). The histogram,
    the range and the bucket counts are seeded here and thereafter reset by the
    kernels themselves, so a captured step allocates nothing and memsets
    nothing."""
    GP = triton.cdiv(G, D.SCAN_BUCKET) * D.SCAN_BUCKET
    NBK = GP // D.SCAN_BUCKET
    return {
        "km": torch.zeros(bs, G, dtype=torch.int8, device=dev),
        "hist": torch.zeros(bs, NBIN, dtype=torch.int32, device=dev),
        "range": torch.tensor(
            [[float("inf"), float("-inf")]] * bs, dtype=torch.float32, device=dev
        ),
        "thr": torch.zeros(bs, dtype=torch.float32, device=dev),
        "hit": torch.zeros(bs, GP, dtype=torch.int8, device=dev),
        "counts": torch.zeros(bs, NBK, dtype=torch.int32, device=dev),
        "offsets": torch.zeros(bs, NBK, dtype=torch.int32, device=dev),
        "total": torch.zeros(bs, dtype=torch.int32, device=dev),
        "ovf_count": torch.zeros(1, dtype=torch.int32, device=dev),
    }

@triton.jit
def _kept_append_kernel(
    slots_ptr,     # [bs] int64 lane -> request slot
    loc_ptr,       # [bs] this step's appended pool row per lane
    kept_buf_ptr,  # [L, R1, CAP] int32
    kept_len_ptr,  # [L, R1] int32
    bs,
    R1,
    CAP: tl.constexpr,
    L: tl.constexpr,
):
    """One decode step's new row, appended to every layer's kept table.

    Forked from `_pack_csr_prep_kernel` in the deleted vestigekv/pack_csr.py,
    keeping only its two append loops: that kernel did double duty, packing a
    CSR the DSA fork never reads AND performing this append, so deleting it
    took the append with it and the kept table stopped growing between closes
    (measured: kept 1120 -> 1005 per call at 32k, fetched 555 -> 371).

    Two passes, both reading only PRE-STEP state, because duplicate slots
    (shared trash lanes) must mirror the torch scatter: every colliding lane
    appends at the same n_old (last write wins) and the length advances by one
    exactly once. A single mutate-as-you-walk loop would let a later duplicate
    read the earlier lane's update. The SHARDED arm of the original is gone --
    a sharded sequence is refused on this branch (_refuse_sharded_sequence).
    """
    li = tl.program_id(0)
    if li < L:
        for i in range(0, bs):  # appends; kept_len still pristine
            slot = tl.load(slots_ptr + i).to(tl.int64)
            n_old = tl.load(kept_len_ptr + li * R1 + slot).to(tl.int64)
            tl.store(kept_buf_ptr + (li * R1 + slot) * CAP + n_old,
                     tl.load(loc_ptr + i))
        for i in range(0, bs):  # lengths: first occurrence only
            slot = tl.load(slots_ptr + i).to(tl.int64)
            dup = 0
            for jj in range(0, i):
                dup += (tl.load(slots_ptr + jj).to(tl.int64) == slot).to(tl.int32)
            if dup == 0:
                n_old = tl.load(kept_len_ptr + li * R1 + slot)
                tl.store(kept_len_ptr + li * R1 + slot, n_old + 1)


def kept_append(*, slots, loc, kept_stack, klen_stack):
    """`kept_stack` [L, R1, CAP], `klen_stack` [L, R1]: every layer in one launch."""
    L, R1, CAP = kept_stack.shape
    _kept_append_kernel[(L,)](
        slots, loc, kept_stack, klen_stack,
        slots.shape[0], R1, CAP=CAP, L=L, num_warps=1,
    )
