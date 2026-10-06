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
# Coarse levels the capacity cap may be set at. The fired count has to be bounded
# by the fetch buffer's width, and the bound must be known BEFORE the compaction's
# offsets are built -- so the hit kernel tallies its fires per coarse level and the
# scan picks a level, instead of a later pass re-counting them. 16 levels cost
# 16*4 B per 256-group block each way, 0.5 B/group, against the 9 B/group a
# candidate-range pass plus a candidate histogram cost.
NCOARSE = 16


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


def _fence_truncate():
    """Whether a would-be-overflowing lane TRUNCATES its own fired set to the fetch
    buffer's width instead of being handed DSA's selection.

    On by default. The fence saved no bytes -- the buffer holds WF*pool rows and
    DSA's index_topk is the same 2048, so it swapped one 2048-row budget for
    another -- and measured worse, because the fallback drops the kept rows (0.3566
    mass against 0.3635, worst head 0.1079 against 0.1143, on fenced steps). What it
    bought was a dependency: the fenced lane is the only consumer of DSA's top-k on
    this arm, and that top-k cannot be skipped speculatively because the indexer
    runs before attention inside a layer's forward.

    Off restores the fallback, which is the control that prices it.
    """
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_VESTIGEKV_FENCE_TRUNCATE.get())


def _coarse_proxy():
    """(C, keep_per_mille) for the coarse-filter proxy, or (0, 0) when unset.

    SGLANG_VESTIGEKV_COARSE_PROXY is "C,f" with f a fraction, e.g. "16,0.5".
    Measurement only; see the block it gates in _branch_hit_kernel.
    """
    from sglang.srt.environ import envs

    v = envs.SGLANG_VESTIGEKV_COARSE_PROXY.get()
    if not v:
        return 0, 0
    c, f = v.split(",")
    return int(c), int(round(float(f) * 1000))


@triton.jit
def _branch_hit_kernel(
    logits_ptr, km_ptr, hist_ptr, range_ptr, plen_ptr,
    hit_ptr,       # [B, GP] int8 out: 0 = no fire, else the fire's level + 1
    counts_ptr,    # [B, NBK, NC] int32 out, fires per bucket per coarse level
    thr_ptr,       # [B] fp32 out: the resolved threshold (observability/tests)
    q,
    G, GP,
    BLOCK: tl.constexpr, NB: tl.constexpr, NC: tl.constexpr,
    CPX_C: tl.constexpr = 0, CPX_KEEP: tl.constexpr = 0,
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
    if CPX_C > 0:
        # COARSE-FILTER PROXY (SGLANG_VESTIGEKV_COARSE_PROXY), a measurement
        # knob, never a serving mode. A two-level scan would pool CPX_C fine
        # groups into one coarse key and read the fine index only for the
        # surviving coarse groups; this drops the same SHAPE of candidates --
        # whole runs of CPX_C -- without building the coarse index, so the
        # question "does losing that fraction of the fired set cost quality"
        # can be answered before any of it is implemented.
        #
        # Deliberately CONSERVATIVE: the survivors here are a hash of the
        # coarse index, while the real filter picks them by score. Whatever
        # quality holds under random clustered dropping holds under the real
        # one, so a pass here is evidence for the design and a failure closes
        # it without the implementation cost.
        cg = offs // CPX_C
        # int64 before the multiply: offs is int32 and the Knuth constant
        # does not fit it (Triton: "Scalar 2654435761 is out of range for
        # type int32"), which failed at compile time, i.e. at server start.
        keep = ((cg.to(tl.int64) * 2654435761) % 1000) < CPX_KEEP
        fired = fired & keep
    # The fire's COARSE LEVEL, carried in the byte `hit` already occupies (0 means
    # no fire, so a level is stored as level+1). The capacity cap is applied later,
    # by the scan and the commit, against this level -- which is why it costs no
    # traffic: both of those already read this byte.
    #
    # Binned over the kept range EXTENDED upward by one width. `range` spans the
    # kept groups' logits and a candidate beating every kept group is exactly what
    # fires, so the kept range alone would clamp the whole fired set into one
    # level. Doubling it covers the fired tail without a pass to measure it;
    # whatever still clamps into the top level is handled by the fence, which stays
    # as the backstop.
    lvl = tl.minimum(
        tl.maximum(((s - lo) / (2.0 * width) * NC).to(tl.int32), 0), NC - 1
    )
    # hit covers the padded width so the compaction's tail buckets read zeros.
    tl.store(
        hit_ptr + b * GP + offs,
        tl.where(fired, lvl + 1, 0).to(tl.int8),
        mask=offs < GP,
    )
    for l in tl.static_range(NC):
        tl.store(
            counts_ptr + b * (nb * NC) + blk * NC + l,
            tl.sum((fired & (lvl == l)).to(tl.int32), 0),
        )
    # Reset km for the next (layer, step): this kernel is its last reader, so
    # the reset rides on a store that is already in flight instead of a memset.
    tl.store(km_ptr + b * G + offs, tl.zeros([BLOCK], dtype=tl.int8), mask=m)


@triton.jit
def _branch_cap_scan_kernel(
    counts_ptr,    # [B, NBK, NC] int32: fires per bucket per coarse level
    offsets_ptr,   # [B, NBK] int32 out: exclusive prefix of the FILTERED counts
    total_ptr,     # [B] int32 out: filtered fired count
    cap_ptr,       # [B] int32 out: levels at or below this one are dropped
    WF,
    NBK,
    NB2: tl.constexpr, NC: tl.constexpr,
):
    """Replaces the sketch chain's _compact_scan_kernel on the branch path.

    One launch does three things that used to need three: pick the capacity level,
    filter the per-bucket counts by it, and prefix-sum the result. Doing it here is
    what makes the cap free -- a later pass to re-count the survivors would be
    1 B/group and two more launches a layer-step, and the break-even on the whole
    change is only ~3.3 us per layer launch.

    The cap is the LOWEST level whose strictly-above population already fits WF, so
    dropping that level and everything under it leaves at most WF fires. -1 keeps
    every level, which is the ordinary non-overflowing step.
    """
    b = tl.program_id(0)
    r = tl.arange(0, NB2)
    c = tl.arange(0, NC)
    m = r < NBK
    cnt = tl.load(
        counts_ptr + b * (NBK * NC) + r[:, None] * NC + c[None, :],
        mask=m[:, None],
        other=0,
    )
    per_lvl = tl.sum(cnt, 0)                       # [NC]
    tot = tl.sum(per_lvl, 0)
    above = tot - tl.cumsum(per_lvl, 0)            # population strictly above level l
    fits = above <= WF
    cap = tl.where(tot <= WF, -1, tl.min(tl.where(fits, c, NC), 0))
    tl.store(cap_ptr + b, cap)
    keep = c > cap
    fc = tl.sum(tl.where(keep[None, :], cnt, 0), 1)        # [NB2] filtered per bucket
    tl.store(total_ptr + b, tl.sum(tl.where(m, fc, 0), 0))
    tl.store(offsets_ptr + b * NBK + r, tl.cumsum(fc, 0) - fc, mask=m)


@triton.jit
def _branch_commit_kernel(
    hit_ptr,       # [B, GP] int8 (hit kernel): 0, else the fire's level + 1
    offsets_ptr,   # [B, NBK] int32 exclusive bucket prefix (cap scan)
    total_ptr,     # [B] int32 fired group count, already filtered (cap scan)
    cap_ptr,       # [B] int32 the cap scan's level; levels <= it are dropped
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
    cap = tl.load(cap_ptr + b)
    # ng is already capped, so this only trips when even the TOP coarse level
    # overpopulates WF -- a lane whose fired scores all clamp into it. The fence
    # stays as that backstop rather than as the ordinary overflow path.
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
            # The cap, applied in the read the compaction already makes: a byte
            # of 0 gives level -1, which fails this for any cap >= -1.
            h = (tl.load(hit_ptr + b * GP + offs, mask=m, other=0).to(tl.int32) - 1) > cap
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
    _cpx = _coarse_proxy()
    _branch_hit_kernel[(NBK, B)](
        logits, sc["km"], sc["hist"], sc["range"], pool_lens,
        sc["hit"], sc["counts"], sc["thr"], float(q), G, GP,
        BLOCK=D.SCAN_BUCKET, NB=NBIN, NC=NCOARSE, num_warps=4,
        CPX_C=_cpx[0], CPX_KEEP=_cpx[1],
    )
    # WF only caps when truncation is on; passing the full width leaves `cap` at
    # -1 on every step, so the control arm reaches the fence exactly as before.
    _branch_cap_scan_kernel[(B,)](
        sc["counts"], sc["offsets"], sc["total"], sc["cap"],
        WF if _fence_truncate() else (1 << 30), NBK,
        NB2=triton.next_power_of_2(NBK), NC=NCOARSE,
    )
    _branch_commit_kernel[(min(NBK, D.SCAN_GRID_CAP), B)](
        sc["hit"], sc["offsets"], sc["total"], sc["cap"], slots, r2t,
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
        # Per BUCKET per COARSE LEVEL, so the scan can filter before it prefixes.
        # 16*4 B a 256-group bucket each way is 0.5 B/group, against the 9 B/group
        # a candidate-range pass plus a candidate histogram would have cost.
        "counts": torch.zeros(bs, NBK * NCOARSE, dtype=torch.int32, device=dev),
        "offsets": torch.zeros(bs, NBK, dtype=torch.int32, device=dev),
        "total": torch.zeros(bs, dtype=torch.int32, device=dev),
        "cap": torch.full((bs,), -1, dtype=torch.int32, device=dev),
        "ovf_count": torch.zeros(1, dtype=torch.int32, device=dev),
    }
