"""Fused scan prologue: one kernel per (pair, head-block) replaces the pack's
eager chain (skept bmm, scale, mask, max, softmax, entropy, gate, qsk bmm,
qres residual, max1g select, and the qside/qsk transpose-casts -- ~12 launches).

The captured scan graph's cost is dominated by a fixed ~0.57 ms intercept of
small-kernel executions (VKSTATS, S-sweep); this kernel is that intercept's
removal. Online-softmax accumulation forked from
sglang/kernels/ops/attention/decode_attention.py (e_max / re-scale / e_sum):
one pass over kept rows keeps max1, the normalizer, and the entropy
accumulator in registers -- the [H, NK] score matrix is never materialized.

Entropy online: with running max m, s = sum(e^(x-m)), t = sum(e^(x-m) * x),
  H = lse - E[x] = (m + log s) - t / s.
Residual via Pythagoras (V rows orthonormal): ||q - qsk V||^2 = ||q||^2 -
||qsk||^2, so the back-projection GEMM disappears; the D-loop accumulates
qk chunks, qsk chunks, and ||q||^2 together.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.vestigekv import defaults as D


@triton.jit
def _fused_prologue_kernel(
    q_ptr,  # [P, H, 576] fp32 (gathered query, expanded form)
    kr_ptr,  # [P, NKm, 576] bf16 kept rows
    v_ptr,  # [P, R, 512] fp32 sketch basis
    nk_len_ptr,  # [P] int64
    thr_ptr,  # [P] fp32 gate threshold
    max1g_ptr,  # [P, H] fp32 out: max kept score, +inf closed gate, -inf empty
    qside_t_ptr,  # [P, 64, H] bf16 out
    qsk_t_ptr,  # [P, R, H] fp16 out
    qres_ptr,  # [P, H] fp32 out
    sc,  # attention scale
    NKm,
    H: tl.constexpr,
    R: tl.constexpr,
    KV: tl.constexpr,  # 512
    DD: tl.constexpr,  # 64 sidecar dims
    BLOCK_NK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    p = tl.program_id(0)
    h = tl.arange(0, H)
    r = tl.arange(0, R)
    nk = tl.load(nk_len_ptr + p)
    thr = tl.load(thr_ptr + p)

    # ---- D-loop: qsk = q[:KV] @ V^T and ||q[:KV]||^2, streamed over D ----
    qsk = tl.zeros([H, R], dtype=tl.float32)
    qnorm2 = tl.zeros([H], dtype=tl.float32)
    for d0 in range(0, KV, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        qc = tl.load(q_ptr + p * H * 576 + h[:, None] * 576 + d[None, :])
        vc = tl.load(v_ptr + p * R * KV + r[:, None] * KV + d[None, :])
        qsk += tl.dot(qc, tl.trans(vc), input_precision="ieee")
        qnorm2 += tl.sum(qc * qc, 1)
    qres2 = qnorm2 - tl.sum(qsk * qsk, 1)
    qres = tl.sqrt(tl.maximum(qres2, 0.0))

    # ---- NK-loop: online softmax over kept scores (decode_attention fork) ----
    e_max = tl.zeros([H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([H], dtype=tl.float32)
    x_sum = tl.zeros([H], dtype=tl.float32)  # sum e^(x-m) * x, for entropy
    for nk0 in range(0, NKm, BLOCK_NK):
        offs = nk0 + tl.arange(0, BLOCK_NK)
        mrow = offs < nk
        qk = tl.zeros([H, BLOCK_NK], dtype=tl.float32)
        for d0 in range(0, 576, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            qc = tl.load(q_ptr + p * H * 576 + h[:, None] * 576 + d[None, :])
            kc = tl.load(
                kr_ptr + p * NKm * 576 + offs[:, None] * 576 + d[None, :],
                mask=mrow[:, None],
                other=0.0,
            )
            qk += tl.dot(qc.to(tl.bfloat16), tl.trans(kc)).to(tl.float32)
        x = tl.where(mrow[None, :], qk * sc, -float("inf"))
        n_max = tl.maximum(tl.max(x, 1), e_max)
        alive = n_max > -float("inf")
        n_max_safe = tl.where(alive, n_max, 0.0)
        rescale = tl.where(alive, tl.exp(e_max - n_max_safe), 1.0)
        pexp = tl.exp(x - n_max_safe[:, None])
        pexp = tl.where(mrow[None, :], pexp, 0.0)
        e_sum = e_sum * rescale + tl.sum(pexp, 1)
        x_sum = x_sum * rescale + tl.sum(pexp * tl.where(mrow[None, :], x, 0.0), 1)
        e_max = n_max
    # entropy = (m + log s) - t/s; empty kept (s==0) -> gate open, max1g=-inf
    nonempty = e_sum > 0.0
    lse = e_max + tl.log(tl.where(nonempty, e_sum, 1.0))
    ent = lse - x_sum / tl.where(nonempty, e_sum, 1.0)
    gate = (ent > thr) | (~nonempty)
    max1 = tl.where(nonempty, e_max, -float("inf"))
    max1g = tl.where(gate, max1, float("inf"))
    tl.store(max1g_ptr + p * H + h, max1g)
    tl.store(qres_ptr + p * H + h, qres)

    # ---- transposed, storage-dtype query outputs for the scan kernel ----
    dd = tl.arange(0, DD)
    qside = tl.load(q_ptr + p * H * 576 + h[:, None] * 576 + (KV + dd)[None, :])
    tl.store(
        qside_t_ptr + p * DD * H + dd[:, None] * H + h[None, :],
        tl.trans(qside).to(tl.bfloat16),
    )
    tl.store(
        qsk_t_ptr + p * R * H + r[:, None] * H + h[None, :],
        tl.trans(qsk).to(tl.float16),
    )


def fused_prologue(q, kr, v, nk_len, thr, sc, out=None):
    """q [P,H,576] fp32, kr [P,NKm,576] bf16, v [P,R,512] fp32.
    Returns (max1g [P,H] fp32, qside_t [P,64,H] bf16, qsk_t [P,R,H] fp16,
    qres [P,H] fp32); pass `out` to reuse fixed-address buffers (capture)."""
    P, H, _ = q.shape
    NKm = kr.shape[1]
    R = v.shape[1]
    if out is None:
        max1g = q.new_empty(P, H)
        qside_t = q.new_empty(P, D.SIDECAR_DIM, H, dtype=torch.bfloat16)
        qsk_t = q.new_empty(P, R, H, dtype=torch.float16)
        qres = q.new_empty(P, H)
    else:
        max1g, qside_t, qsk_t, qres = out
    _fused_prologue_kernel[(P,)](
        q,
        kr,
        v,
        nk_len,
        thr,
        max1g,
        qside_t,
        qsk_t,
        qres,
        sc,
        NKm,
        H=H,
        R=R,
        KV=D.KV_LORA_RANK,
        DD=D.SIDECAR_DIM,
        BLOCK_NK=64,
        BLOCK_D=64,
        num_warps=4,
    )
    return max1g, qside_t, qsk_t, qres


# ---- split-NK variant: flash-decoding style parallelism over kept rows ----
# The single-kernel form launches one CTA per pair (P ~= 7): >90% of the GPU
# idles and the kept sweep runs at 140 GB/s. Splitting the NK loop across a
# 2D grid restores occupancy; partials (m, s, t) merge exactly (the online-
# softmax rescale identity), so the result matches the unsplit form up to
# fp32 accumulation order.


@triton.jit
def _prologue_scores_kernel(
    qbuf_ptr,  # [L, RR, H, 576] recall query stack, read in place (see qside)
    li_ptr,
    slot_ptr,
    RR,
    kr_ptr,
    nk_len_ptr,
    pm_ptr,
    ps_ptr,
    pt_ptr,  # [P, NSPLIT, H] partials
    sc,
    NKm,
    NSPLIT: tl.constexpr,
    H: tl.constexpr,
    BLOCK_NK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    p = tl.program_id(0)
    sp = tl.program_id(1)
    h = tl.arange(0, H)
    qb = qbuf_ptr + (tl.load(li_ptr + p) * RR + tl.load(slot_ptr + p)) * H * 576
    nk = tl.load(nk_len_ptr + p)
    span = (NKm + NSPLIT - 1) // NSPLIT
    lo = sp * span
    # Clamp by the pair's REAL kept count, not the capacity: rows past nk are
    # fully masked, but a masked row still burns its tl.dot FLOPs. A split
    # landing entirely past nk (or an empty pair) runs zero iterations and
    # still stores the correct empty partial (-inf, 0, 0) below, so the merge
    # (and the empty-kept fire-all contract) is untouched.
    hi = tl.minimum(lo + span, tl.minimum(NKm, nk))
    e_max = tl.zeros([H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([H], dtype=tl.float32)
    x_sum = tl.zeros([H], dtype=tl.float32)
    for nk0 in range(lo, hi, BLOCK_NK):
        offs = nk0 + tl.arange(0, BLOCK_NK)
        mrow = (offs < hi) & (offs < nk)
        qk = tl.zeros([H, BLOCK_NK], dtype=tl.float32)
        for d0 in range(0, 576, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            qc = tl.load(qb + h[:, None] * 576 + d[None, :])
            kc = tl.load(
                kr_ptr + p * NKm * 576 + offs[:, None] * 576 + d[None, :],
                mask=mrow[:, None],
                other=0.0,
            )
            qk += tl.dot(qc.to(tl.bfloat16), tl.trans(kc)).to(tl.float32)
        x = tl.where(mrow[None, :], qk * sc, -float("inf"))
        n_max = tl.maximum(tl.max(x, 1), e_max)
        alive = n_max > -float("inf")
        n_safe = tl.where(alive, n_max, 0.0)
        rescale = tl.where(alive, tl.exp(e_max - n_safe), 1.0)
        pexp = tl.where(mrow[None, :], tl.exp(x - n_safe[:, None]), 0.0)
        e_sum = e_sum * rescale + tl.sum(pexp, 1)
        x_sum = x_sum * rescale + tl.sum(pexp * tl.where(mrow[None, :], x, 0.0), 1)
        e_max = n_max
    base = p * NSPLIT * H + sp * H
    tl.store(pm_ptr + base + h, e_max)
    tl.store(ps_ptr + base + h, e_sum)
    tl.store(pt_ptr + base + h, x_sum)


@triton.jit
def _prologue_merge_kernel(
    pm_ptr,
    ps_ptr,
    pt_ptr,
    nk_len_ptr,
    thr_ptr,
    max1g_ptr,
    qbuf_ptr,  # qside/qsk/qres work folded in: same [P] grid as the merge,
    li_ptr,  # and the scan (its only consumer) runs strictly after.
    slot_ptr,
    RR,
    v_ptr,
    qside_t_ptr,
    qsk_t_ptr,
    qres_ptr,
    NSPLIT: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    KV: tl.constexpr,
    DD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    p = tl.program_id(0)
    h = tl.arange(0, H)
    r = tl.arange(0, R)
    qb = qbuf_ptr + (tl.load(li_ptr + p) * RR + tl.load(slot_ptr + p)) * H * 576
    qsk = tl.zeros([H, R], dtype=tl.float32)
    qnorm2 = tl.zeros([H], dtype=tl.float32)
    for d0 in range(0, KV, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        qc = tl.load(qb + h[:, None] * 576 + d[None, :]).to(tl.float32)
        vc = tl.load(v_ptr + p * R * KV + r[:, None] * KV + d[None, :])
        qsk += tl.dot(qc, tl.trans(vc), input_precision="ieee")
        qnorm2 += tl.sum(qc * qc, 1)
    tl.store(
        qres_ptr + p * H + h,
        tl.sqrt(tl.maximum(qnorm2 - tl.sum(qsk * qsk, 1), 0.0)),
    )
    dd = tl.arange(0, DD)
    qside = tl.load(qb + h[:, None] * 576 + (KV + dd)[None, :]).to(tl.float32)
    tl.store(
        qside_t_ptr + p * DD * H + dd[:, None] * H + h[None, :],
        tl.trans(qside).to(tl.bfloat16),
    )
    tl.store(
        qsk_t_ptr + p * R * H + r[:, None] * H + h[None, :],
        tl.trans(qsk).to(tl.float16),
    )
    sp = tl.arange(0, NSPLIT)
    base = p * NSPLIT * H
    m = tl.load(pm_ptr + base + sp[:, None] * H + h[None, :])
    s = tl.load(ps_ptr + base + sp[:, None] * H + h[None, :])
    t = tl.load(pt_ptr + base + sp[:, None] * H + h[None, :])
    gm = tl.max(m, 0)
    alive = gm > -float("inf")
    gm_safe = tl.where(alive, gm, 0.0)
    w = tl.exp(m - gm_safe[None, :])
    w = tl.where(m > -float("inf"), w, 0.0)
    gs = tl.sum(s * w, 0)
    gt = tl.sum(t * w, 0)
    nk = tl.load(nk_len_ptr + p)
    thr = tl.load(thr_ptr + p)
    nonempty = (gs > 0.0) & (nk > 0)
    lse = gm + tl.log(tl.where(nonempty, gs, 1.0))
    ent = lse - gt / tl.where(nonempty, gs, 1.0)
    gate = (ent > thr) | (~nonempty)
    max1 = tl.where(nonempty, gm, -float("inf"))
    tl.store(max1g_ptr + p * H + h, tl.where(gate, max1, float("inf")))


_NSPLIT = 32


def fused_prologue_split(qbuf, li, slot, kr, v, nk_len, thr, sc, out, partials):
    """Split-NK prologue reading queries in place from the stacked qbuf.

    qbuf [L, RR, H, 576] (bf16 in production; fp32 accepted -- loads upcast),
    li/slot [P] int64 select each pair's query row. Same outputs as
    fused_prologue; replaces the per-step torch gather+cast+contiguous."""
    P = li.shape[0]
    H = qbuf.shape[2]
    RR = qbuf.shape[1]
    NKm = kr.shape[1]
    R = v.shape[1]
    max1g, qside_t, qsk_t, qres = out
    pm, ps, pt = partials
    _prologue_scores_kernel[(P, _NSPLIT)](
        qbuf,
        li,
        slot,
        RR,
        kr,
        nk_len,
        pm,
        ps,
        pt,
        sc,
        NKm,
        NSPLIT=_NSPLIT,
        H=H,
        BLOCK_NK=64,
        BLOCK_D=64,
        num_warps=4,
    )
    _prologue_merge_kernel[(P,)](
        pm,
        ps,
        pt,
        nk_len,
        thr,
        max1g,
        qbuf,
        li,
        slot,
        RR,
        v,
        qside_t,
        qsk_t,
        qres,
        NSPLIT=_NSPLIT,
        H=H,
        R=R,
        KV=D.KV_LORA_RANK,
        DD=D.SIDECAR_DIM,
        BLOCK_D=64,
        num_warps=4,
    )
    return max1g, qside_t, qsk_t, qres


# ---- deterministic two-phase compaction: replaces the torch cumsum chain ----
# nsys (128k, node0): the int64 tensor_kernel_scan_innermost_dim alone cost
# 157 us/step -- 1.5x the scan kernel itself -- plus the scatter/where chain.
# Order-preserving two-phase compaction in int32: per-block fired counts,
# a tiny exclusive scan over blocks, then an ordered write of arch ids into
# fetch_buf. Deterministic by construction (same fired set -> same layout),
# preserving the overflow guarantee the atomic-append form would lose.


@triton.jit
def _compact_scan_kernel(counts_ptr, offsets_ptr, total_ptr, NB, NB2: tl.constexpr):
    p = tl.program_id(0)
    b = tl.arange(0, NB2)  # NB2 = next pow2 >= NB; masked beyond NB
    m = b < NB
    c = tl.load(counts_ptr + p * NB + b, mask=m, other=0)
    excl = tl.cumsum(c, 0) - c
    tl.store(offsets_ptr + p * NB + b, excl, mask=m)
    tl.store(total_ptr + p, tl.sum(c, 0))
    # Reset the fused-count buckets for the NEXT step's scan (the graph
    # replays this every step; zeroing here removes a standalone memset).
    tl.store(counts_ptr + p * NB + b, tl.zeros([NB2], dtype=tl.int32), mask=m)


@triton.jit
def _compact_write_kernel(
    hit_ptr,
    offsets_ptr,
    arch_ptr,
    out_ptr,
    out_len_ptr,
    li_ptr,
    slot_ptr,
    total_ptr,
    a_len_ptr,
    Am,
    W,
    NSLOT,
    BLOCK_A: tl.constexpr,
):
    p = tl.program_id(1)
    b = tl.program_id(0)
    offs = b * BLOCK_A + tl.arange(0, BLOCK_A)
    alen = tl.load(a_len_ptr + p)
    m = (offs < Am) & (offs < alen)
    h = tl.load(hit_ptr + p * Am + offs, mask=m, other=0) != 0
    nb = tl.num_programs(0)
    base = tl.load(offsets_ptr + p * nb + b)
    pos = base + tl.cumsum(h.to(tl.int32), 0) - 1
    li = tl.load(li_ptr + p)
    slot = tl.load(slot_ptr + p)
    arch = tl.load(arch_ptr + p * Am + offs, mask=m, other=0)
    ok = h & (pos < W)
    tl.store(out_ptr + li * NSLOT * W + slot * W + pos, arch, mask=ok)
    if b == 0:
        t = tl.load(total_ptr + p)
        tl.store(out_len_ptr + li * NSLOT + slot, tl.minimum(t, W))


def compact_fired(hit, arch, a_len, li, slot, fetch_buf, fetch_len, scratch):
    """Deterministic fired-row compaction. scratch: (counts, offsets, total)
    int32 [P, NB] x2 + [P]; fetch_buf [n_li, n_slot, W] int64-compatible."""
    P, Am = hit.shape
    BLOCK_A = 1024
    NB = triton.cdiv(Am, BLOCK_A)
    counts, offsets, total = scratch
    # counts arrive pre-filled by the scan kernel's fused per-block
    # accumulation (zeroed at alloc and re-zeroed by the prefix below).
    _compact_scan_kernel[(P,)](
        counts, offsets, total, NB, NB2=triton.next_power_of_2(NB)
    )
    W = fetch_buf.shape[-1]
    _compact_write_kernel[(NB, P)](
        hit,
        offsets,
        arch,
        fetch_buf,
        fetch_len,
        li,
        slot,
        total,
        a_len,
        Am,
        W,
        fetch_buf.shape[1],
        BLOCK_A=BLOCK_A,
    )
