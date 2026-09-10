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

import logging

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.attention.vestigekv import defaults as D

logger = logging.getLogger(__name__)


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
    kr_ptr,  # FROM_POOL=0: [P, NKm, 576] bf16 kept rows (else unused)
    kslot_ptr,  # FROM_POOL=1: [P, NKm] int32 pool row ids (else unused)
    kbase_ptr,  # FROM_POOL=1: [P] int64 pool base for the pair's layer
    nk_len_ptr,
    pm_ptr,
    ps_ptr,
    pt_ptr,  # [P, NSPLIT, H] partials
    sc,
    NKm,
    NSPLIT: tl.constexpr,
    H: tl.constexpr,
    FROM_POOL: tl.constexpr,  # 0 snapshot, 1 indirect load, 2 TMA gather
    ROW: tl.constexpr,  # pool row width (576); unused when FROM_POOL=0
    POOL_ROWS: tl.constexpr,  # pool row count, for the TMA descriptor
    BLOCK_NK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    p = tl.program_id(0)
    sp = tl.program_id(1)
    h = tl.arange(0, H)
    qb = qbuf_ptr + (tl.load(li_ptr + p) * RR + tl.load(slot_ptr + p)) * H * 576
    nk = tl.load(nk_len_ptr + p)
    # The KV pool is one allocation PER LAYER, so a kernel batched across
    # layers cannot reach it from a single base plus a stride; the pair's base
    # comes from a device pointer table instead. The pool is allocated once
    # for the server's life, so those addresses are stable across the replays
    # of a captured graph.
    if FROM_POOL:
        kbase = tl.load(kbase_ptr + p).to(tl.pointer_type(tl.bfloat16))
    if FROM_POOL == 2:
        desc = tl.make_tensor_descriptor(
            kbase,
            shape=[POOL_ROWS, ROW],
            strides=[ROW, 1],
            block_shape=[1, BLOCK_D],
        )
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
        if FROM_POOL == 1:
            sl = tl.load(kslot_ptr + p * NKm + offs, mask=mrow, other=0).to(tl.int64)
        if FROM_POOL == 2:
            sl32 = tl.load(kslot_ptr + p * NKm + offs, mask=mrow, other=0)
        for d0 in range(0, 576, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            qc = tl.load(qb + h[:, None] * 576 + d[None, :])
            if FROM_POOL == 2:
                # TMA row gather: the ONE way to get a data-dependent index
                # staged into shared memory asynchronously. A plain indirect
                # tl.load is not affine, so the pipeliner emits no cp.async for
                # it and falls back to synchronous ld.global (measured on this
                # kernel: 26 cp.async / 3 ld.global becomes 11 / 68).
                kc = desc.gather(sl32, d0)
            elif FROM_POOL:
                # A latent row is written once, when its token enters the
                # pool, and never rewritten -- so reading it here is
                # bit-identical to the snapshot this kernel used to carry, at
                # 4 bytes of row id per kept row instead of a 1152-byte copy.
                kc = tl.load(
                    kbase + sl[:, None] * ROW + d[None, :],
                    mask=mrow[:, None],
                    other=0.0,
                )
            else:
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
# (BLOCK_NK, BLOCK_D, num_warps) keyed by whether kept rows come from the pool.
# The two paths want different tiles: the snapshot streams one contiguous block
# per pair, while the pool read issues one request per row, so what each
# request covers is the lever there and not here. Swept on a 5-layer 64k and
# 256k config; see the pool-read commit for the numbers.
# (BLOCK_NK, BLOCK_D, num_warps, num_stages) keyed by whether kept rows come
# from the pool. The two sources want different tiles because they compile to
# different memory instructions -- see _pool_read_mode.
_TILE = {False: (64, 64, 4, 3), True: (64, 64, 8, 3)}

_TMA_ALLOCATOR_SET = False
_POOL_MODE = None


def _pool_read_mode(device):
    """Pick how the kernel reads kept rows out of the pool: 2 = TMA gather,
    1 = indirect tl.load.

    An indirect load is not affine, so Triton's pipeliner emits no cp.async
    for it and falls back to synchronous ld.global -- measured on this kernel,
    26 cp.async / 3 ld.global becomes 11 / 68, registers go 96 -> 236, and the
    prologue costs +5% at 64k and +23% at 256k. Neither num_stages nor
    num_warps moves that: the loads simply are not staged. A TMA row gather is
    the one form of data-dependent read the hardware will stage into shared
    memory asynchronously, and it restores both (25 cp.async + 12 gather4,
    125 registers, +0.3% / +8.8%).

    Decided once per process by compiling a gather and running it, not by an
    architecture table: what matters is whether this Triton and this device
    agree, and only running it answers that.
    """
    global _POOL_MODE, _TMA_ALLOCATOR_SET
    if _POOL_MODE is not None:
        return _POOL_MODE
    forced = envs.SGLANG_VESTIGEKV_POOL_READ.get()
    if forced in (1, 2):
        if forced == 2:
            _set_tma_allocator(device)
        _POOL_MODE = forced
        logger.info("vestigekv: kept-row read pinned to mode %d", forced)
        return _POOL_MODE
    _POOL_MODE = 1
    if hasattr(tl, "make_tensor_descriptor"):
        _set_tma_allocator(device)
        try:
            src = torch.zeros(64, 64, device=device, dtype=torch.bfloat16)
            idx = torch.zeros(32, device=device, dtype=torch.int32)
            out = torch.zeros(32, 64, device=device, dtype=torch.bfloat16)
            _tma_probe_kernel[(1,)](src, idx, out, 64, W=64, N=32)
            torch.cuda.synchronize()
            _POOL_MODE = 2
        except Exception as e:  # unsupported arch, older Triton, no allocator
            logger.info(
                "vestigekv: TMA row gather unavailable (%s), reading kept rows "
                "with indirect loads instead",
                type(e).__name__,
            )
    return _POOL_MODE


def _set_tma_allocator(device):
    """TMA descriptors need a global scratch allocator; set it once.

    Allocate fresh and let the caching allocator recycle, exactly as
    sglang.kernels.ops.moe.fused_moe_triton_kernels does. triton.set_allocator
    is process-GLOBAL and this model's MoE path sets one too, so whichever runs
    last serves both: a cached buffer handed to two unrelated kernels is shared
    mutable scratch, and it pins its high-water mark for the process.
    """
    global _TMA_ALLOCATOR_SET
    if _TMA_ALLOCATOR_SET:
        return

    def _alloc(size: int, alignment: int, stream):
        return torch.empty(size, device=device, dtype=torch.int8)

    triton.set_allocator(_alloc)
    _TMA_ALLOCATOR_SET = True


@triton.jit
def _tma_probe_kernel(src, idx_ptr, out, R, W: tl.constexpr, N: tl.constexpr):
    """Smallest thing that fails if TMA gather does not work here."""
    desc = tl.make_tensor_descriptor(
        src, shape=[R, W], strides=[W, 1], block_shape=[1, W]
    )
    t = desc.gather(tl.load(idx_ptr + tl.arange(0, N)), 0)
    tl.store(out + tl.arange(0, N)[:, None] * W + tl.arange(0, W)[None, :], t)


def fused_prologue_split(
    qbuf,
    li,
    slot,
    kr,
    v,
    nk_len,
    thr,
    sc,
    out,
    partials,
    p_live=None,
    kslot=None,
    kbase=None,
    nkm=None,
    row=None,
    pool_rows=None,
    mode=None,
):
    """Split-NK prologue reading queries in place from the stacked qbuf.

    qbuf [L, RR, H, 576] (bf16 in production; fp32 accepted -- loads upcast),
    li/slot [P] int64 select each pair's query row. Same outputs as
    fused_prologue; replaces the per-step torch gather+cast+contiguous.

    Kept rows come from one of two places. Pass `kr` [P, NKm, 576] to score a
    snapshot the caller owns, or pass `kslot` [P, NKm] int32 + `kbase` [P]
    int64 (a pool base pointer per pair) and `nkm` to score the rows in place
    in the KV pool, which holds them already. The two are bit-identical: a
    latent row never changes after the write that created it."""
    from_pool = kslot is not None
    if from_pool:
        if kbase is None or nkm is None or row is None:
            raise ValueError(
                "kslot needs kbase, nkm and row: there is no tensor here whose "
                "shape they could be read off, and a guessed row stride reads "
                "the wrong rows silently"
            )
        NKm, ROW = nkm, row
        if mode is None:
            mode = _pool_read_mode(qbuf.device)
        if mode == 2 and pool_rows is None:
            raise ValueError("the TMA descriptor needs the pool's row count")
    else:
        NKm, ROW = kr.shape[1], 0
        mode = 0
    P = p_live if p_live is not None else li.shape[0]
    H = qbuf.shape[2]
    RR = qbuf.shape[1]
    R = v.shape[1]
    max1g, qside_t, qsk_t, qres = out
    pm, ps, pt = partials
    _prologue_scores_kernel[(P, _NSPLIT)](
        qbuf,
        li,
        slot,
        RR,
        kr,
        kslot,
        kbase,
        nk_len,
        pm,
        ps,
        pt,
        sc,
        NKm,
        NSPLIT=_NSPLIT,
        H=H,
        FROM_POOL=mode,
        ROW=ROW,
        POOL_ROWS=pool_rows or 1,
        BLOCK_NK=_TILE[from_pool][0],
        BLOCK_D=_TILE[from_pool][1],
        num_warps=_TILE[from_pool][2],
        num_stages=_TILE[from_pool][3],
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
    a_off_ptr,  # [P] int64 arena offset per pair
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
    abase = tl.load(a_off_ptr + p).to(tl.int64)
    h = tl.load(hit_ptr + abase + offs, mask=m, other=0) != 0
    nb = tl.num_programs(0)
    base = tl.load(offsets_ptr + p * nb + b)
    pos = base + tl.cumsum(h.to(tl.int32), 0) - 1
    li = tl.load(li_ptr + p)
    slot = tl.load(slot_ptr + p)
    arch = tl.load(arch_ptr + abase + offs, mask=m, other=0)
    ok = h & (pos < W)
    tl.store(out_ptr + li * NSLOT * W + slot * W + pos, arch, mask=ok)
    if b == 0:
        t = tl.load(total_ptr + p)
        tl.store(out_len_ptr + li * NSLOT + slot, tl.minimum(t, W))


def compact_fired(
    hit,
    arch,
    a_len,
    a_off,
    li,
    slot,
    fetch_buf,
    fetch_len,
    scratch,
    am_grid,
    p_live=None,
):
    """Deterministic fired-row compaction. scratch: (counts, offsets, total)
    int32 [P, NB] x2 + [P]; fetch_buf [n_li, n_slot, W] int64-compatible.

    am_grid is the largest archive a SINGLE pair can hold -- the bucket grid
    and the counts table are sized by it, not by the shared arena (which only
    bounds the sum). Rows are addressed as a_off[p] + i, masked by a_len[p].
    """
    Am = am_grid
    P = p_live if p_live is not None else a_len.shape[0]
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
        a_off,
        Am,
        W,
        fetch_buf.shape[1],
        BLOCK_A=BLOCK_A,
    )
