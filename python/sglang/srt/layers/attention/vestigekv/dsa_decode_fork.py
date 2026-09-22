# SPDX-License-Identifier: Apache-2.0
"""DSA's split-K sparse decode, forked to read a lane's rows from the tiers.

Forked from `_sparse_mla_decode_split_kernel` in
sglang/kernels/ops/attention/dsa/triton_sparse_mla_decode.py. Three things
change: where a row id comes from (the kept table then the fetch buffer, or
the indexer's selection when the lane is fenced, instead of one [N, topk]
table), the per-lane row count (a runtime value instead of the constexpr
width), and an empty split writes its partials instead of returning early,
because the split count is fixed per capture while the count is per lane.
The gather, the masking of pad rows, the exp2 softmax, the bf16 P.V, the
bf16 split partials and the reduce kernel are DSA's own, so a fenced lane
computes exactly what the baseline computes on the same rows.

Why a second fork: the MLA-decode fork (decode_fork.py) costs 20 us per
layer per step on GLM-5.3-Flash against 9 us for DSA's kernel on the same
~2k rows (node-level nsys, 2026-09-22), and the difference is the schedule:
DSA's kernel was tiled for a few thousand scattered rows, upstream's MLA
stage 1 for a long contiguous CSR.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
    _PREFERRED_BLOCK_K,
    _no_async_copy,
    _row_strides,
    _sparse_mla_block_k,
)
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
    _FP8_MAX,
    _G,
    LOG2E,
    _cu_count,
    _kv_splits_heuristic,
    _sparse_mla_decode_reduce_kernel,
)


@triton.jit
def _vk_dsa_decode_split_kernel(
    q_nope_ptr,  # [N, H, D_V]
    q_rope_ptr,  # [N, H, D_TAIL]
    kv_ptr,  # [num_pages, 1, KV_DIM]
    lse_partial_ptr,  # [N, KV_SPLITS, H_padded]  fp32
    acc_partial_ptr,  # [N, KV_SPLITS, H_padded, D_V]  bf16
    qk_scale,
    fp8_max,
    # ---- the tiers (VestigeKVRows) ----
    vk_slots,  # [N] int64 pool slot per lane
    vk_kept_buf,  # [R1, VK_CAP] int32
    vk_kept_len,  # [R1] int32
    vk_fetch_buf,  # [R1, VK_FW] int32
    vk_fetch_len,  # [R1] int32
    vk_fetch_ovf,  # [R1] int32; nonzero = fenced
    vk_r2t,  # [R1 - 1, VK_R2T] int32 page table (VK_TOPK == 0 fence)
    vk_seq,  # [N] int64 this step's seq_len per lane
    vk_loc_ptr,  # [N] int64 this step's appended row (VK_TOPK == 0 fence)
    vk_topk,  # VK_TOPK > 0: [N, VK_TOPK] int32 selection, -1 padded
    vk_qbuf,  # VK_QBUF: [R1, H, D_V + D_TAIL]; this step's q filed by slot
    qbuf_stride_s,
    qbuf_stride_h,
    H: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    STRIDE_QN_T: tl.constexpr,
    STRIDE_QN_H: tl.constexpr,
    STRIDE_QR_T: tl.constexpr,
    STRIDE_QR_H: tl.constexpr,
    USE_FP8_DOT: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    VK_CAP: tl.constexpr,
    VK_FW: tl.constexpr,
    VK_R2T: tl.constexpr,
    FENCE: tl.constexpr,
    VK_TOPK: tl.constexpr,
    VK_TOPK_K: tl.constexpr,
    VK_KPOOL: tl.constexpr,
    VK_QBUF: tl.constexpr,
):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    g = tl.arange(0, _G)

    input_type = kv_ptr.dtype.element_ty if USE_FP8_DOT else tl.bfloat16
    if USE_FP8_DOT:
        p_dot_scale = 1.0 / fp8_max
    else:
        p_dot_scale = 1.0

    # ---- the lane's row set: count and source (the fork's first change) ----
    slot_t = tl.load(vk_slots + t).to(tl.int64)
    nk = tl.load(vk_kept_len + slot_t).to(tl.int64)
    nf = tl.load(vk_fetch_len + slot_t).to(tl.int64)
    fenced = False
    if FENCE:
        fenced = tl.load(vk_fetch_ovf + slot_t) != 0
    seq_t = tl.load(vk_seq + t).to(tl.int64)
    if VK_TOPK > 0:
        # DSA's own attended count (dsa/utils.compute_dsa_seqlens): whole
        # index pools clamped to the budget plus the unpooled tail.
        tail = seq_t % VK_KPOOL
        fenced_n = tl.minimum(seq_t - tail, VK_TOPK_K) + tail
    else:
        fenced_n = seq_t
    n_t = tl.where(fenced, fenced_n, nk + nf)
    loc_t = tl.load(vk_loc_ptr + t)

    qn_base = q_nope_ptr + t * STRIDE_QN_T
    qn_row = qn_base + h_offs[:, None] * STRIDE_QN_H
    q0 = tl.load(qn_row + g[None, :], mask=h_mask[:, None], other=0.0)
    if VK_QBUF:
        # This step's query, filed for the next step's recall scan: one split
        # writes, every head tile writes its own heads. Replaces a per-layer
        # index_copy_ launch. (The row pointer is defined outside the runtime
        # branch: a name bound inside one is not visible after it.)
        qb_row = vk_qbuf + slot_t * qbuf_stride_s + h_offs[:, None] * qbuf_stride_h
        if pid_k == 0:
            tl.store(qb_row + g[None, :], q0, mask=h_mask[:, None])
    q0 = q0.to(input_type)
    if NUM_GROUPS >= 2:
        q1 = tl.load(qn_row + (_G + g)[None, :], mask=h_mask[:, None], other=0.0)
        if VK_QBUF:
            if pid_k == 0:
                tl.store(qb_row + (_G + g)[None, :], q1, mask=h_mask[:, None])
        q1 = q1.to(input_type)
    if NUM_GROUPS >= 3:
        q2 = tl.load(qn_row + (2 * _G + g)[None, :], mask=h_mask[:, None], other=0.0)
        if VK_QBUF:
            if pid_k == 0:
                tl.store(qb_row + (2 * _G + g)[None, :], q2, mask=h_mask[:, None])
        q2 = q2.to(input_type)
    if NUM_GROUPS >= 4:
        q3 = tl.load(qn_row + (3 * _G + g)[None, :], mask=h_mask[:, None], other=0.0)
        if VK_QBUF:
            if pid_k == 0:
                tl.store(qb_row + (3 * _G + g)[None, :], q3, mask=h_mask[:, None])
        q3 = q3.to(input_type)
    # tl.arange rejects empty ranges; rope-less MLA (D_TAIL == 0) has no tail dot.
    if D_TAIL > 0:
        dt = tl.arange(0, D_TAIL)
        qr_row = q_rope_ptr + t * STRIDE_QR_T + h_offs[:, None] * STRIDE_QR_H
        q_tail = tl.load(qr_row + dt[None, :], mask=h_mask[:, None], other=0.0)
        if VK_QBUF:
            if pid_k == 0:
                tl.store(qb_row + (D_V + dt)[None, :], q_tail, mask=h_mask[:, None])
        q_tail = q_tail.to(input_type)

    H_padded = tl.cdiv(H, BLOCK_H) * BLOCK_H
    lse_base = t * KV_SPLITS * H_padded + pid_k * H_padded
    ap_base = t * KV_SPLITS * H_padded * D_V + pid_k * H_padded * D_V

    # ---- the split partition, per lane (the fork's second change) ----
    tiles_per_segment = tl.cdiv(n_t, KV_SPLITS * BLOCK_K)
    num_tiles = tl.cdiv(n_t, BLOCK_K)
    tile_start = pid_k * tiles_per_segment
    tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 2:
        acc1 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 3:
        acc2 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 4:
        acc3 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    for j in tl.range(tile_start, tile_end, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < n_t

        # ---- the row source (the fork's third change) ----
        if VK_TOPK > 0:
            sel = tl.load(vk_topk + t * VK_TOPK + k_pos, mask=valid & fenced, other=-1)
        else:
            sel = tl.load(vk_r2t + slot_t * VK_R2T + k_pos, mask=valid & fenced, other=-1)
            sel = tl.where(k_pos == seq_t - 1, loc_t.to(sel.dtype), sel)
        kept = tl.load(
            vk_kept_buf + slot_t * VK_CAP + k_pos,
            mask=valid & (k_pos < nk) & (not fenced),
            other=-1,
        )
        fired = tl.load(
            vk_fetch_buf + slot_t * VK_FW + (k_pos - nk),
            mask=valid & (k_pos >= nk) & (not fenced),
            other=-1,
        )
        slot = tl.where(fenced, sel, tl.where(k_pos < nk, kept, fired))
        valid = valid & (slot >= 0)
        page = tl.where(valid, slot, 0).to(tl.int64)

        kv_base = kv_ptr + page[:, None] * KV_DIM
        kv0 = tl.load(kv_base + g[None, :], mask=valid[:, None], other=0.0).to(input_type)
        if NUM_GROUPS >= 2:
            kv1 = tl.load(kv_base + (_G + g)[None, :], mask=valid[:, None], other=0.0).to(input_type)
        if NUM_GROUPS >= 3:
            kv2 = tl.load(kv_base + (2 * _G + g)[None, :], mask=valid[:, None], other=0.0).to(input_type)
        if NUM_GROUPS >= 4:
            kv3 = tl.load(kv_base + (3 * _G + g)[None, :], mask=valid[:, None], other=0.0).to(input_type)

        scores = tl.dot(q0, tl.trans(kv0))
        if NUM_GROUPS >= 2:
            scores += tl.dot(q1, tl.trans(kv1))
        if NUM_GROUPS >= 3:
            scores += tl.dot(q2, tl.trans(kv2))
        if NUM_GROUPS >= 4:
            scores += tl.dot(q3, tl.trans(kv3))
        if D_TAIL > 0:
            kv_tail = tl.load(
                kv_base + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
            ).to(input_type)
            scores += tl.dot(q_tail, tl.trans(kv_tail))
        scores = scores * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        if USE_FP8_DOT:
            p_dot = (p * fp8_max).to(input_type)
        else:
            p_dot = p.to(input_type)
        acc0 = acc0 * alpha[:, None] + tl.dot(p_dot, kv0).to(tl.float32) * p_dot_scale
        if NUM_GROUPS >= 2:
            acc1 = acc1 * alpha[:, None] + tl.dot(p_dot, kv1).to(tl.float32) * p_dot_scale
        if NUM_GROUPS >= 3:
            acc2 = acc2 * alpha[:, None] + tl.dot(p_dot, kv2).to(tl.float32) * p_dot_scale
        if NUM_GROUPS >= 4:
            acc3 = acc3 * alpha[:, None] + tl.dot(p_dot, kv3).to(tl.float32) * p_dot_scale
        m_i = m_new
        l_i = l_new

    # An empty split (a lane shorter than the partition) writes its partials
    # -- lse = neg_large, acc = 0 -- rather than returning: the reduce reads
    # every split of every lane, and a stale bf16 partial could carry a NaN.
    neg_large = -1073741824.0
    denom = tl.maximum(l_i, 1.0e-30)
    inv_denom = 1.0 / denom
    has_data = l_i > 0.0
    acc0 = tl.where(has_data[:, None], acc0 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 2:
        acc1 = tl.where(has_data[:, None], acc1 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 3:
        acc2 = tl.where(has_data[:, None], acc2 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 4:
        acc3 = tl.where(has_data[:, None], acc3 * inv_denom[:, None], 0.0)

    lse = tl.where(has_data, tl.log2(l_i) + m_i, neg_large)
    tl.store(lse_partial_ptr + lse_base + h_offs, lse, mask=h_mask)
    tl.store(
        acc_partial_ptr + ap_base + h_offs[:, None] * D_V + g[None, :],
        acc0.to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    if NUM_GROUPS >= 2:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (_G + g)[None, :],
            acc1.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
    if NUM_GROUPS >= 3:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (2 * _G + g)[None, :],
            acc2.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
    if NUM_GROUPS >= 4:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (3 * _G + g)[None, :],
            acc3.to(tl.bfloat16),
            mask=h_mask[:, None],
        )


_vk_splitk = {}


def _vk_splitk_bufs(bs, kv_splits, h_padded, d_v, device):
    # DSA's _get_splitk_bufs reserves for a batch of 128 at the split count of
    # the first call; the tiers' split count is up to three times DSA's
    # (max_rows is the kept capacity plus the fetch width), and that
    # reservation was 0.4 GB of the graph pool at bs=1 on GLM-5.3-Flash. The
    # partials are scratch within one launch, so a buffer sized to the largest
    # call so far serves every graph.
    needed_lse = bs * kv_splits * h_padded
    needed_acc = needed_lse * d_v
    bufs = _vk_splitk.get(device)
    if bufs is None or bufs[0].numel() < needed_lse or bufs[1].numel() < needed_acc:
        bufs = (
            torch.empty(needed_lse, dtype=torch.float32, device=device),
            torch.empty(needed_acc, dtype=torch.bfloat16, device=device),
        )
        _vk_splitk[device] = bufs
    lse = bufs[0][:needed_lse].view(bs, kv_splits, h_padded)
    acc = bufs[1][:needed_acc].view(bs, kv_splits, h_padded, d_v)
    return lse, acc


def vk_dsa_decode(
    q, kv, out, vk, sm_scale, *, d_v=512, kv_splits=None, max_rows=None, q_rope=None
):
    """q [bs, H, D_V + D_TAIL] bf16 (the absorbed query), or with `q_rope`
    [bs, H, D_TAIL] given, q is the [bs, H, D_V] latent part and the two are
    read in place, as DSA's own decode reads the model's split q; kv
    [pool, 1, KV_DIM] bf16; out [bs, H, D_V] bf16, written; vk:
    VestigeKVRows; sm_scale: the layer's scaling (LOG2E is applied here, as
    DSA does). `max_rows` bounds a lane's row count (kept capacity + fetch
    width, or the selection width) and sizes the split count; it is a
    capture-time constant."""
    bs, H, dim = q.shape
    if q_rope is None:
        d_tail = dim - d_v
        q_nope = q[:, :, :d_v]
        q_rope = q[:, :, d_v:]
    else:
        assert dim == d_v, f"split q: latent width {dim} != d_v {d_v}"
        d_tail = q_rope.shape[-1]
        q_nope = q
    kv_dim = kv.shape[-1]
    if kv.dtype != torch.bfloat16 or q.dtype != torch.bfloat16:
        raise NotImplementedError("vk_dsa_decode serves the bf16 latent pool")
    q_nope, stride_qn_t, stride_qn_h = _row_strides(q_nope)
    q_rope, stride_qr_t, stride_qr_h = _row_strides(q_rope)
    BLOCK_H = 16
    BLOCK_K = _sparse_mla_block_k(kv)
    n_head_blocks = (H + BLOCK_H - 1) // BLOCK_H
    h_padded = n_head_blocks * BLOCK_H
    assert d_v % 128 == 0, f"d_v must be divisible by 128, got {d_v}"
    num_groups = d_v // 128
    if max_rows is None:
        max_rows = vk.kept_buf.shape[1] + vk.fetch_buf.shape[1]
        if vk.topk is not None:
            max_rows = max(max_rows, vk.topk.shape[1])
    max_kv_splits = max(1, max_rows // _PREFERRED_BLOCK_K)
    if kv_splits is None:
        num_cu = _cu_count()
        base_ctas = max(1, bs * n_head_blocks)
        target_wg_per_cu = 2.0 if base_ctas <= max(1, num_cu // 16) else 1.0
        kv_splits = min(
            _kv_splits_heuristic(
                bs, H, BLOCK_H, num_cu=num_cu,
                target_wg_per_cu=target_wg_per_cu, max_kv_splits=max_kv_splits,
            ),
            max_kv_splits,
        )
    else:
        kv_splits = min(kv_splits, max_kv_splits)
    kv_splits = max(1, kv_splits)
    qk_scale = float(sm_scale) * LOG2E
    lse_partial, acc_partial = _vk_splitk_bufs(bs, kv_splits, h_padded, d_v, q.device)
    topk = vk.topk
    qbuf = vk.qbuf if (vk.qbuf is not None and vk.tiers) else None
    grid_split = (bs, n_head_blocks, kv_splits)
    with _no_async_copy():
        _vk_dsa_decode_split_kernel[grid_split](
            q_nope,
            q_rope,
            kv,
            lse_partial,
            acc_partial,
            qk_scale,
            _FP8_MAX,
            vk.slots,
            vk.kept_buf,
            vk.kept_len,
            vk.fetch_buf,
            vk.fetch_len,
            vk.fetch_ovf,
            vk.r2t,
            vk.seq,
            vk.loc,
            topk if topk is not None else vk.seq,
            qbuf if qbuf is not None else q,
            qbuf.stride(0) if qbuf is not None else 0,
            qbuf.stride(1) if qbuf is not None else 0,
            H=H,
            KV_DIM=kv_dim,
            D_V=d_v,
            D_TAIL=d_tail,
            NUM_GROUPS=num_groups,
            STRIDE_QN_T=stride_qn_t,
            STRIDE_QN_H=stride_qn_h,
            STRIDE_QR_T=stride_qr_t,
            STRIDE_QR_H=stride_qr_h,
            USE_FP8_DOT=False,
            KV_SPLITS=kv_splits,
            BLOCK_H=BLOCK_H,
            BLOCK_K=BLOCK_K,
            VK_CAP=vk.kept_buf.shape[1],
            VK_FW=vk.fetch_buf.shape[1],
            VK_R2T=vk.r2t.shape[1],
            FENCE=vk.fence,
            VK_TOPK=topk.shape[1] if topk is not None else 0,
            VK_TOPK_K=vk.topk_k if topk is not None else 0,
            VK_KPOOL=max(1, vk.kpool) if topk is not None else 1,
            VK_QBUF=qbuf is not None,
            num_warps=4,
            num_stages=2,
        )
    D_CHUNK = 64
    grid_reduce = (bs, H, (d_v + D_CHUNK - 1) // D_CHUNK)
    _sparse_mla_decode_reduce_kernel[grid_reduce](
        lse_partial,
        acc_partial,
        out,
        H=H,
        D_V=d_v,
        KV_SPLITS=kv_splits,
        ACTIVE_SPLITS=kv_splits,
        ACTIVE_SPLITS_POW2=triton.next_power_of_2(kv_splits),
        D_CHUNK=D_CHUNK,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return out
