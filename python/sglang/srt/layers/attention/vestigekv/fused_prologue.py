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
