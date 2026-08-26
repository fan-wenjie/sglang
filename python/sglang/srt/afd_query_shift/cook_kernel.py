"""The cook as one launch: Triton kernels for the early half's arithmetic.

The three clocks priced the eager cook at 0.573 ms live against 0.143 in isolation -- about
twenty small launches, each a bite of interpreter time the departure thread shares with the
reader and the sender. This folds the arithmetic after the projections into two launches (one
per output pair), so the interpreter's share shrinks with the launch count.

Bit-faithfulness is NOT claimed across this boundary: a fused kernel reassociates what eager
kernels ordered, exactly as attention backends do. `test_afd_cook_kernel.py` holds it to
allclose against `pool_cook`'s eager sequence at float32 tolerances instead, and the served
accuracy figures belong to whichever implementation serves.

Eager remains the reference and the fallback: `cook_early_fused` refuses shapes it was not
built for rather than guessing, and the caller falls back to `pool_cook.cook_early`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cook_early_kernel(
    qk_ptr,  # (rows, 2*KEY) bf16 -- this step's raw early projection
    partial_ptr,  # (rows, 2*KEY) bf16 -- the history, weighted and summed by its owner
    w_ptr,  # (2*KEY, T) weight; only the LAST tap is read here
    beta_ptr,  # (rows, VH) fp32
    q_out_ptr,  # (rows, VH, DK) fp32 -- normalised, expanded
    qt_out_ptr,  # (rows, VH, DK) fp32 -- the coefficient q~
    KEY: tl.constexpr,  # KH * DK
    DK: tl.constexpr,
    GROUP: tl.constexpr,  # VH // KH
    TAPS: tl.constexpr,
):
    """One program per (row, key head): finish the convolution from the partial, normalise,
    then for each of the head's GROUP value heads form q~ = q - beta (k.q) k."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, DK)
    KH: tl.constexpr = KEY // DK

    q_off = row * 2 * KEY + head * DK + d
    k_off = q_off + KEY

    q_acc = tl.load(partial_ptr + q_off).to(tl.float32) + tl.load(qk_ptr + q_off).to(
        tl.float32
    ) * tl.load(w_ptr + (head * DK + d) * TAPS + (TAPS - 1)).to(tl.float32)
    k_acc = tl.load(partial_ptr + k_off).to(tl.float32) + tl.load(qk_ptr + k_off).to(
        tl.float32
    ) * tl.load(w_ptr + (KEY + head * DK + d) * TAPS + (TAPS - 1)).to(tl.float32)
    # SiLU, in the eager path's dtype: the eager conv runs in bf16 and silu's input there is
    # bf16, so the fused path rounds the accumulator the same way before the nonlinearity
    q_acc = q_acc.to(tl.bfloat16).to(tl.float32)
    k_acc = k_acc.to(tl.bfloat16).to(tl.float32)
    q_c = q_acc * tl.sigmoid(q_acc)
    k_c = k_acc * tl.sigmoid(k_acc)

    # L2 normalise; scale multiplies q alone, as `linear_history.normalise` does
    scale = 1.0 / tl.sqrt(tl.cast(DK, tl.float32))
    q_n = q_c / tl.maximum(tl.sqrt(tl.sum(q_c * q_c)), 1e-12) * scale
    k_n = k_c / tl.maximum(tl.sqrt(tl.sum(k_c * k_c)), 1e-12)

    kq = tl.sum(k_n * q_n)
    for g in tl.static_range(GROUP):
        vh = head * GROUP + g
        b = tl.load(beta_ptr + row * KH * GROUP + vh)
        out = row * KH * GROUP * DK + vh * DK + d
        tl.store(q_out_ptr + out, q_n)
        tl.store(qt_out_ptr + out, q_n - b * kq * k_n)


def cook_early_fused(
    early_qk: torch.Tensor,
    beta: torch.Tensor,
    partial_qk: torch.Tensor,
    weight_qk: torch.Tensor,
    *,
    key_heads: int,
    value_heads: int,
    head_k_dim: int,
):
    """`pool_cook.cook_early` in two stores and one launch. Same signature, same returns."""
    rows = early_qk.shape[0]
    key = key_heads * head_k_dim
    if early_qk.shape[-1] != 2 * key or value_heads % key_heads:
        raise ValueError(
            f"the fused cook was built for [q|k] of width {2 * key} and value heads a "
            f"multiple of key heads; got {tuple(early_qk.shape)} and "
            f"{value_heads}/{key_heads}."
        )
    taps = weight_qk.shape[-1]
    q = torch.empty(rows, value_heads, head_k_dim, device=early_qk.device)
    q_tilde = torch.empty_like(q)
    _cook_early_kernel[(rows, key_heads)](
        early_qk.contiguous(),
        partial_qk.contiguous(),
        weight_qk.contiguous(),
        beta.float().contiguous(),
        q,
        q_tilde,
        KEY=key,
        DK=head_k_dim,
        GROUP=value_heads // key_heads,
        TAPS=taps,
    )
    return q_tilde, q


@triton.jit
def _assemble_pack_kernel(
    a_ptr,  # (rows, VH) fp -- decay input
    b_ptr,  # (rows, VH) fp -- write-strength input
    alog_ptr,  # (VH,)
    dtb_ptr,  # (VH,)
    k_ptr,  # (rows, VH, DK) fp32 -- cooked current key
    q_ptr,  # (rows, VH, DK) fp32 -- the early query, kept home
    read_ptr,  # (rows, VH, DV) -- the pushed reading
    v_ptr,  # (rows, VH, DV) -- cooked current value
    rq_ptr,  # (rows, RQ) -- the raw query column for the ring
    pk_ptr,  # (rows, PK) -- the raw [k|v] suffix for the ring
    core_ptr,  # (rows, VH, DV) fp32 out
    slab_ptr,  # (rows, W) fp32 out -- the whole APPLY frame: [k|v|alpha|beta|rq|pk]
    DK: tl.constexpr,
    DV: tl.constexpr,
    VH: tl.constexpr,
    RQ: tl.constexpr,
    PK: tl.constexpr,
    W: tl.constexpr,
    TAIL_BLOCK: tl.constexpr,
):
    """gates + s + core in one launch, and the APPLY frame packed on the way through.

    The naive packing -- five casts and a cat on the serial span path -- gave back on the
    pool what it saved on the host (flown, mixed-to-negative). Here the kernel already
    holds k and v for the core, so storing them into the slab is free bytes; the gates are
    its own outputs; and the ring column's two halves ride extra grid programs. One launch
    where the eager sequence paid roughly eight.
    """
    row = tl.program_id(0)
    pid = tl.program_id(1)
    if pid < VH:
        head = pid
        off = row * VH + head
        a = tl.load(a_ptr + off).to(tl.float32)
        b = tl.load(b_ptr + off).to(tl.float32)
        alog = tl.load(alog_ptr + head).to(tl.float32)
        dtb = tl.load(dtb_ptr + head).to(tl.float32)
        # softplus(x) = log(1 + exp(x)), computed stably as in torch
        x = a + dtb
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        alpha = tl.exp(-tl.exp(alog) * sp)
        beta = tl.sigmoid(b)
        dk = tl.arange(0, DK)
        k = tl.load(k_ptr + off * DK + dk).to(tl.float32)
        kq = tl.sum(k * tl.load(q_ptr + off * DK + dk).to(tl.float32))
        s = beta * kq
        dv = tl.arange(0, DV)
        reading = tl.load(read_ptr + off * DV + dv).to(tl.float32)
        v = tl.load(v_ptr + off * DV + dv).to(tl.float32)
        tl.store(core_ptr + off * DV + dv, alpha * reading + s * v)
        base = row * W
        tl.store(slab_ptr + base + head * DK + dk, k)
        tl.store(slab_ptr + base + VH * DK + head * DV + dv, v)
        tl.store(slab_ptr + base + VH * DK + VH * DV + head, alpha)
        tl.store(slab_ptr + base + VH * DK + VH * DV + VH + head, beta)
    else:
        # the ring column's two halves, block-copied by the programs past the heads
        block = pid - VH
        idx = block * TAIL_BLOCK + tl.arange(0, TAIL_BLOCK)
        tail_off = VH * DK + VH * DV + 2 * VH
        rq = tl.load(rq_ptr + row * RQ + idx, mask=idx < RQ, other=0.0).to(tl.float32)
        tl.store(slab_ptr + row * W + tail_off + idx, rq, mask=idx < RQ)
        pidx = idx  # the same block sweep serves both halves; PK <= RQ blocks re-check
        pk = tl.load(pk_ptr + row * PK + pidx, mask=pidx < PK, other=0.0).to(tl.float32)
        tl.store(slab_ptr + row * W + tail_off + RQ + pidx, pk, mask=pidx < PK)


def assemble_and_pack(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    reading: torch.Tensor,
    v: torch.Tensor,
    raw_q: torch.Tensor,
    packed_kv: torch.Tensor,
):
    """`gates` + `s = beta (k.q)` + `core = alpha reading + s v`, and the APPLY frame's
    packed tensor `[k | v | alpha | beta | raw_q | packed_kv]`, one launch.

    Returns `(core flat, slab)`. The slab is the whole wire payload for the advance; its
    last two segments are the ring column exactly as the old `torch.cat` built it.
    """
    rows, vh, dk = k.shape
    dv = reading.shape[-1]
    rq_w = raw_q.shape[-1]
    pk_w = packed_kv.shape[-1]
    width = vh * dk + vh * dv + 2 * vh + rq_w + pk_w
    core = torch.empty(rows, vh, dv, device=k.device, dtype=torch.float32)
    slab = torch.empty(rows, width, device=k.device, dtype=torch.float32)
    tail_block = 1024
    tail_programs = max(
        (rq_w + tail_block - 1) // tail_block, (pk_w + tail_block - 1) // tail_block
    )
    _assemble_pack_kernel[(rows, vh + tail_programs)](
        a.contiguous(),
        b.contiguous(),
        A_log.contiguous(),
        dt_bias.contiguous(),
        k.contiguous(),
        q.contiguous(),
        reading.contiguous(),
        v.contiguous(),
        raw_q.contiguous(),
        packed_kv.contiguous(),
        core,
        slab,
        DK=dk,
        DV=dv,
        VH=vh,
        RQ=rq_w,
        PK=pk_w,
        W=width,
        TAIL_BLOCK=tail_block,
    )
    return core.reshape(rows, -1), slab
