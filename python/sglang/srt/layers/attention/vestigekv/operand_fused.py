"""Fused tier-2 operand builder: gather + sketch-project + residual + casts.

Replaces the chunked torch loop in RecallTier.build's operand branch (the
step-0 synchronous provisional build's dominant remaining cost after the
eigh removal): per archived row, gather the pool row, project the content
onto the sketch basis, compute the residual norm, and store the 16-bit
index operands -- ONE kernel, one pass over the archive (content is read
twice: once for the projection, once for the residual, exactly like the
torch loop, so peak memory stays flat and no [A,512] fp32 block is ever
materialized).

Numerics: bf16 rows upcast to fp32, tl.dot with input_precision="ieee",
storage rounding at the end (csk fp16, side bf16, rho fp32) -- the
multiply-then-add discipline. Reduction order differs from cuBLAS at the
~1 ulp level, so the gate is fire-set stability + retrieval, not bit
equality against the torch loop (the same bar the 16-bit index shipped
under).
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.vestigekv import defaults as D


@triton.jit
def _operand_fused_kernel(
    kbuf_ptr,  # [pool, 576] bf16
    slots_ptr,  # [A] int64 archived pool slots
    v_ptr,  # [R, KV] fp32 sketch basis
    csk_ptr,  # [A, R] fp16 out
    rho_ptr,  # [A] fp32 out
    side_ptr,  # [A, SD] bf16 out
    amax_ptr,  # [1] fp32 out (atomic max of |csk|, fp16 range guard)
    A,
    BA: tl.constexpr,  # rows per program
    BD: tl.constexpr,  # content K-chunk
    R: tl.constexpr,  # 64
    KV: tl.constexpr,  # 512
    SD: tl.constexpr,  # 64
    ROW: tl.constexpr,  # 576
):
    p = tl.program_id(0)
    a = p * BA + tl.arange(0, BA)
    m = a < A
    slots = tl.load(slots_ptr + a, mask=m, other=0)
    # phase 1: csk = content @ V^T   (fp32 ieee, K-loop over 512)
    csk = tl.zeros([BA, R], dtype=tl.float32)
    for d0 in range(0, KV, BD):
        d = d0 + tl.arange(0, BD)
        c = tl.load(
            kbuf_ptr + slots[:, None] * ROW + d[None, :], mask=m[:, None], other=0.0
        ).to(tl.float32)
        v = tl.load(v_ptr + tl.arange(0, R)[:, None] * KV + d[None, :])
        csk += tl.dot(c, tl.trans(v), input_precision="ieee")
    # phase 2: rho = || content - csk @ V ||   (second pass over content)
    rho2 = tl.zeros([BA], dtype=tl.float32)
    for d0 in range(0, KV, BD):
        d = d0 + tl.arange(0, BD)
        c = tl.load(
            kbuf_ptr + slots[:, None] * ROW + d[None, :], mask=m[:, None], other=0.0
        ).to(tl.float32)
        v = tl.load(v_ptr + tl.arange(0, R)[:, None] * KV + d[None, :])
        recon = tl.dot(csk, v, input_precision="ieee")
        diff = c - recon
        rho2 += tl.sum(diff * diff, 1)
    tl.store(rho_ptr + a, tl.sqrt(rho2), mask=m)
    tl.store(
        csk_ptr + a[:, None] * R + tl.arange(0, R)[None, :],
        csk.to(tl.float16),
        mask=m[:, None],
    )
    tl.atomic_max(amax_ptr, tl.max(tl.abs(csk)))
    # side: branch slice cast (bf16 -> bf16 passthrough copy into the operand
    # table's own layout)
    sd = tl.arange(0, SD)
    s = tl.load(
        kbuf_ptr + slots[:, None] * ROW + (KV + sd)[None, :], mask=m[:, None], other=0.0
    )
    tl.store(side_ptr + a[:, None] * SD + sd[None, :], s, mask=m[:, None])


def build_operands_fused(kbuf: torch.Tensor, arch_slots: torch.Tensor, V: torch.Tensor):
    """kbuf [pool,576] bf16; arch_slots [A] int64; V [r,512] fp32.
    Returns (csk fp16 [A,r], rho fp32 [A], side bf16 [A,64])."""
    A = arch_slots.numel()
    dev = kbuf.device
    r = V.shape[0]
    csk = torch.empty(A, r, dtype=torch.float16, device=dev)
    rho = torch.empty(A, dtype=torch.float32, device=dev)
    side = torch.empty(A, D.SIDECAR_DIM, dtype=torch.bfloat16, device=dev)
    amax = torch.zeros(1, dtype=torch.float32, device=dev)
    if A == 0:
        return csk, rho, side
    grid = (triton.cdiv(A, 64),)
    _operand_fused_kernel[grid](
        kbuf,
        arch_slots,
        V.contiguous(),
        csk,
        rho,
        side,
        amax,
        A,
        BA=64,
        BD=64,
        R=r,
        KV=D.KV_LORA_RANK,
        SD=D.SIDECAR_DIM,
        ROW=kbuf.shape[-1],
        num_warps=4,
    )
    # fp16 range guard, same contract as the torch loop's assert
    torch._assert_async((amax < 6e4).all().to(torch.bool))
    return csk, rho, side
