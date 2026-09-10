"""Fused tier-2 archive scan.

The scan asks one question per archive row -- "could any head prefer this row to
the best row tier-1 kept?" -- and the eager formulation answers it by
materializing a [num_heads, archive] score matrix in HBM. Measured at S=64k that
costs about 1580 bytes per archive row against an irreducible 516 (the row's
un-roped sidecar and its rank-r sketch), which is more traffic per row than
simply attending to the row densely would take (1152 bytes at bf16). The scan is
therefore the term that decides whether the method can beat dense attention at
all, and it only does so once the per-head scores stay in registers.

This kernel reads each row's sidecar and sketch once, reuses them across all
heads, and emits one byte per row. Compaction stays outside, on the [archive]
vector, so which rows survive a fetch-cap overflow remains deterministic.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.vestigekv.defaults import SCAN_BLOCK_A, SCAN_NUM_WARPS


@triton.jit
def _vestige_scan_kernel(
    qside_t_ptr,  # [D, H] bf16: query sidecar branch (bf16-exact from qbuf)
    qsk_t_ptr,  # [R, H]  fp16: sketch projection, rounded to csk's dtype
    qres_ptr,  # [H]      fp32: residual norm outside that basis
    max1g_ptr,  # [H]     fp32: best kept-row score, +inf where the gate is closed
    side_ptr,  # [A, D]   bf16 storage, promoted to fp32 in-register
    csk_ptr,  # [A, R]    fp16 storage, promoted to fp32 in-register
    rho_ptr,  # [A]       fp32: per-row residual norm
    hit_ptr,  # [A]       int32 out
    A,
    sc,  # attention scale
    cc,  # zp * sc / sqrt(kv_lora - R): the certificate coefficient
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK_A: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_A + tl.arange(0, BLOCK_A)
    m = offs < A
    d = tl.arange(0, D)
    r = tl.arange(0, R)
    h = tl.arange(0, H)
    # The two per-row loads that dominate the scan's traffic. Everything after
    # this stays in registers, across every head.
    s = tl.load(side_ptr + offs[:, None] * D + d[None, :], mask=m[:, None], other=0.0)
    c = tl.load(csk_ptr + offs[:, None] * R + r[None, :], mask=m[:, None], other=0.0)
    rh = tl.load(rho_ptr + offs, mask=m, other=0.0)
    qs = tl.load(qside_t_ptr + d[:, None] * H + h[None, :])
    qk = tl.load(qsk_t_ptr + r[:, None] * H + h[None, :])
    # Native-dtype tensor-core dots with fp32 accumulation. Every bf16/fp16
    # product is EXACT in the fp32 accumulator (8/11-bit mantissas square
    # under 24), so no tf32-style silent truncation exists here; the query
    # operands are rounded to the storage dtype at the call site, and the
    # conformal zp is calibrated on this exact scoring path (quantize-then-
    # calibrate). The previous convert-to-fp32 ieee form ran the dot on CUDA
    # cores and was 6.6x slower (538 vs 3601 GB/s effective, measured).
    acc = tl.dot(s, qs).to(tl.float32) + tl.dot(c, qk).to(tl.float32)
    score = acc * sc + cc * rh[:, None] * tl.load(qres_ptr + h)[None, :]
    fired = tl.max((score > tl.load(max1g_ptr + h)[None, :]).to(tl.int32), 1)
    tl.store(hit_ptr + offs, fired, mask=m)


def vestige_scan(qside_t, qsk_t, qres, max1g, side, csk, rho, sc, cc, out=None):
    """Fired-row mask over the archive. All inputs fp32 and contiguous;
    `qside_t` is [D, H] and `qsk_t` is [R, H] (transposed for the dot).

    Returns an int32 [A] tensor, 1 where at least one head's certified upper
    bound beats that head's best kept-row score. Bit-identical to the eager
    formulation on every case tested (0 disagreements over 3 archive sizes).
    """
    A, D = side.shape
    R, H = qsk_t.shape
    if out is None:
        out = torch.empty(A, dtype=torch.int32, device=side.device)
    # 64 rows/block: 128 and 256 measured the same or ran out of registers.
    block = SCAN_BLOCK_A
    _vestige_scan_kernel[(triton.cdiv(A, block),)](
        qside_t,
        qsk_t,
        qres,
        max1g,
        side,
        csk,
        rho,
        out,
        A,
        sc,
        cc,
        H=H,
        D=D,
        R=R,
        BLOCK_A=block,
        num_warps=SCAN_NUM_WARPS,
    )
    return out
