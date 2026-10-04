# SPDX-License-Identifier: Apache-2.0
"""One launch to file a batch of salience keys into the per-request ring.

Replaces, per MLA layer per decode step, act_quant + three index_copy_ + the
address arithmetic (seven launches of 2-5 us each; 0.24-0.30 ms of an 11.9 ms
step at 11 layers, measured by node-level nsys on GLM-5.3-Flash). The
quantisation is DSA's `_act_quant_kernel` with round_scale (ue8m0), copied
line for line: the row max is exact in any order, and log2/ceil/exp2, the
division, the clamp and the fp8 cast are the same device functions on the
same fp32 values, so the ring holds the bytes the old path wrote.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ring_write_kernel(
    key_ptr,  # [T, D] bf16/fp32 keys, one per token of the batch
    pos_ptr,  # [T] int64 positions
    slot_ptr,  # [T] int64 request slots
    ring_ptr,  # [R1, RING, D] fp8 e4m3 (FP8) or bf16
    stamp_ptr,  # [R1, RING] int32
    scale_ptr,  # [R1, RING] fp32 (FP8 only)
    T,
    RING: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    FP8: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    m = rows < T
    pos = tl.load(pos_ptr + rows, mask=m, other=0).to(tl.int64)
    slot = tl.load(slot_ptr + rows, mask=m, other=0).to(tl.int64)
    flat = slot * RING + pos % RING
    cols = tl.arange(0, D)
    x = tl.load(
        key_ptr + rows[:, None] * D + cols[None, :], mask=m[:, None], other=0.0
    ).to(tl.float32)
    # Stamped with the key: sigma() trusts a row only when the stamp says this
    # exact position wrote it.
    tl.store(stamp_ptr + flat, pos.to(tl.int32), mask=m)
    if FP8:
        fp8_min = -448.0
        fp8_max = 448.0
        fp8_max_inv = 1.0 / fp8_max
        amax = tl.max(tl.abs(x), axis=1)
        amax = tl.maximum(amax, 1e-4)
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
        y = x / scale[:, None]
        y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)
        tl.store(ring_ptr + flat[:, None] * D + cols[None, :], y, mask=m[:, None])
        tl.store(scale_ptr + flat, scale, mask=m)
    else:
        tl.store(
            ring_ptr + flat[:, None] * D + cols[None, :],
            x.to(tl.bfloat16),
            mask=m[:, None],
        )


def ring_write(key, positions, slots, ring, stamp, scale):
    """key [T, D]; positions/slots [T] int64; ring [R1, RING, D]; stamp
    [R1, RING] int32; scale [R1, RING] fp32 or None (bf16 ring)."""
    T, D = key.shape
    if T == 0:
        return
    fp8 = ring.dtype == torch.float8_e4m3fn
    if fp8 and scale is None:
        raise ValueError("an fp8 ring needs its scale table")
    key = key.contiguous()
    RING = ring.shape[1]
    BLOCK_T = 32
    _ring_write_kernel[(triton.cdiv(T, BLOCK_T),)](
        key,
        positions,
        slots,
        ring,
        stamp,
        scale if scale is not None else stamp,
        T,
        RING=RING,
        D=D,
        BLOCK_T=BLOCK_T,
        FP8=fp8,
        num_warps=4,
    )
