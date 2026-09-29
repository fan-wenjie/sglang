"""Operand build with no sketch: one pass over the content, and no dot.

A fork of operand_fused._operand_fused_kernel. Measured at 65536 rows: 389.1
-> 21.5 us, 18.1x, against a traffic ceiling of 2.00x. The gap is not the
traffic: the sketch kernel allocates 255 registers, spills 108 bytes, and runs
92 LDL / 94 STL, which holds it to 389 GB/s where this one reaches 3517 -- the
same bandwidth the scan kernels get. The spill is a property of the SERVED
kernel and worth its own fix; this fork only avoids inheriting it.

The per-request build is 7.5 ms and this kernel is about 1% of it at the
serving shape, so 18x here is not 18x on a build. What sets the build cost
sits elsewhere and is being profiled; do not quote this speedup as a request
-level number.

With a zero basis both of its phases collapse. `csk = content @ V^T` is
identically zero, so the first K-loop over the content and every load of V
disappear; and `rho = ||content - csk @ V||` becomes `||content||`, so the
second loop keeps its content read and loses its dot. The content is then read
ONCE instead of twice, no tensor-core work remains, and the [R, KV] basis is
never touched: 2048 -> 1024 bytes per row of content traffic.

csk is still RETURNED as a [A, r] tensor of zeros, because three readers index
it by shape (the tier's csk property, refresh_membership's tgt_csk, and
extend_closed's cat). It is returned as a broadcast VIEW of one zero row, so
those readers see the shape they require and the storage is r*2 bytes instead
of A*r*2 -- 128 bytes against 16 MiB per layer at 131072 closed rows. A zero
matrix only has to be stored once. The kernel therefore writes rho and side
only, and the fp16 range guard has nothing left to check.

A fork rather than a constexpr arm in the original, for the reason
.claude/rules/disassemble-check-for-spills.md gives: an arm that exists to
DELETE a loop cannot be predicated off inside one, and the served kernel's
register allocation must not move because of a variant it never runs.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.vestigekv import defaults as D

# BA is tuned against this kernel. BD is NOT: it is taken from the same helper
# the sketch kernel uses, because it sets the K-loop's summation order and rho
# is only bit-identical to the flag form while that order matches.
BRANCH_BA = 64
BRANCH_NUM_WARPS = 4


@triton.jit
def _operand_branch_kernel(
    kbuf_ptr,  # [pool, 576] bf16
    slots_ptr,  # [A] int64 archived pool slots
    rho_ptr,  # [A] fp32 out
    side_ptr,  # [A, SD] bf16 out
    A,
    BA: tl.constexpr,  # rows per program
    BD: tl.constexpr,  # content K-chunk
    KV: tl.constexpr,  # 512
    SD: tl.constexpr,  # 64
    ROW: tl.constexpr,  # 576
):
    p = tl.program_id(0)
    a = p * BA + tl.arange(0, BA)
    m = a < A
    slots = tl.load(slots_ptr + a, mask=m, other=0)
    # rho = ||content||: with nothing projected out, the residual IS the row.
    rho2 = tl.zeros([BA], dtype=tl.float32)
    for d0 in range(0, KV, BD):
        d = d0 + tl.arange(0, BD)
        c = tl.load(
            kbuf_ptr + slots[:, None] * ROW + d[None, :], mask=m[:, None], other=0.0
        ).to(tl.float32)
        rho2 += tl.sum(c * c, 1)
    tl.store(rho_ptr + a, tl.sqrt(rho2), mask=m)
    sd = tl.arange(0, SD)
    s = tl.load(
        kbuf_ptr + slots[:, None] * ROW + (KV + sd)[None, :], mask=m[:, None], other=0.0
    )
    tl.store(side_ptr + a[:, None] * SD + sd[None, :], s, mask=m[:, None])


def build_operands_branch(kbuf: torch.Tensor, arch_slots: torch.Tensor, V: torch.Tensor):
    """Same contract as build_operands_fused; V is read only for its rank."""
    A = arch_slots.numel()
    dev = kbuf.device
    r = V.shape[0]
    # One zero row, viewed as [A, r]: see the module docstring. Nothing on
    # this path writes into it, so a non-contiguous view is safe; build()
    # assigns whole tensors and extend_closed re-expands.
    csk = torch.zeros(1, r, dtype=torch.float16, device=dev).expand(A, r)
    rho = torch.empty(A, dtype=torch.float32, device=dev)
    side = torch.empty(A, D.SIDECAR_DIM, dtype=torch.bfloat16, device=dev)
    if A == 0:
        return csk, rho, side
    _operand_branch_kernel[(triton.cdiv(A, BRANCH_BA),)](
        kbuf,
        arch_slots,
        rho,
        side,
        A,
        BA=BRANCH_BA,
        BD=D.d_block_for_rank(r),
        KV=D.KV_LORA_RANK,
        SD=D.SIDECAR_DIM,
        ROW=kbuf.shape[-1],
        num_warps=BRANCH_NUM_WARPS,
    )
    return csk, rho, side
