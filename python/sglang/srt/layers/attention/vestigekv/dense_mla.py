# SPDX-License-Identifier: Apache-2.0
"""Dense MLA decode for a lane the recall fenced, with no index list at all.

A fenced lane attends its whole row set, and that set is already described by
`req_to_token`: copying it into the CSR (262144 int64 at 256k, written by the
pack and read back by the attention) only restates what the page table says. So
the fenced lane builds nothing, and this kernel reads the rows itself.

Which addressing it uses is a compile-time choice, not a branch, because the two
cases do not compile to the same thing. An affine row id (`row0 + n`, what the
allocator gives a fresh long request) is an address the compiler can vectorize,
prefetch arbitrarily far ahead and hand to a bulk copy; a row id loaded from the
page table puts a dependent load in front of every row and is ineligible for the
tensor-descriptor path. `ADDR` selects between them, as `SIDE_POOL` already does
for the scan.

Whether a lane is fenced is read inside the kernel, from the recall count the
compaction already produced: a captured graph cannot skip a launch, so an
unfenced lane exits on one scalar load.

The output is the absorbed-latent context, [H, kv_lora_rank]; the up-projection
stays where it is.
"""

import torch
import triton
import triton.language as tl

ADDR_AFFINE = 0
ADDR_PAGE_TABLE = 1


@triton.jit
def _accumulate(rid, mask, base, q, m, l, acc, sc, ROW: tl.constexpr, KV: tl.constexpr):
    """One block of rows into the running online-softmax state. bf16 operands
    with fp32 accumulation, the stock decode kernel's numerics class."""
    d = tl.arange(0, ROW)
    rows = tl.load(
        base + rid.to(tl.int64)[:, None] * ROW + d[None, :], mask=mask[:, None], other=0.0
    )
    s_hn = tl.dot(q, tl.trans(rows)) * sc
    s_hn = tl.where(mask[None, :], s_hn, float("-inf"))
    m_new = tl.maximum(m, tl.max(s_hn, 1))
    p = tl.exp(s_hn - m_new[:, None])
    alpha = tl.exp(m - m_new)
    l = l * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), rows[:, :KV])
    return m_new, l, acc


@triton.jit
def _dense_mla_split_kernel(
    q_ptr,  # [H, ROW] fp32 absorbed query of this (layer, lane)
    r2t_ptr,  # [R2T] int32 page table (ADDR_PAGE_TABLE) or [1] int32 first row (ADDR_AFFINE)
    kbase_ptr,  # [1] int64 pool base pointer of this layer
    loc_ptr,  # [1] int64 this step's appended pool row; it is the last row attended
    seq_ptr,  # [1] int64 rows attended
    cnt_ptr,  # [1] int32 rows the scan fired; <= W means this lane is not fenced
    om_ptr,  # [NSPLIT, H] fp32 partial max
    ol_ptr,  # [NSPLIT, H] fp32 partial sumexp
    oa_ptr,  # [NSPLIT, H, KV] fp32 partial weighted sum
    sc,
    W,  # recall capacity: the fence is cnt > W
    ROW: tl.constexpr,
    KV: tl.constexpr,
    H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ADDR: tl.constexpr,
):
    s = tl.program_id(0)
    S = tl.num_programs(0)
    h = tl.arange(0, H)
    k = tl.arange(0, KV)
    if tl.load(cnt_ptr) <= W:
        # Not fenced: the CSR path serves this lane. A captured graph launches
        # this kernel anyway, so the exit is one scalar load per program.
        tl.store(om_ptr + s * H + h, float("-inf"))
        tl.store(ol_ptr + s * H + h, 0.0)
        return
    seq = tl.load(seq_ptr)
    base = tl.load(kbase_ptr).to(tl.pointer_type(tl.bfloat16))
    d = tl.arange(0, ROW)
    q = tl.load(q_ptr + h[:, None] * ROW + d[None, :]).to(tl.bfloat16)
    per = (seq + S - 1) // S
    lo = s * per
    hi = tl.minimum(lo + per, seq)
    # The step's own row is in the pool but not yet in the page table, so the
    # program that owns the last position takes it from loc, outside the loop:
    # inside, a per-element select on the row id would defeat the affine case.
    tail = hi == seq
    body = hi - tail.to(tl.int64)
    m = tl.full([H], float("-inf"), dtype=tl.float32)
    l = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, KV], dtype=tl.float32)
    row0 = tl.load(r2t_ptr).to(tl.int64) if ADDR == ADDR_AFFINE else 0
    for n0 in range(lo, body, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        mask = n < body
        if ADDR == ADDR_AFFINE:
            rid = row0 + n
        else:
            rid = tl.load(r2t_ptr + n, mask=mask, other=0)
        m, l, acc = _accumulate(rid, mask, base, q, m, l, acc, sc, ROW=ROW, KV=KV)
    if tail:
        n = tl.arange(0, BLOCK_N)
        m, l, acc = _accumulate(
            tl.load(loc_ptr) + n * 0, n == 0, base, q, m, l, acc, sc, ROW=ROW, KV=KV
        )
    tl.store(om_ptr + s * H + h, m)
    tl.store(ol_ptr + s * H + h, l)
    tl.store(oa_ptr + (s * H + h[:, None]) * KV + k[None, :], acc)


@triton.jit
def _dense_mla_merge_kernel(
    om_ptr, ol_ptr, oa_ptr, out_ptr, NSPLIT: tl.constexpr, H: tl.constexpr, KV: tl.constexpr
):
    h = tl.program_id(0)
    k = tl.arange(0, KV)
    m = tl.full([1], float("-inf"), dtype=tl.float32)
    l = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([KV], dtype=tl.float32)
    for s in range(NSPLIT):
        ms = tl.load(om_ptr + s * H + h)
        ls = tl.load(ol_ptr + s * H + h)
        m_new = tl.maximum(m, ms)
        a_old = tl.exp(m - m_new)
        a_new = tl.exp(ms - m_new)
        acc = acc * a_old + tl.load(oa_ptr + (s * H + h) * KV + k) * a_new
        l = l * a_old + ls * a_new
        m = m_new
    tl.store(out_ptr + h * KV + k, acc / tl.maximum(l, 1e-20))


def page_table_is_affine(page_table, seq) -> bool:
    """Whether the lane's rows are one contiguous run, which is what the
    allocator hands a fresh long request; a shared prefix can break it."""
    rows = page_table[:seq]
    return bool(torch.equal(rows, rows[0] + torch.arange(seq, dtype=rows.dtype, device=rows.device)))


def dense_mla_decode(
    q, page_table, kbase, loc, seq, cnt, *, scale, capacity, addr=ADDR_PAGE_TABLE,
    nsplit=64, block_n=64, out=None,
):
    """Attention of one (layer, lane) over its whole row set, with no index list.
    `cnt` is the recall count; the kernel exits unless it exceeds `capacity`."""
    H, ROW = q.shape
    KV = ROW - 64
    if out is None:
        out = torch.empty(H, KV, dtype=torch.float32, device=q.device)
    om = torch.empty(nsplit, H, dtype=torch.float32, device=q.device)
    ol = torch.empty(nsplit, H, dtype=torch.float32, device=q.device)
    oa = torch.empty(nsplit, H, KV, dtype=torch.float32, device=q.device)
    _dense_mla_split_kernel[(nsplit,)](
        q, page_table, kbase, loc, seq, cnt, om, ol, oa, scale, capacity,
        ROW=ROW, KV=KV, H=H, BLOCK_N=block_n, ADDR=addr, num_warps=8,
    )
    _dense_mla_merge_kernel[(H,)](om, ol, oa, out, NSPLIT=nsplit, H=H, KV=KV, num_warps=4)
    return out
