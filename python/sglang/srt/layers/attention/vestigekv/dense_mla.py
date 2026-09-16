# SPDX-License-Identifier: Apache-2.0
"""Dense MLA decode straight off the page table, for a lane the recall fenced.

A fenced lane attends its whole row set. Today that set is copied into the CSR
(262144 int64 at 256k, written by the pack and read back by the attention) and
the stock decode kernel reads it from there. The copy is pure transport: the row
ids already exist in `req_to_token`, and the rows themselves are in the pool.
This computes the same output reading the page table directly, so the fenced
path costs the rows it must read and nothing else.

Same math as the stock path and the same indirection class (a pool row id per
row); what disappears is one materialization of the id list per fenced layer and
step. Flash decoding: each program owns a contiguous slice of the row range and
returns an online-softmax partial (max, sumexp, weighted sum), and the merge
combines them per head.

The output is the absorbed-latent context, [H, kv_lora_rank]; the up-projection
stays where it is.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dense_mla_split_kernel(
    q_ptr,  # [H, ROW] fp32 absorbed query of this (layer, lane)
    r2t_ptr,  # [R2T] int32 page table of the lane's pool row ids
    kbase_ptr,  # [1] int64 pool base pointer of this layer (see _prologue_scores_kernel)
    loc_ptr,  # [1] int64 this step's appended pool row; it replaces r2t[seq-1]
    seq_ptr,  # [1] int64 rows attended
    om_ptr,  # [NSPLIT, H] fp32 partial max
    ol_ptr,  # [NSPLIT, H] fp32 partial sumexp
    oa_ptr,  # [NSPLIT, H, KV] fp32 partial weighted sum
    sc,  # attention scale
    ROW: tl.constexpr,  # 576 = KV + rope
    KV: tl.constexpr,  # 512, the part that is summed into the output
    H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    s = tl.program_id(0)
    S = tl.num_programs(0)
    seq = tl.load(seq_ptr)
    loc = tl.load(loc_ptr)
    base = tl.load(kbase_ptr).to(tl.pointer_type(tl.bfloat16))
    h = tl.arange(0, H)
    d = tl.arange(0, ROW)
    k = tl.arange(0, KV)
    q = tl.load(q_ptr + h[:, None] * ROW + d[None, :]).to(tl.bfloat16)  # [H, ROW]
    # even slices of the row range; the last program takes the remainder
    per = (seq + S - 1) // S
    lo = s * per
    hi = tl.minimum(lo + per, seq)
    m = tl.full([H], float("-inf"), dtype=tl.float32)
    l = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, KV], dtype=tl.float32)
    for n0 in range(lo, hi, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        mask = n < hi
        rid = tl.load(r2t_ptr + n, mask=mask, other=0)
        # the step's own row is appended to the pool but not yet in the table
        rid = tl.where(n == seq - 1, loc.to(rid.dtype), rid)
        # bf16 operands with fp32 accumulation, the stock decode kernel's
        # numerics class; an fp32 dot would silently run tf32 here.
        rows = tl.load(
            base + rid.to(tl.int64)[:, None] * ROW + d[None, :],
            mask=mask[:, None],
            other=0.0,
        )  # [BLOCK_N, ROW] bf16
        s_hn = tl.dot(q, tl.trans(rows)) * sc  # [H, BLOCK_N]
        s_hn = tl.where(mask[None, :], s_hn, float("-inf"))
        m_new = tl.maximum(m, tl.max(s_hn, 1))
        p = tl.exp(s_hn - m_new[:, None])
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), rows[:, :KV])
        m = m_new
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


def dense_mla_decode(q, page_table, kbase, loc, seq, *, scale, nsplit=64, block_n=64, out=None):
    """Attention of one (layer, lane) over its whole row set, read from the page
    table. `q` is [H, ROW] fp32 absorbed; `kbase` a 1-element int64 tensor holding
    the layer's pool base pointer; `loc`/`seq` 1-element int64 tensors."""
    H, ROW = q.shape
    KV = ROW - 64
    if out is None:
        out = torch.empty(H, KV, dtype=torch.float32, device=q.device)
    om = torch.empty(nsplit, H, dtype=torch.float32, device=q.device)
    ol = torch.empty(nsplit, H, dtype=torch.float32, device=q.device)
    oa = torch.empty(nsplit, H, KV, dtype=torch.float32, device=q.device)
    _dense_mla_split_kernel[(nsplit,)](
        q, page_table, kbase, loc, seq, om, ol, oa, scale,
        ROW=ROW, KV=KV, H=H, BLOCK_N=block_n, num_warps=8,
    )
    _dense_mla_merge_kernel[(H,)](om, ol, oa, out, NSPLIT=nsplit, H=H, KV=KV, num_warps=4)
    return out
