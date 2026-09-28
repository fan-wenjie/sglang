"""Branch-only tier-2 scan: the decode scan with no sketch.

Branch-only recall scores an archive row on the 64-dim sidecar alone and bounds
the whole content block by Cauchy-Schwarz instead of projecting it onto a
rank-r basis. The sketch is then identically zero, so `tl.dot(c, qk)` is
EXACTLY 0 and dropping it leaves the fire set bit-identical to the served scan
run over a zeroed basis -- which is the form the branch-only quality numbers
were measured in (`SGLANG_DEBUG_VESTIGEKV_BRANCH_ONLY`, which zeroes V and
leaves every load and both dots in place). That flag therefore measures what
branch-only recall RECALLS and not what it costs; this module is the other
half.

What it buys is traffic, which is what the scan is (see scan_kernel.py's
header). Per archive row the served scan reads arch(4) + sidecar(128) +
aidx(4) + csk(128) + rho(4) and writes hit(1); this one reads arch(4) +
sidecar(128) + rho(4) and writes hit(1). 269 -> 137 bytes, a 1.96x ceiling.

Measured at 4 pairs x 131072 rows on SM120: 39.8 -> 20.1 us, 1.99x, at 3581
GB/s against the served scan's 3541 -- the same achieved bandwidth, so the gain
IS the traffic and there is nothing further in this kernel to get. Fire sets
agree exactly at both 0% and 97.9% fired. Registers 124 vs 146, no spill, and
the sidecar's TMA gather survives the fork (mexp/kimi/branch_scan_disasm.py).

A fork, not a constexpr arm inside `_scan_batched_kernel`. The arm exists to
DELETE a load and a dot, and .claude/rules/disassemble-check-for-spills.md is
explicit that such an arm has to be separate code outside the loop: predicated
off, both arms' operands stay live across the loop body and the allocator
spills, which taxes every step rather than the ones the feature fires on. A
separate file also keeps the served kernel's diff empty, so this line cannot
regress the arm the paper's numbers come from.

Block shape is tuned here and not inherited: defaults.SCAN_BLOCK_A records that
128 and 256 rows per block "ran out of registers on SM120", a constraint set by
a kernel holding twice these operands.

What is left to specialize, in order of what it is worth. The scan is now at
the bandwidth its traffic allows, so the remaining work is elsewhere in the
step:

  the prologue. fused_prologue_split still computes qsk -- a rank-r projection
  of the query per pair per head -- and a qres relative to that basis. Under
  branch-only the basis is zero, so qsk is zero and qres is the whole content
  norm; both are computable without the projection. Cost is O(H*r*512) per
  pair per step, independent of archive length, so it matters at short context
  and not at long. This is the next fork, and run() below calls the shared
  prologue by name precisely so it can be swapped for one.

  the tables. self.v ([P, r, 512]) and self.qsk_t ([P, r, H]) are allocated and
  never read here, and update() still fills the sketch row indices. All are
  per-tier-change or per-allocation, not per step, which is why none of them
  was worth touching before the scan.

  the row. rho is fp32 for 4 of the 137 bytes and hit is a byte for 1; a bf16
  rho and a bitmask hit would take ~4% together and cost a format change the
  compaction pass shares.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack
from sglang.srt.layers.attention.vestigekv.fused_prologue import (
    compact_fired,
    fused_prologue_split,
)

# Tuned against this kernel, not inherited from the sketch scan. The product is
# pinned to 1024: that is the compact bucket compact_fired's prefix pass is
# built for, so changing it changes the tail as well as the scan.
#
# Swept over block x warps at 4 pairs x 131072 rows on SM120 (the sweep pins
# block*multi=1024, so multi follows block):
#
#   block  warps   us    GB/s        block  warps   us    GB/s
#      64      2  20.1  3581  <--      128      4  21.0  3412
#      64      4  27.1  2655           256      4  23.1  3105
#      32      2  29.2  2463            16      2  47.6  1510
#
# 2 warps, not the sketch scan's 4: halving the operands halves the shared
# memory (25352 -> 12808 B) and 124 registers at 2 warps still does not spill,
# so the narrower launch buys occupancy the wider one cannot use.
BRANCH_BLOCK_A = 64
BRANCH_MULTI = 16
BRANCH_NUM_WARPS = 2


@triton.jit
def _branch_scan_kernel(
    qside_t_ptr,  # [P, D, H] bf16
    qres_ptr,  # [P, H] fp32: ||q_content||, nothing is projected out
    max1g_ptr,  # [P, H] fp32, +inf where the gate is closed
    side_ptr,  # SIDE_POOL=0: [arena, D] bf16 sidecars (else unused)
    arch_ptr,  # [arena] int32 pool row id per archive row
    kbase_ptr,  # SIDE_POOL>0: [P] int64 pool base for the pair's layer
    rho_ptr,  # [arena] fp32: ||c_u||, the whole content norm
    a_len_ptr,  # [P] int64: real archive rows of this pair
    a_off_ptr,  # [P] int64 arena offset per pair
    cc_ptr,  # [P] fp32 certificate coefficient
    hit_ptr,  # [arena] int8 out (0/1 fired flag)
    counts_ptr,  # [P, NB] int32 fused compact-count out
    Amax,
    sc,
    H: tl.constexpr,
    DD: tl.constexpr,
    SIDE_POOL: tl.constexpr,  # 0 packed table, 1 indirect load, 2 TMA gather
    ROW: tl.constexpr,  # pool row width when SIDE_POOL
    KV_OFF: tl.constexpr,  # sidecar's offset inside the row
    POOL_ROWS: tl.constexpr,  # pool row count, for the TMA descriptor
    BLOCK_A: tl.constexpr,
    MULTI: tl.constexpr,
):
    p = tl.program_id(1)
    al = tl.load(a_len_ptr + p)
    BUCKET = MULTI * BLOCK_A
    nb = (Amax + BUCKET - 1) // BUCKET
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    nb_live = ((al + BUCKET - 1) // BUCKET).to(tl.int32)
    trips = (tl.maximum(nb_live - pid, 0) + G - 1) // G
    if trips <= 0:
        return
    abase = tl.load(a_off_ptr + p).to(tl.int64)
    d = tl.arange(0, DD)
    h = tl.arange(0, H)
    qs = tl.load(qside_t_ptr + p * DD * H + d[:, None] * H + h[None, :])
    cc = tl.load(cc_ptr + p)
    qr = tl.load(qres_ptr + p * H + h)
    m1 = tl.load(max1g_ptr + p * H + h)
    if SIDE_POOL:
        kbase = tl.load(kbase_ptr + p).to(tl.pointer_type(tl.bfloat16))
    if SIDE_POOL == 2:
        sdesc = tl.make_tensor_descriptor(
            kbase,
            shape=[POOL_ROWS, ROW],
            strides=[ROW, 1],
            block_shape=[1, DD],
        )
    for i in range(trips):
        b = pid + i * G
        base = b * BUCKET
        cnt = 0
        for kb in range(MULTI):
            offs = base + kb * BLOCK_A + tl.arange(0, BLOCK_A)
            m = offs < al
            if SIDE_POOL:
                sl = tl.load(arch_ptr + abase + offs, mask=m, other=0)
            if SIDE_POOL == 2:
                s = sdesc.gather(sl, KV_OFF)
            elif SIDE_POOL == 1:
                s = tl.load(
                    kbase + sl.to(tl.int64)[:, None] * ROW + (KV_OFF + d)[None, :],
                    mask=m[:, None],
                    other=0.0,
                )
            else:
                s = tl.load(
                    side_ptr + (abase + offs[:, None]) * DD + d[None, :],
                    mask=m[:, None],
                    other=0.0,
                )
            rh = tl.load(rho_ptr + abase + offs, mask=m, other=0.0)
            # Native-dtype tensor-core dot, fp32 accumulation (scan_kernel.py).
            acc = tl.dot(s, qs).to(tl.float32)
            score = acc * sc + cc * rh[:, None] * qr[None, :]
            fired = tl.max((score > m1[None, :]).to(tl.int32), 1)
            tl.store(hit_ptr + abase + offs, fired.to(tl.int8), mask=m)
            cnt += tl.sum(tl.where(m, fired, 0), 0)
        tl.store(counts_ptr + p * nb + b, cnt)


def branch_scan(pack, P_eff, block=None, multi=None, warps=None):
    """Launch the branch-only scan over a pack's tensors.

    Same contract as the scan inside BatchedScanPack.run: fills pack.hit and
    pack.c_counts for the first P_eff pairs. The block shape is overridable so
    the tuning sweep can drive it without editing the module constants.
    """
    block = BRANCH_BLOCK_A if block is None else block
    multi = BRANCH_MULTI if multi is None else multi
    warps = BRANCH_NUM_WARPS if warps is None else warps
    Am = pack.am_grid
    grid = (min(triton.cdiv(Am, block * multi), D.SCAN_GRID_CAP), P_eff)
    _branch_scan_kernel[grid](
        pack.qside_t,
        pack.qres,
        pack.max1g,
        pack.side,
        pack.arch,
        pack.kbase,
        pack.rho,
        pack.a_len,
        pack.a_off,
        pack.cc,
        pack.hit,
        pack.c_counts,
        Am,
        pack.scale,
        H=pack.q_heads,
        DD=D.SIDECAR_DIM,
        SIDE_POOL=0 if pack.side is not None else pack.side_mode,
        ROW=pack.pool_row or 0,
        KV_OFF=D.KV_LORA_RANK,
        POOL_ROWS=pack._pool_rows or 1,
        BLOCK_A=block,
        MULTI=multi,
        num_warps=warps,
    )


class BranchScanPack(BatchedScanPack):
    """A pack whose step runs the branch-only scan.

    Everything else is inherited unchanged -- construction, fits(), and the
    update() that fills the sketch tables. Those tables cost 4 bytes per
    archive row and are written once per tier change, not per step, and
    leaving them is what keeps this class a pure override: the scan simply
    never reads them, which is where the traffic is.
    """

    def run(self, p_live=None):
        P_eff = p_live if p_live is not None else self.li.shape[0]
        fused_prologue_split(
            self.qbuf,
            self.li,
            self.slot,
            self.kr,
            self.v,
            self.nk_len,
            self.thr_flat,
            self.scale,
            out=(self.max1g, self.qside_t, self.qsk_t, self.qres),
            margin=self.margin,
            ent_gain=self.ent_gain,
            thr_lse=self.thr_lse,
            partials=(self.pm, self.ps, self.pt),
            kslot=self.kslot,
            kbase=self.kbase,
            nkm=self.nkm,
            row=self.pool_row,
            pool_rows=self._pool_rows,
            mode=self.pool_mode,
            a_len=self.a_len,
        )
        branch_scan(self, P_eff)
        compact_fired(
            self.hit,
            self.arch,
            self.a_len,
            self.a_off,
            self.li,
            self.slot,
            self.fetch_buf,
            self.fetch_len,
            self.fetch_ovf,
            self.ovf_count,
            (self.c_counts, self.c_offsets, self.c_total),
            self.am_grid,
            fence_rows=self.fence_rows,
            rand_fence=self.rand_coins,
        )
