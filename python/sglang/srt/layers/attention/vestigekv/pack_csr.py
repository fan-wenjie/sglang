"""Two-kernel CSR pack for the in-graph scan (all layers per launch).

Forked from create_flashinfer_kv_indices_triton (kernels/ops/kvcache/
kv_indices.py): programs loop row blocks with runtime bounds -- no
capacity-sized grid, so launch cost does not scale with max_context_len.
Replaces the ~12-node torch chain _pack_csr ran per layer (~84 graph nodes,
~70 us/step).

Split mirrors the ancestor's contract (indptr is an input to the gather):
- PREP (one program): walks every (layer, lane) once; applies this step's
  append (kept_buf[slot, n] = loc, kept_len += 1), then writes each layer's
  indptr row from the running prefix. Single program = the only writer of
  kept_len/indptr this step, so the gather kernel reads a settled state --
  the cross-program visibility race a fused form would have is structural,
  not a tuning matter (bs <= 32 and L <= 7 keep the walk trivial).
- PACK (grid L x bs): pure gather, the ancestor's shape. Row j of a lane:
  j < kept_len -> kept_buf[slot, j], else fetch_buf[slot, j - kept_len].
  Masked stores replace the torch form's trash-slot scatter.

Storage is the stacked form ([L, R1, CAP] etc.); the backend's per-lid dict
entries are views into it (the _qbuf_stack pattern), so eager call sites are
untouched. Bit-identical to the torch _pack_csr; a registered test asserts
it on shared inputs.
"""

import triton
import triton.language as tl


@triton.jit
def _pack_csr_prep_kernel(
    slots_ptr,  # [bs] int64 pool slots (padded lanes -> trash slot)
    loc_ptr,  # [bs] this step's appended pool row per lane
    kept_buf_ptr,  # [L, R1, CAP]
    kept_len_ptr,  # [L, R1] int64
    fetch_len_ptr,  # [L, R1] int64
    indptr_ptr,  # [L, MAXBS1] out
    R1,
    CAP,
    bs,
    L,
    MAXBS1,
):
    # One program PER LAYER (layers share no state), three passes each
    # reading only PRE-STEP state:
    # duplicate slots (shared trash lanes) must mirror the torch scatter --
    # every colliding lane appends at the same n_old (last write wins) and
    # the length advances by one exactly once. A single mutate-as-you-walk
    # loop would let a later duplicate read the earlier lane's update.
    li = tl.program_id(0)
    if li < L:
        tl.store(indptr_ptr + li * MAXBS1, 0)
        run = tl.load(kept_len_ptr) * 0  # int64 scalar zero
        for i in range(0, bs):
            slot = tl.load(slots_ptr + i).to(tl.int64)
            run += tl.load(kept_len_ptr + li * R1 + slot) + 1
            run += tl.load(fetch_len_ptr + li * R1 + slot)
            tl.store(indptr_ptr + li * MAXBS1 + i + 1, run)
        k = bs + 1 + tl.arange(0, 64)
        tl.store(indptr_ptr + li * MAXBS1 + k, run, mask=k < MAXBS1)
        for i in range(0, bs):  # appends (kept_len still pristine)
            slot = tl.load(slots_ptr + i).to(tl.int64)
            n_old = tl.load(kept_len_ptr + li * R1 + slot)
            tl.store(
                kept_buf_ptr + (li * R1 + slot) * CAP + n_old, tl.load(loc_ptr + i)
            )
        for i in range(0, bs):  # lengths: first occurrence only (+1 once,
            slot = tl.load(slots_ptr + i).to(tl.int64)  # duplicates skip)
            dup = 0
            for jj in range(0, i):
                dup += (tl.load(slots_ptr + jj).to(tl.int64) == slot).to(tl.int32)
            if dup == 0:
                n_old = tl.load(kept_len_ptr + li * R1 + slot)
                tl.store(kept_len_ptr + li * R1 + slot, n_old + 1)


@triton.jit
def _pack_csr_gather_kernel(
    slots_ptr,
    kept_buf_ptr,  # [L, R1, CAP]
    kept_len_ptr,  # [L, R1] (post-append)
    fetch_len_ptr,  # [L, R1]
    fetch_buf_ptr,  # [L, R1, FW]
    indptr_ptr,  # [L, MAXBS1] (from prep)
    indices_ptr,  # [L, CAPI] out
    R1,
    CAP,
    FW,
    CAPI,
    MAXBS1,
    BLOCK: tl.constexpr,
):
    li = tl.program_id(0)
    lane = tl.program_id(1)
    slot = tl.load(slots_ptr + lane).to(tl.int64)
    start = tl.load(indptr_ptr + li * MAXBS1 + lane)
    n = tl.load(kept_len_ptr + li * R1 + slot)
    f_len = tl.load(fetch_len_ptr + li * R1 + slot)
    lens_tot = n + f_len
    num_loop = tl.cdiv(lens_tot, BLOCK)
    for i in range(num_loop):
        j = i * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        m = j < lens_tot
        kept = tl.load(
            kept_buf_ptr + (li * R1 + slot) * CAP + j, mask=m & (j < n), other=0
        )
        fired = tl.load(
            fetch_buf_ptr + (li * R1 + slot) * FW + (j - n),
            mask=m & (j >= n),
            other=0,
        )
        tl.store(
            indices_ptr + li * CAPI + start + j, tl.where(j < n, kept, fired), mask=m
        )


def pack_csr_all_layers(
    slots, loc, kept_buf, kept_len, fetch_len, fetch_buf, indices, indptr
):
    """Two launches for every layer's CSR. Stacked tensors: kept_buf
    [L, R1, CAP], kept_len/fetch_len [L, R1], fetch_buf [L, R1, FW],
    indices [L, CAPI], indptr [L, MAXBS1]; slots/loc [bs]."""
    L, R1, CAP = kept_buf.shape
    bs = slots.shape[0]
    _pack_csr_prep_kernel[(L,)](
        slots,
        loc,
        kept_buf,
        kept_len,
        fetch_len,
        indptr,
        R1,
        CAP,
        bs,
        L,
        indptr.shape[1],
    )
    _pack_csr_gather_kernel[(L, bs)](
        slots,
        kept_buf,
        kept_len,
        fetch_len,
        fetch_buf,
        indptr,
        indices,
        R1,
        CAP,
        fetch_buf.shape[2],
        indices.shape[1],
        indptr.shape[1],
        BLOCK=512,
    )
