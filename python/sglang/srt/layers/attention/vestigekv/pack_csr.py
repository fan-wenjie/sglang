"""Two-kernel CSR pack for the graph-replayed decode step (all layers per launch).

Forked from create_flashinfer_kv_indices_triton (kernels/ops/kvcache/
kv_indices.py): programs loop row blocks with runtime bounds -- no
capacity-sized grid, so launch cost does not scale with max_context_len.
Replaces the ~12-node torch chain _pack_csr ran per layer (~84 graph nodes,
~70 us/step); the torch chain still serves the eager (uncaptured) step.

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

Recall-overflow fence (FENCE): a lane whose fetch_ovf flag is set (the
compaction found more fired rows than the fetch buffer holds) is packed as
its full row set instead -- req_to_token[slot, :seq] with this step's own
slot written at seq - 1 -- so the step attends densely rather than a
truncated fetch. The kept-table append still happens; only what this step
attends changes. With FENCE off the kernels are the pre-fence form.

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
    seq_ptr,  # [bs] int64 this step's seq_len per lane (FENCE only)
    kept_buf_ptr,  # [L, R1, CAP]
    kept_len_ptr,  # [L, R1] int32
    fetch_len_ptr,  # [L, R1] int32
    fetch_ovf_ptr,  # [L, R1] int32 overflow flag (FENCE only)
    affine_ok_ptr,  # [R1] int32 in/out: page table still one run (FENCE only)
    affine_base_ptr,  # [R1] int64 where it starts (FENCE only)
    indptr_ptr,  # [L, MAXBS1] out
    R1,
    CAP,
    bs,
    L,
    MAXBS1,
    FENCE: tl.constexpr,
    AFFINE_TRACK: tl.constexpr,
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
        run = tl.load(kept_len_ptr).to(tl.int64) * 0  # int64 scalar zero
        for i in range(0, bs):
            slot = tl.load(slots_ptr + i).to(tl.int64)
            n = tl.load(kept_len_ptr + li * R1 + slot).to(tl.int64) + 1
            n += tl.load(fetch_len_ptr + li * R1 + slot).to(tl.int64)
            if FENCE:
                fenced = tl.load(fetch_ovf_ptr + li * R1 + slot) != 0
                n = tl.where(fenced, tl.load(seq_ptr + i).to(tl.int64), n)
            run += n
            tl.store(indptr_ptr + li * MAXBS1 + i + 1, run)
        k = bs + 1 + tl.arange(0, 64)
        tl.store(indptr_ptr + li * MAXBS1 + k, run, mask=k < MAXBS1)
        for i in range(0, bs):  # appends (kept_len still pristine)
            slot = tl.load(slots_ptr + i).to(tl.int64)
            n_old = tl.load(kept_len_ptr + li * R1 + slot).to(tl.int64)
            tl.store(
                kept_buf_ptr + (li * R1 + slot) * CAP + n_old, tl.load(loc_ptr + i)
            )
        if AFFINE_TRACK:
            # The per-step half of the affine guarantee. An AFFINE build bakes
            # "a fenced lane's rows are base .. base+seq-1" into the kernel, so
            # the moment a lane's row set stops being that run its fence flag is
            # cleared here: the lane truncates to its fired rows for this step,
            # which loses rows but never reads rows that belong to someone else.
            # Per slot and per layer, because the flag it clears is per layer.
            for i in range(0, bs):
                slot = tl.load(slots_ptr + i).to(tl.int64)
                seq_i = tl.load(seq_ptr + i).to(tl.int64)
                base = tl.load(affine_base_ptr + slot)
                still = tl.load(affine_ok_ptr + slot) != 0
                still = still and (
                    tl.load(loc_ptr + i).to(tl.int64) == base + seq_i - 1
                )
                if li == 0:
                    tl.store(affine_ok_ptr + slot, still.to(tl.int32))
                if not still:
                    tl.store(fetch_ovf_ptr + li * R1 + slot, 0)
        for i in range(0, bs):  # lengths: first occurrence only (+1 once,
            slot = tl.load(slots_ptr + i).to(tl.int64)  # duplicates skip)
            dup = 0
            for jj in range(0, i):
                dup += (tl.load(slots_ptr + jj).to(tl.int64) == slot).to(tl.int32)
            if dup == 0:
                n_old = tl.load(kept_len_ptr + li * R1 + slot)
                tl.store(kept_len_ptr + li * R1 + slot, n_old + 1)


def pack_csr_all_layers(
    slots,
    loc,
    kept_buf,
    kept_len,
    fetch_len,
    fetch_buf,
    indices,
    indptr,
    *,
    seq=None,
    fetch_ovf=None,
    req_to_token=None,
    affine_ok=None,
    affine_base=None,
):
    """One launch for every layer: the per-lane row counts. Stacked tensors: kept_buf
    [L, R1, CAP], kept_len/fetch_len [L, R1], fetch_buf [L, R1, FW],
    indices [L, CAPI], indptr [L, MAXBS1]; slots/loc [bs].

    Passing `seq` [bs], `fetch_ovf` [L, R1] and `req_to_token` [reqs, ctx]
    together arms the overflow fence (see the module docstring); all three
    or none.

    `indices` is accepted but never written: decode stage 1 reads a lane's
    rows from the tiers, so the only output consumed here is the per-lane
    count in `indptr`, which stage 2 sizes its reduction from."""
    fence = seq is not None
    if fence != (fetch_ovf is not None) or fence != (req_to_token is not None):
        raise ValueError("the fence needs seq, fetch_ovf and req_to_token together")
    if not fence:
        # Unused pointer arguments still have to be tensors.
        seq, fetch_ovf, req_to_token = loc, fetch_len, fetch_len
    track = affine_ok is not None
    if not track:
        # Unused pointer arguments still have to be tensors; the block that
        # would write through them is compiled out, so aliasing a real one
        # here would be a silent corruption rather than a dead store.
        affine_ok, affine_base = fetch_len, loc
    L, R1, CAP = kept_buf.shape
    bs = slots.shape[0]
    _pack_csr_prep_kernel[(L,)](
        slots,
        loc,
        seq,
        kept_buf,
        kept_len,
        fetch_len,
        fetch_ovf,
        affine_ok,
        affine_base,
        indptr,
        R1,
        CAP,
        bs,
        L,
        indptr.shape[1],
        FENCE=fence,
        AFFINE_TRACK=track,
    )
