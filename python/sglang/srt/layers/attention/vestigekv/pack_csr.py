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
    own_ptr,  # SHARDED: [bs] int32, 1 where this rank holds the step's new row
    sh_off,  # SHARDED: this rank's first global position
    sh_stride,  # SHARDED: positions between two rows this rank holds
    indptr_ptr,  # [L, MAXBS1] out
    R1,
    CAP,
    bs,
    L,
    MAXBS1,
    FENCE: tl.constexpr,
    SHARDED: tl.constexpr,
    TOPK: tl.constexpr = 0,
    KPOOL: tl.constexpr = 1,
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
            grew = 1
            if SHARDED:
                # The step's new row lands in exactly one rank's pool; the
                # others must not count a row they do not hold.
                grew = tl.load(own_ptr + i).to(tl.int64)
            n = tl.load(kept_len_ptr + li * R1 + slot).to(tl.int64) + grew
            n += tl.load(fetch_len_ptr + li * R1 + slot).to(tl.int64)
            if FENCE:
                fenced = tl.load(fetch_ovf_ptr + li * R1 + slot) != 0
                seq = tl.load(seq_ptr + i).to(tl.int64)
                if SHARDED:
                    seq = tl.cdiv(seq - sh_off, sh_stride)
                if TOPK > 0:
                    # DSA fallback: a fenced lane attends what DSA attends
                    # (dsa/utils.compute_dsa_seqlens), not its whole page table.
                    tail = seq % KPOOL
                    seq = tl.minimum(seq - tail, TOPK) + tail
                n = tl.where(fenced, seq, n)
            run += n
            tl.store(indptr_ptr + li * MAXBS1 + i + 1, run)
        k = bs + 1 + tl.arange(0, 64)
        tl.store(indptr_ptr + li * MAXBS1 + k, run, mask=k < MAXBS1)
        for i in range(0, bs):  # appends (kept_len still pristine)
            slot = tl.load(slots_ptr + i).to(tl.int64)
            own = 1
            if SHARDED:
                own = tl.load(own_ptr + i).to(tl.int32)
            if own != 0:
                n_old = tl.load(kept_len_ptr + li * R1 + slot).to(tl.int64)
                tl.store(
                    kept_buf_ptr + (li * R1 + slot) * CAP + n_old, tl.load(loc_ptr + i)
                )
        for i in range(0, bs):  # lengths: first occurrence only (+1 once,
            slot = tl.load(slots_ptr + i).to(tl.int64)  # duplicates skip)
            dup = 0
            for jj in range(0, i):
                dup += (tl.load(slots_ptr + jj).to(tl.int64) == slot).to(tl.int32)
            own = 1
            if SHARDED:
                own = tl.load(own_ptr + i).to(tl.int32)
            if dup == 0 and own != 0:
                n_old = tl.load(kept_len_ptr + li * R1 + slot)
                tl.store(kept_len_ptr + li * R1 + slot, n_old + 1)


@triton.jit
def _pack_csr_gather_kernel(
    slots_ptr,
    loc_ptr,  # [bs] (FENCE only: the dense row set ends with this step's slot)
    seq_ptr,  # [bs] int64 (FENCE only)
    kept_buf_ptr,  # [L, R1, CAP]
    kept_len_ptr,  # [L, R1] (post-append)
    fetch_len_ptr,  # [L, R1]
    fetch_buf_ptr,  # [L, R1, FW]
    fetch_ovf_ptr,  # [L, R1] int32 (FENCE only)
    r2t_ptr,  # [R1 - 1, R2T] req_to_token (FENCE only)
    sh_off,  # SHARDED: this rank's first global position
    sh_stride,  # SHARDED: positions between two rows this rank holds
    indptr_ptr,  # [L, MAXBS1] (from prep)
    indices_ptr,  # [L, CAPI] out
    R1,
    CAP,
    FW,
    CAPI,
    MAXBS1,
    R2T,
    BLOCK: tl.constexpr,
    FENCE: tl.constexpr,
    SHARDED: tl.constexpr,
    TOPK: tl.constexpr = 0,
    KPOOL: tl.constexpr = 1,
):
    li = tl.program_id(0)
    lane = tl.program_id(1)
    slot = tl.load(slots_ptr + lane).to(tl.int64)
    start = tl.load(indptr_ptr + li * MAXBS1 + lane).to(tl.int64)
    n = tl.load(kept_len_ptr + li * R1 + slot).to(tl.int64)
    # One program-uniform branch per lane: the fenced loop is a separate
    # region, so an unfenced lane runs exactly the pre-fence loop (no extra
    # masks or selects) and pays one scalar load for the flag.
    fenced = False
    if FENCE:
        fenced = tl.load(fetch_ovf_ptr + li * R1 + slot) != 0
    if fenced:
        seq = tl.load(seq_ptr + lane).to(tl.int64)
        loc = tl.load(loc_ptr + lane)
        # `seq` is the request's GLOBAL length; a rank walks the positions it
        # holds, so the rows it emits are cdiv(seq - off, stride) of them and
        # the step's own row belongs to whichever rank owns position seq - 1.
        rows = seq
        if SHARDED:
            rows = tl.cdiv(seq - sh_off, sh_stride)
        if TOPK > 0:
            # Placeholder page-table rows within the indptr segment the prep
            # sized; the fork reads the indexer's rows instead (VK_TOPK).
            tail = rows % KPOOL
            rows = tl.minimum(rows - tail, TOPK) + tail
        for i in range(tl.cdiv(rows, BLOCK)):
            j = i * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            m = j < rows
            gj = j
            if SHARDED:
                gj = sh_off + j * sh_stride
            dense = tl.load(r2t_ptr + slot * R2T + gj, mask=m, other=0)
            dense = tl.where(gj == seq - 1, loc.to(dense.dtype), dense)
            tl.store(
                indices_ptr + li * CAPI + start + j,
                dense.to(indices_ptr.dtype.element_ty),
                mask=m,
            )
    else:
        f_len = tl.load(fetch_len_ptr + li * R1 + slot).to(tl.int64)
        lens_tot = n + f_len
        for i in range(tl.cdiv(lens_tot, BLOCK)):
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
                indices_ptr + li * CAPI + start + j,
                tl.where(j < n, kept, fired).to(indices_ptr.dtype.element_ty),
                mask=m,
            )


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
    own=None,
    shard=None,
    topk=0,
    kpool=1,
):
    """Two launches for every layer's CSR. Stacked tensors: kept_buf
    [L, R1, CAP], kept_len/fetch_len [L, R1], fetch_buf [L, R1, FW],
    indices [L, CAPI], indptr [L, MAXBS1]; slots/loc [bs].

    Passing `seq` [bs], `fetch_ovf` [L, R1] and `req_to_token` [reqs, ctx]
    together arms the overflow fence (see the module docstring); all three
    or none.

    `own` [bs] int32 and `shard` (offset, stride) arm the sharded form, where
    this rank holds only the positions `offset, offset+stride, ...` of each
    request. `own[i]` says whether the step's new row for lane i landed in
    this rank's pool -- exactly one rank's does -- and the fence walks the
    page table with the same stride rather than its whole length. Both or
    neither; without them the kernels are the single-holder ones."""
    fence = seq is not None
    sharded = own is not None
    if sharded != (shard is not None):
        raise ValueError("the sharded form needs own and shard together")
    sh_off, sh_stride = shard if sharded else (0, 1)
    if fence != (fetch_ovf is not None) or fence != (req_to_token is not None):
        raise ValueError("the fence needs seq, fetch_ovf and req_to_token together")
    if not fence:
        # Unused pointer arguments still have to be tensors.
        seq, fetch_ovf, req_to_token = loc, fetch_len, fetch_len
    if not sharded:
        own = loc
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
        own,
        sh_off,
        sh_stride,
        indptr,
        R1,
        CAP,
        bs,
        L,
        indptr.shape[1],
        FENCE=fence,
        SHARDED=sharded,
        TOPK=topk,
        KPOOL=max(1, kpool),
    )
    _pack_csr_gather_kernel[(L, bs)](
        slots,
        loc,
        seq,
        kept_buf,
        kept_len,
        fetch_len,
        fetch_buf,
        fetch_ovf,
        req_to_token,
        sh_off,
        sh_stride,
        indptr,
        indices,
        R1,
        CAP,
        fetch_buf.shape[2],
        indices.shape[1],
        indptr.shape[1],
        req_to_token.shape[1] if fence else 0,
        BLOCK=512,
        FENCE=fence,
        SHARDED=sharded,
        TOPK=topk,
        KPOOL=max(1, kpool),
    )
