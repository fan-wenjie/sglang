"""Cross-(layer, request) batched tier-2 step.

The per-(layer, slot) form of query_fixed launches ~36 kernels per call -- with
five local MLA layers that is 180 graph nodes whose total GPU time is 0.53 ms
but whose per-node dispatch floor sets both the replay time and the ~40 ms it
costs to capture the graph. Every one of those calls has the same shape modulo
a few dozen rows, so the whole step batches: stack the pairs' parameters once
(padded to the widest pair), then run ONE set of batched ops. Measured on the
serving shape: 180 kernels -> 34, replay GPU 0.53 -> 0.31 ms, bit-identical
fetch sets.

Padding is masked, not trusted: a padded kept row would otherwise win max1 with
its zero score, and a padded archive row would fire on it. nk_len masks the
former to -inf before the max; a_len bounds the scan kernel's row mask.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.defaults import ieee_fp32
from sglang.srt.layers.attention.vestigekv.fused_prologue import (
    _NSPLIT,
    compact_fired,
    fused_prologue_split,
)


@triton.jit
def _scan_batched_kernel(
    qside_t_ptr,  # [P, D, H] fp32
    qsk_t_ptr,  # [P, R, H] fp32
    qres_ptr,  # [P, H] fp32
    max1g_ptr,  # [P, H] fp32, +inf where the gate is closed
    side_ptr,  # SIDE_POOL=0: [arena, D] bf16 sidecars (else unused)
    arch_ptr,  # [arena] int32 pool row id per archive row
    kbase_ptr,  # SIDE_POOL>0: [P] int64 pool base for the pair's layer
    csk_ptr,  # CSK_TIER=0: [arena, R] fp16 packed copy (else unused)
    aidx_ptr,  # CSK_TIER>0: [arena] int32 row of the tier's closed-prefix cache
    cbase_ptr,  # CSK_TIER>0: [P] int64 base of that cache, per pair
    rho_ptr,  # [P, Amax] fp32
    a_len_ptr,  # [P] int64: real archive rows of this pair
    a_off_ptr,  # [P] int64 arena offset per pair
    cc_ptr,  # [P] fp32: zp * sc / sqrt(kv_lora - R)
    hit_ptr,  # [P, Amax] int8 out (0/1 fired flag)
    counts_ptr,  # [P, NB] int32 fused compact-count out (NB = ceil(Amax/1024))
    Amax,
    sc,
    H: tl.constexpr,
    DD: tl.constexpr,
    R: tl.constexpr,
    SIDE_POOL: tl.constexpr,  # 0 packed table, 1 indirect load, 2 TMA gather
    CSK_TIER: tl.constexpr,  # same, for the sketch projections
    CSK_ROWS: tl.constexpr,  # rows in the tier cache, for the TMA descriptor
    ROW: tl.constexpr,  # pool row width when SIDE_POOL
    KV_OFF: tl.constexpr,  # sidecar's offset inside the row
    POOL_ROWS: tl.constexpr,  # pool row count, for the TMA descriptor
    BLOCK_A: tl.constexpr,
    MULTI: tl.constexpr,
):
    p = tl.program_id(1)
    al = tl.load(a_len_ptr + p)
    # Grid is capacity-sized (the capture bakes it) but capped at SCAN_GRID_CAP
    # programs per pair: program pid covers buckets pid, pid+G, ... by a
    # grid-stride loop, so worst-case coverage is unchanged while the dispatch
    # floor (programs fully past a_len exit after one scalar load) shrinks
    # ~8x. Each bucket is one MULTI*BLOCK_A=1024-row compact unit; the fused
    # compact count is a plain per-bucket store (the prefix kernel's
    # pre-zeroing covers buckets nobody ran).
    BUCKET = MULTI * BLOCK_A
    nb = (Amax + BUCKET - 1) // BUCKET
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    nb_live = ((al + BUCKET - 1) // BUCKET).to(tl.int32)
    trips = (tl.maximum(nb_live - pid, 0) + G - 1) // G
    if trips <= 0:
        return
    # Archive rows live in ONE per-layer arena, not in a per-pair slab: each
    # pair's rows start at a_off[p]. The arena is sized by what the KV pool
    # can physically hold (sum of live contexts) rather than by
    # max_bs x max_context, which over-provisions badly at long context
    # (2 x 512k slabs for a pool that holds 576k tokens). The graph bakes the
    # arena base and the offset-table pointer; the offset VALUES live in
    # device memory and are refreshed by update(), exactly like a_len.
    abase = tl.load(a_off_ptr + p).to(tl.int64)
    d = tl.arange(0, DD)
    r = tl.arange(0, R)
    h = tl.arange(0, H)
    # pair-invariant operands load once per program, not per sub-block
    qs = tl.load(qside_t_ptr + p * DD * H + d[:, None] * H + h[None, :])
    qk = tl.load(qsk_t_ptr + p * R * H + r[:, None] * H + h[None, :])
    cc = tl.load(cc_ptr + p)
    qr = tl.load(qres_ptr + p * H + h)
    m1 = tl.load(max1g_ptr + p * H + h)
    if SIDE_POOL:
        kbase = tl.load(kbase_ptr + p).to(tl.pointer_type(tl.bfloat16))
    if CSK_TIER:
        # The sketch projections live in the tier's closed-prefix cache, and
        # the archive is a selection over it -- so the pack carries 4 bytes of
        # row index per archived row instead of a 128-byte copy of the row.
        # The cache is reallocated by torch.cat at every block close, which is
        # why its address is read from a device table refreshed by update()
        # rather than baked into the graph.
        cbase = tl.load(cbase_ptr + p).to(tl.pointer_type(tl.float16))
    if CSK_TIER == 2:
        cdesc = tl.make_tensor_descriptor(
            cbase,
            shape=[CSK_ROWS, R],
            strides=[R, 1],
            block_shape=[1, R],
        )
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
            # quantized storage, fp32 ieee arithmetic (see scan_kernel)
            if SIDE_POOL:
                # The sidecar is the un-roped tail of an archived pool row and arch
                # already holds that row's id, so the packed table duplicates it --
                # 40 MiB of the pack's 84 at 64k bs1. Reading it back through the id
                # is a data-dependent access, which the pipeliner will not stage
                # (see fused_prologue._pool_read_mode); the TMA form is the one that
                # keeps it asynchronous.
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
            if CSK_TIER:
                ai = tl.load(aidx_ptr + abase + offs, mask=m, other=0)
            if CSK_TIER == 2:
                c = cdesc.gather(ai, 0)
            elif CSK_TIER == 1:
                c = tl.load(
                    cbase + ai.to(tl.int64)[:, None] * R + r[None, :],
                    mask=m[:, None],
                    other=0.0,
                )
            else:
                c = tl.load(
                    csk_ptr + (abase + offs[:, None]) * R + r[None, :],
                    mask=m[:, None],
                    other=0.0,
                )
            rh = tl.load(rho_ptr + abase + offs, mask=m, other=0.0)
            # ieee, not tf32: the fast path disagreed with the eager fire set on
            # 6 rows in 58900, a silent change to which rows the model attends.
            # native-dtype tensor-core dots, fp32 accumulation (scan_kernel.py)
            acc = tl.dot(s, qs).to(tl.float32) + tl.dot(c, qk).to(tl.float32)
            score = acc * sc + cc * rh[:, None] * qr[None, :]
            fired = tl.max((score > m1[None, :]).to(tl.int32), 1)
            tl.store(hit_ptr + abase + offs, fired.to(tl.int8), mask=m)
            cnt += tl.sum(tl.where(m, fired, 0), 0)
        # Fused compact count: the bucket index is the store slot (pre-zeroing
        # by the prefix kernel covers buckets no program ran).
        tl.store(counts_ptr + p * nb + b, cnt)


class BatchedScanPack:
    """The (layer, slot) pairs of one captured step, stacked and padded.

    Built outside capture (stacking copies and syncs are legal there), then
    `run()` is pure fixed-address tensor ops and is what the graph captures.
    """

    # Extra rows allocated beyond the founding request's sizes, so the pack
    # (and the graph captured over it) can be REUSED for later requests whose
    # archive fits: an in-place update() costs ~0.03 ms where a recapture
    # costs ~30-55 ms -- the dominant per-request fixed cost once builds went
    # async.
    HEADROOM = 1.05

    def __init__(self, pairs, tiers, qbuf, fetch_buf, fetch_len, q_heads):
        # pairs: list of (lid, slot); tiers: matching RecallTier list.
        dev = tiers[0].arch.device
        self.q_heads = q_heads
        self._pad_slot = None  # legacy packs are always fully occupied
        P = len(tiers)
        # Floor to 1: an all-empty-kept capture (every pair's tier-1 kept the
        # empty set -- the re-prefill anomaly) would otherwise allocate a
        # zero-width kr and crash skept.max(-1). One padded row, masked off by
        # nk_len=0, keeps the reduction well-formed; the empty pair is served
        # as full-archive recall below.
        NKm = max(1, int(max(t.kept_rows.shape[0] for t in tiers) * self.HEADROOM))
        Am = max(1, int(max(t.arch.shape[0] for t in tiers) * self.HEADROOM))
        r = tiers[0].r
        self.kr = torch.zeros(P, NKm, D.LATENT_DIM, device=dev, dtype=torch.bfloat16)
        self.nkm = NKm
        self.kslot = self.kbase = self._pool_bases = None
        self.pool_row = self._pool_rows = self.pool_mode = None
        self.side_mode = 0
        self.aidx = self.cbase = None
        self.csk_mode = 0
        self.rank = r
        self._csk_rows = 0
        self._csk_refs = []
        self.v = torch.zeros(P, r, D.KV_LORA_RANK, device=dev)
        # Archive tables are ONE arena, addressed by a_off[p] (see the scan
        # kernel): `arena` rows total instead of P x Am, which at long context
        # is what the pool can hold rather than max_bs x max_context.
        arena = P * Am  # eager path: one contiguous slab, offsets are p*Am
        self.am_grid = Am
        self.arena = arena
        self.side = torch.zeros(arena, D.SIDECAR_DIM, device=dev, dtype=torch.bfloat16)
        self.csk = torch.zeros(arena, r, device=dev, dtype=torch.float16)
        self.rho = torch.zeros(arena, device=dev)
        self.a_off = torch.arange(P, dtype=torch.int64, device=dev) * Am
        # int32: archive entries are pool row indices, bounded by max_total_tokens
        # (~2M), and the fetch buffer they land in is already the stock
        # kv-indices dtype. Halves this table.
        self.arch = torch.zeros(arena, dtype=torch.int32, device=dev)
        self.a_len = torch.zeros(P, dtype=torch.int64, device=dev)
        self.nk_len = torch.zeros(P, dtype=torch.int64, device=dev)
        self.thr = torch.zeros(P, 1, device=dev)
        self.cc = torch.zeros(P, device=dev)
        self._nk_col = torch.arange(NKm, device=dev)
        self.nk_mask = torch.zeros(P, 1, NKm, dtype=torch.bool, device=dev)
        self.scale = tiers[0].scale
        self.inf = torch.tensor(float("inf"), device=dev)
        self.li = torch.zeros(P, dtype=torch.int64, device=dev)
        self.slot = torch.zeros(P, dtype=torch.int64, device=dev)
        W = fetch_buf.shape[-1]
        # zeros, not empty: the kernel never stores to the padded region
        # (masked by a_len), so anything there at init is there forever --
        # torch.empty garbage would read as fired rows of ARCH padding.
        # int8: a 0/1 fired flag; the compaction reads it as a predicate.
        self.hit = torch.zeros(arena, dtype=torch.int8, device=dev)
        # fixed-address fused-prologue outputs (graph reads/writes in place)
        H = q_heads
        self.max1g = torch.zeros(P, H, device=dev)
        self.qside_t = torch.zeros(
            P, D.SIDECAR_DIM, H, device=dev, dtype=torch.bfloat16
        )
        self.qsk_t = torch.zeros(P, r, H, device=dev, dtype=torch.float16)
        self.qres = torch.zeros(P, H, device=dev)
        self.thr_flat = torch.zeros(P, device=dev)
        self.pm = torch.zeros(P, _NSPLIT, H, device=dev)
        self.ps = torch.zeros(P, _NSPLIT, H, device=dev)
        self.pt = torch.zeros(P, _NSPLIT, H, device=dev)
        NB = (Am + 1023) // 1024
        self.c_counts = torch.zeros(P, NB, dtype=torch.int32, device=dev)
        self.c_offsets = torch.zeros(P, NB, dtype=torch.int32, device=dev)
        self.c_total = torch.zeros(P, dtype=torch.int32, device=dev)
        self.scratch = torch.zeros(P, W + 1, dtype=torch.int64, device=dev)
        self.qbuf, self.fetch_buf, self.fetch_len = qbuf, fetch_buf, fetch_len
        self.update(pairs, tiers)

    @classmethod
    def at_capacity(
        cls,
        P,
        NKm,
        Am,
        r,
        q_heads,
        qbuf,
        fetch_buf,
        fetch_len,
        pad_slot,
        arena=None,
        pool_bases=None,
        pool_row=None,
        pool_rows=None,
        side_from_pool=False,
        csk_from_tier=False,
    ):
        """An empty pack sized for the worst case, for the in-graph scan.

        Built once at model-graph capture time, before any tier exists: every
        pair starts as a placeholder (a_len = nk_len = 0 -> the scan fires
        nothing and compact writes fetch_len[pad_slot] = 0), and update() later
        fills any subset of the P slots in place. Nothing about it ever
        reallocates, so a model graph that baked its addresses never needs a
        recapture."""
        self = cls.__new__(cls)
        dev = qbuf.device
        self.q_heads = q_heads
        self._pad_slot = pad_slot
        # Kept rows: either a snapshot this pack owns, or -- when the caller
        # hands over the pool's per-layer base pointers -- 4 bytes of row id
        # per kept row, scored in place out of the KV pool. The snapshot was
        # the pack's largest table by far (87% of it at bs16), and every byte
        # of it duplicated a pool row that can no longer change.
        self.nkm = NKm
        if pool_bases is None:
            self.kr = torch.zeros(
                P, NKm, D.LATENT_DIM, device=dev, dtype=torch.bfloat16
            )
            self.kslot = self.kbase = self._pool_bases = None
            self.pool_row = self._pool_rows = self.pool_mode = None
        else:
            self.kr = None
            self.kslot = torch.zeros(P, NKm, dtype=torch.int32, device=dev)
            self.kbase = torch.zeros(P, dtype=torch.int64, device=dev)
            self._pool_bases = pool_bases
            # The row stride is the POOL's, not this module's: taking it from
            # the buffer the caller actually hands over is what keeps a pool
            # layout change from silently addressing the wrong rows.
            self.pool_row = D.LATENT_DIM if pool_row is None else pool_row
            self._pool_rows = pool_rows
            # None = pick per device (TMA gather when it runs here);
            # set explicitly only to pin one path, as the tests do.
            self.pool_mode = None
            # How the scan reads the sidecar when it is not packed: same
            # choice the kept rows make, and for the same reason -- an
            # indirect load is not staged, a TMA gather is.
            from sglang.srt.layers.attention.vestigekv.fused_prologue import (
                _pool_read_mode,
            )

            self.side_mode = _pool_read_mode(dev) if side_from_pool else 0
        self.v = torch.zeros(P, r, D.KV_LORA_RANK, device=dev)
        # Archive tables are ONE arena, addressed by a_off[p] (see the scan
        # kernel): `arena` rows total instead of P x Am, which at long context
        # is what the pool can hold rather than max_bs x max_context.
        # Default P * Am reproduces the old per-pair slab exactly; a caller
        # that knows the KV pool's own bound passes it instead, which is what
        # actually shrinks the footprint (the pool caps the SUM of the
        # per-pair archives well below max_bs x max_context).
        arena = P * Am if arena is None else max(arena, Am)
        self.arena = arena
        self.am_grid = Am
        # The sidecar table can go the way of the kept rows: it is the tail of
        # an archived pool row and arch carries that row's id. csk and rho
        # stay -- those are computed (a rank-64 projection and its residual
        # norm), not copies of anything the pool still holds.
        self.side = (
            None
            if side_from_pool
            else torch.zeros(arena, D.SIDECAR_DIM, device=dev, dtype=torch.bfloat16)
        )
        # The sketch projections are a selection over the tier's closed-prefix
        # cache; carrying a compacted copy here is the same numbers twice.
        # The pack keeps the 4-byte row index instead and reads through it.
        if csk_from_tier:
            self.csk = None
            self.aidx = torch.zeros(arena, dtype=torch.int32, device=dev)
            self.cbase = torch.zeros(P, dtype=torch.int64, device=dev)
            from sglang.srt.layers.attention.vestigekv.fused_prologue import (
                _pool_read_mode,
            )

            self.csk_mode = _pool_read_mode(dev)
        else:
            self.csk = torch.zeros(arena, r, device=dev, dtype=torch.float16)
            self.aidx = self.cbase = None
            self.csk_mode = 0
        self.rank = r
        self._csk_rows = 0
        self._csk_refs = []  # keeps the tiers' caches alive while cbase points at them
        self.rho = torch.zeros(arena, device=dev)
        self.a_off = torch.zeros(P, dtype=torch.int64, device=dev)
        # int32: archive entries are pool row indices, bounded by max_total_tokens
        # (~2M), and the fetch buffer they land in is already the stock
        # kv-indices dtype. Halves this table.
        self.arch = torch.zeros(arena, dtype=torch.int32, device=dev)
        self.a_len = torch.zeros(P, dtype=torch.int64, device=dev)
        self.nk_len = torch.zeros(P, dtype=torch.int64, device=dev)
        self.thr = torch.zeros(P, 1, device=dev)
        self.cc = torch.zeros(P, device=dev)
        self._nk_col = torch.arange(NKm, device=dev)
        # all pairs empty -> every kept column masked True from step one
        self.nk_mask = torch.ones(P, 1, NKm, dtype=torch.bool, device=dev)
        self.scale = D.ATTN_SCALE
        self.inf = torch.tensor(float("inf"), device=dev)
        self.li = torch.zeros(P, dtype=torch.int64, device=dev)
        self.slot = torch.full((P,), pad_slot, dtype=torch.int64, device=dev)
        W = fetch_buf.shape[-1]
        # int8: a 0/1 fired flag; the compaction reads it as a predicate.
        self.hit = torch.zeros(arena, dtype=torch.int8, device=dev)
        H = q_heads
        self.max1g = torch.zeros(P, H, device=dev)
        self.qside_t = torch.zeros(
            P, D.SIDECAR_DIM, H, device=dev, dtype=torch.bfloat16
        )
        self.qsk_t = torch.zeros(P, r, H, device=dev, dtype=torch.float16)
        self.qres = torch.zeros(P, H, device=dev)
        self.thr_flat = torch.zeros(P, device=dev)
        self.pm = torch.zeros(P, _NSPLIT, H, device=dev)
        self.ps = torch.zeros(P, _NSPLIT, H, device=dev)
        self.pt = torch.zeros(P, _NSPLIT, H, device=dev)
        NB = (Am + 1023) // 1024
        self.c_counts = torch.zeros(P, NB, dtype=torch.int32, device=dev)
        self.c_offsets = torch.zeros(P, NB, dtype=torch.int32, device=dev)
        self.c_total = torch.zeros(P, dtype=torch.int32, device=dev)
        self.scratch = torch.zeros(P, W + 1, dtype=torch.int64, device=dev)
        self.qbuf, self.fetch_buf, self.fetch_len = qbuf, fetch_buf, fetch_len
        self.pairs = []
        self.tier_ids = ()
        return self

    def fits(self, pairs, tiers) -> bool:
        """Whether update() can host these pairs without reallocating (and
        therefore without invalidating the captured graph)."""
        if not tiers:
            return self._pad_slot is not None
        n_ok = (
            len(tiers) <= self.li.shape[0]
            if self._pad_slot is not None
            else len(tiers) == self.li.shape[0]
        )
        return (
            n_ok
            and max(t.kept_rows.shape[0] for t in tiers) <= self.nkm
            and sum(t.arch.shape[0] for t in tiers) <= self.arena
            and all(t.r == self.rank for t in tiers)
        )

    def update(self, pairs, tiers):
        """Point the pack at a new set of tiers IN PLACE.

        Every shape-dependent quantity the kernel and the tail consume
        (a_len, nk_mask, cc, thr, arch) is a tensor read at replay time, so
        refreshing the contents revalidates the captured graph without a
        recapture. Ordering is the main stream's: these copies are enqueued
        before the replay that consumes them.

        The nk_mask beyond each pair's kept count must stay True (padded kept
        rows would otherwise win max1 with a zero score), and the a_len mask
        keeps the kernel off each pair's stale archive tail, so shrinking
        requests leave garbage that is never read.
        """
        self.pairs = list(pairs)
        # Assign each pair a contiguous run in the archive arena, in pair
        # order. Offsets live in device memory and are refreshed here, so the
        # captured graph (which baked only the arena base and this table's
        # address) picks them up on replay.
        # Lengths come from the INDEX tables (arch, kept_slots), never from
        # the content tables (side, kept_rows): the content ones are becoming
        # lazily-materialised views of the pool, and touching them for a shape
        # would gather the whole archive to read one integer.
        offs, run = [], 0
        for t in tiers:
            offs.append(run)
            run += t.arch.shape[0]
        if run > self.arena:
            raise RuntimeError(
                f"archive arena overflow: {run} rows needed, {self.arena} "
                f"available -- fits() should have refused this update"
            )
        if offs:
            self.a_off[: len(offs)] = torch.tensor(
                offs, dtype=torch.int64, device=self.a_off.device
            )
        csk_refs, rows = [], 0
        for i, t in enumerate(tiers):
            nk, av = t.kept_rows.shape[0], t.arch.shape[0]
            if nk > self.nkm:
                raise RuntimeError(
                    f"kept-row overflow: pair {i} keeps {nk} rows, capacity is "
                    f"{self.nkm} -- fits() should have refused this update"
                )
            o = offs[i]
            if self.kslot is None:
                self.kr[i, :nk] = t.kept_rows
                self.kr[i, nk:] = 0
            else:
                # Row ids only. Rows past nk are never read (the scan clamps
                # by nk_len), but a stale id there would read a live pool row
                # if that clamp ever slipped, so zero it like the snapshot did.
                self.kslot[i, :nk] = t.kept_slots
                self.kslot[i, nk:] = 0
                self.kbase[i] = self._pool_bases[pairs[i][0]]
            self.v[i] = t.V
            if self.side is not None:
                self.side[o : o + av] = t.side
            # Select straight out of the tier's closed-prefix caches into
            # the arena. Going through t.csk would materialise the archive's
            # selection first, which is the copy this change exists to avoid.
            # A tier without those caches (a test double, or one adopted
            # before they existed) still has the selection, so it copies.
            has_caches = (
                getattr(t, "_csk_all", None) is not None
                and getattr(t, "_arch_idx", None) is not None
            )
            if self.csk is None:
                # Read-through: store the row index and where to read it from.
                # The cache is reallocated by torch.cat at every block close, so
                # cbase is refreshed here -- update() runs on the epoch bump a
                # close raises, before the next replay, so no replay ever sees a
                # stale address.
                if not has_caches:
                    raise RuntimeError(
                        "pack reads projections from the tier's cache, but this "
                        "tier carries none (_csk_all/_arch_idx missing)"
                    )
                self.aidx[o : o + av] = t._arch_idx
                self.cbase[i] = t._csk_all.data_ptr()
                csk_refs.append(t._csk_all)
                rows = max(rows, int(t._csk_all.shape[0]))
                torch.index_select(
                    t._rho_all,
                    0,
                    t._arch_idx.to(torch.int64),
                    out=self.rho[o : o + av],
                )
            elif has_caches:
                aidx64 = t._arch_idx.to(torch.int64)
                torch.index_select(t._csk_all, 0, aidx64, out=self.csk[o : o + av])
                torch.index_select(
                    t._rho_all,
                    0,
                    t._arch_idx.to(torch.int64),
                    out=self.rho[o : o + av],
                )
            else:
                self.csk[o : o + av] = t.csk
                self.rho[o : o + av] = t.rho
            self.arch[o : o + av] = t.arch
            self.a_len[i] = av
            self.nk_len[i] = nk
            self.thr[i, 0] = t.thr_g
            self.thr_flat[i] = t.thr_g
            self.cc[i] = t.zp * t.scale / (D.KV_LORA_RANK - t.r) ** 0.5
        if self.csk is None:
            # Hold the caches alive: cbase is a raw address, and a tier going
            # out of scope would free the memory the graph still points at.
            self._csk_refs = csk_refs
            self._csk_rows = rows
        if run and self.side is None and self._pool_rows is not None:
            # The scan now reads sidecars THROUGH arch, so arch must already
            # hold pool row ids. RecallTier.build leaves prefix POSITIONS there
            # and the backend remaps them; a caller that skips the remap would
            # read whatever sits at those row numbers -- plausible values, a
            # wrong fetch set, and nothing to notice it by. Runs on a tier
            # change, not per step.
            hi = int(self.arch[:run].max())
            if hi >= self._pool_rows:
                raise RuntimeError(
                    f"archive row id {hi} is outside the {self._pool_rows}-row "
                    "KV pool: arch still holds prefix positions, not pool row "
                    "ids (the backend remaps them after build)"
                )
        # copy_, not reassignment: the captured graph reads THIS tensor's
        # address; a fresh tensor here would silently detach every later
        # update from the replayed kernel's view of the mask.
        n = len(tiers)
        if n < self.li.shape[0]:
            # Capacity pack, partially occupied: the tail pairs are silenced
            # (zero lengths -> the scan fires nothing) and their compact
            # output lands in the pad slot's fetch row, which nothing reads.
            # Stale kr/side contents behind the zero lengths are never read.
            self.a_len[n:] = 0
            self.nk_len[n:] = 0
        self.nk_mask.copy_(self._nk_col[None, None, :] >= self.nk_len[:, None, None])
        pad = self.li.shape[0] - n
        li_l = [p[0] for p in pairs] + [0] * pad
        slot_l = [p[1] for p in pairs] + [self._pad_slot or 0] * pad
        self.li.copy_(torch.tensor(li_l, device=self.li.device))
        self.slot.copy_(torch.tensor(slot_l, device=self.li.device))
        self.tier_ids = tuple((id(t), getattr(t, "version", 0)) for t in tiers)

    @ieee_fp32
    def run(self, p_live=None):
        """p_live: number of leading pair slots this launch covers (the
        decode graph bakes it per batch-size class: layers x class_bs).
        update() fills live pairs contiguously from position 0, so a grid
        clipped to p_live sees every live pair; capacity-tail placeholders
        beyond it are never launched at all. Measured: the placeholder tax
        was +0.32 ms/step at --cuda-graph-max-bs 16 serving bs=1."""
        P_eff = p_live if p_live is not None else self.li.shape[0]
        sc = self.scale
        # Fused prologue: skept/softmax/entropy/gate/qsk/qres/max1g and the
        # transpose-casts in ONE kernel (see fused_prologue.py). Replaces the
        # ~12-launch eager chain whose fixed ~0.57 ms execution intercept
        # dominated the captured graph (VKSTATS S-sweep). Empty tier-1 pairs
        # (nk_len==0) come back with max1g=-inf: whole archive fires, full
        # attention, never under-recall.
        fused_prologue_split(
            self.qbuf,
            self.li,
            self.slot,
            self.kr,
            self.v,
            self.nk_len,
            self.thr_flat,
            sc,
            out=(self.max1g, self.qside_t, self.qsk_t, self.qres),
            partials=(self.pm, self.ps, self.pt),
            kslot=self.kslot,
            kbase=self.kbase,
            nkm=self.nkm,
            row=self.pool_row,
            pool_rows=self._pool_rows,
            mode=self.pool_mode,
            a_len=self.a_len,
        )
        qside_t, qsk_t, qres, max1g = self.qside_t, self.qsk_t, self.qres, self.max1g
        # Grid covers the LARGEST per-pair archive, not the arena: programs
        # past a pair's a_len do no work, so a grid sized for the arena would
        # waste 1/P of its blocks on every pair. am_grid is baked at capture
        # (max_bs x max_context worth of rows is still the worst case a single
        # pair can reach), while the arena only bounds the SUM. The launch is
        # capped at SCAN_GRID_CAP programs per pair and the kernel grid-strides
        # over the 1024-row buckets: same worst-case coverage, much smaller
        # dispatch floor at short context (defaults.SCAN_GRID_CAP).
        Am = self.am_grid
        P = P_eff
        _scan_batched_kernel[
            (min(triton.cdiv(Am, D.SCAN_BLOCK_A * 16), D.SCAN_GRID_CAP), P)
        ](
            qside_t,
            qsk_t,
            qres,
            max1g,
            self.side,
            self.arch,
            self.kbase,
            self.csk,
            self.aidx,
            self.cbase,
            self.rho,
            self.a_len,
            self.a_off,
            self.cc,
            self.hit,
            self.c_counts,
            Am,
            sc,
            H=self.q_heads,
            DD=D.SIDECAR_DIM,
            R=self.v.shape[1],
            SIDE_POOL=0 if self.side is not None else self.side_mode,
            CSK_TIER=0 if self.csk is not None else self.csk_mode,
            CSK_ROWS=self._csk_rows or 1,
            ROW=self.pool_row or 0,
            KV_OFF=D.KV_LORA_RANK,
            POOL_ROWS=self._pool_rows or 1,
            BLOCK_A=D.SCAN_BLOCK_A,
            MULTI=16,  # 16 x BLOCK_A(64) = one 1024-row compact bucket
            num_warps=D.SCAN_NUM_WARPS,
        )
        # Deterministic two-phase Triton compaction: the torch chain's int64
        # cumsum alone cost 157 us/step at 128k (nsys, 1.5x the scan kernel).
        compact_fired(
            self.hit,
            self.arch,
            self.a_len,
            self.a_off,
            self.li,
            self.slot,
            self.fetch_buf,
            self.fetch_len,
            (self.c_counts, self.c_offsets, self.c_total),
            self.am_grid,
        )
