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
    side_ptr,  # [P, Amax, D] fp32
    csk_ptr,  # [P, Amax, R] fp32
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
    BLOCK_A: tl.constexpr,
    MULTI: tl.constexpr,
):
    p = tl.program_id(1)
    al = tl.load(a_len_ptr + p)
    # Grid is capacity-sized (the capture bakes it). Each program covers one
    # 1024-row bucket (MULTI sub-blocks of BLOCK_A): a 16x smaller grid than
    # one-block programs -- the capacity grid's launch floor was the step's
    # largest fixed cost (117 us content-independent; an all-placeholder
    # pack scanned as slow as a live one) -- and the fused compact count
    # becomes a plain per-bucket store instead of a 16-way atomic. Programs
    # fully past a_len exit on one scalar load.
    base = tl.program_id(0) * (MULTI * BLOCK_A)
    if base >= al:
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
    cnt = 0
    for kb in range(MULTI):
        offs = base + kb * BLOCK_A + tl.arange(0, BLOCK_A)
        m = offs < al
        # quantized storage, fp32 ieee arithmetic (see scan_kernel)
        s = tl.load(
            side_ptr + (abase + offs[:, None]) * DD + d[None, :],
            mask=m[:, None],
            other=0.0,
        )
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
    # Fused compact count: the program IS the 1024-row bucket, so the total
    # is a plain store (pre-zeroing by the prefix kernel covers programs
    # that exited above).
    nb = (Amax + 1023) // 1024
    tl.store(counts_ptr + p * nb + tl.program_id(0), cnt)


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
        dev = tiers[0].side.device
        self.q_heads = q_heads
        self._pad_slot = None  # legacy packs are always fully occupied
        P = len(tiers)
        # Floor to 1: an all-empty-kept capture (every pair's tier-1 kept the
        # empty set -- the re-prefill anomaly) would otherwise allocate a
        # zero-width kr and crash skept.max(-1). One padded row, masked off by
        # nk_len=0, keeps the reduction well-formed; the empty pair is served
        # as full-archive recall below.
        NKm = max(1, int(max(t.kept_rows.shape[0] for t in tiers) * self.HEADROOM))
        Am = max(1, int(max(t.side.shape[0] for t in tiers) * self.HEADROOM))
        r = tiers[0].r
        self.kr = torch.zeros(P, NKm, D.LATENT_DIM, device=dev, dtype=torch.bfloat16)
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
        cls, P, NKm, Am, r, q_heads, qbuf, fetch_buf, fetch_len, pad_slot, arena=None
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
        self.kr = torch.zeros(P, NKm, D.LATENT_DIM, device=dev, dtype=torch.bfloat16)
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
        self.side = torch.zeros(arena, D.SIDECAR_DIM, device=dev, dtype=torch.bfloat16)
        self.csk = torch.zeros(arena, r, device=dev, dtype=torch.float16)
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
            and max(t.kept_rows.shape[0] for t in tiers) <= self.kr.shape[1]
            and sum(t.side.shape[0] for t in tiers) <= self.arena
            and all(t.r == self.csk.shape[1] for t in tiers)
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
        offs, run = [], 0
        for t in tiers:
            offs.append(run)
            run += t.side.shape[0]
        if run > self.arena:
            raise RuntimeError(
                f"archive arena overflow: {run} rows needed, {self.arena} "
                f"available -- fits() should have refused this update"
            )
        if offs:
            self.a_off[: len(offs)] = torch.tensor(
                offs, dtype=torch.int64, device=self.a_off.device
            )
        for i, t in enumerate(tiers):
            nk, av = t.kept_rows.shape[0], t.side.shape[0]
            o = offs[i]
            self.kr[i, :nk] = t.kept_rows
            self.kr[i, nk:] = 0
            self.v[i] = t.V
            self.side[o : o + av] = t.side
            self.csk[o : o + av] = t.csk
            self.rho[o : o + av] = t.rho
            self.arch[o : o + av] = t.arch
            self.a_len[i] = av
            self.nk_len[i] = nk
            self.thr[i, 0] = t.thr_g
            self.thr_flat[i] = t.thr_g
            self.cc[i] = t.zp * t.scale / (D.KV_LORA_RANK - t.r) ** 0.5
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
        )
        qside_t, qsk_t, qres, max1g = self.qside_t, self.qsk_t, self.qres, self.max1g
        # Grid covers the LARGEST per-pair archive, not the arena: programs
        # past a pair's a_len exit on one scalar load, so a grid sized for the
        # arena would waste 1/P of its blocks on every pair. am_grid is baked
        # at capture (max_bs x max_context worth of rows is still the worst
        # case a single pair can reach), while the arena only bounds the SUM.
        Am = self.am_grid
        P = P_eff
        _scan_batched_kernel[(triton.cdiv(Am, D.SCAN_BLOCK_A * 16), P)](
            qside_t,
            qsk_t,
            qres,
            max1g,
            self.side,
            self.csk,
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
