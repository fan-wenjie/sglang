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
    cc_ptr,  # [P] fp32: zp * sc / sqrt(kv_lora - R)
    hit_ptr,  # [P, Amax] int32 out
    Amax,
    sc,
    H: tl.constexpr,
    DD: tl.constexpr,
    R: tl.constexpr,
    BLOCK_A: tl.constexpr,
):
    p = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK_A + tl.arange(0, BLOCK_A)
    m = offs < tl.load(a_len_ptr + p)
    d = tl.arange(0, DD)
    r = tl.arange(0, R)
    h = tl.arange(0, H)
    # quantized storage, fp32 ieee arithmetic (see scan_kernel)
    s = tl.load(
        side_ptr + p * Amax * DD + offs[:, None] * DD + d[None, :],
        mask=m[:, None],
        other=0.0,
    )
    c = tl.load(
        csk_ptr + p * Amax * R + offs[:, None] * R + r[None, :],
        mask=m[:, None],
        other=0.0,
    )
    rh = tl.load(rho_ptr + p * Amax + offs, mask=m, other=0.0)
    qs = tl.load(qside_t_ptr + p * DD * H + d[:, None] * H + h[None, :])
    qk = tl.load(qsk_t_ptr + p * R * H + r[:, None] * H + h[None, :])
    cc = tl.load(cc_ptr + p)
    # ieee, not tf32: the fast path disagreed with the eager fire set on 6 rows
    # in 58900, a silent change to which rows the model attends.
    # native-dtype tensor-core dots, fp32 accumulation -- see scan_kernel.py
    acc = tl.dot(s, qs).to(tl.float32) + tl.dot(c, qk).to(tl.float32)
    score = acc * sc + cc * rh[:, None] * tl.load(qres_ptr + p * H + h)[None, :]
    fired = tl.max((score > tl.load(max1g_ptr + p * H + h)[None, :]).to(tl.int32), 1)
    tl.store(hit_ptr + p * Amax + offs, fired, mask=m)


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
        self.side = torch.zeros(P, Am, D.SIDECAR_DIM, device=dev, dtype=torch.bfloat16)
        self.csk = torch.zeros(P, Am, r, device=dev, dtype=torch.float16)
        self.rho = torch.zeros(P, Am, device=dev)
        self.arch = torch.zeros(P, Am, dtype=torch.int64, device=dev)
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
        self.hit = torch.zeros(P, Am, dtype=torch.int32, device=dev)
        # fixed-address fused-prologue outputs (graph reads/writes in place)
        H = q_heads
        self.max1g = torch.zeros(P, H, device=dev)
        self.qside_t = torch.zeros(P, D.SIDECAR_DIM, H, device=dev, dtype=torch.bfloat16)
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

    def fits(self, pairs, tiers) -> bool:
        """Whether update() can host these pairs without reallocating (and
        therefore without invalidating the captured graph)."""
        return (
            len(tiers) == self.li.shape[0]
            and max(t.kept_rows.shape[0] for t in tiers) <= self.kr.shape[1]
            and max(t.side.shape[0] for t in tiers) <= self.side.shape[1]
            and all(t.r == self.csk.shape[2] for t in tiers)
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
        for i, t in enumerate(tiers):
            nk, av = t.kept_rows.shape[0], t.side.shape[0]
            self.kr[i, :nk] = t.kept_rows
            self.kr[i, nk:] = 0
            self.v[i] = t.V
            self.side[i, :av] = t.side
            self.csk[i, :av] = t.csk
            self.rho[i, :av] = t.rho
            self.arch[i, :av] = t.arch
            self.a_len[i] = av
            self.nk_len[i] = nk
            self.thr[i, 0] = t.thr_g
            self.thr_flat[i] = t.thr_g
            self.cc[i] = t.zp * t.scale / (D.KV_LORA_RANK - t.r) ** 0.5
        # copy_, not reassignment: the captured graph reads THIS tensor's
        # address; a fresh tensor here would silently detach every later
        # update from the replayed kernel's view of the mask.
        self.nk_mask.copy_(
            self._nk_col[None, None, :] >= self.nk_len[:, None, None]
        )
        self.li.copy_(torch.tensor([p[0] for p in pairs], device=self.li.device))
        self.slot.copy_(torch.tensor([p[1] for p in pairs], device=self.li.device))
        self.tier_ids = tuple((id(t), getattr(t, "version", 0)) for t in tiers)

    @ieee_fp32
    def run(self):
        sc = self.scale
        qe = self.qbuf[self.li, self.slot].float()  # [P, H, 576], one gather+cast
        # Fused prologue: skept/softmax/entropy/gate/qsk/qres/max1g and the
        # transpose-casts in ONE kernel (see fused_prologue.py). Replaces the
        # ~12-launch eager chain whose fixed ~0.57 ms execution intercept
        # dominated the captured graph (VKSTATS S-sweep). Empty tier-1 pairs
        # (nk_len==0) come back with max1g=-inf: whole archive fires, full
        # attention, never under-recall.
        fused_prologue_split(
            qe.contiguous(),
            self.kr,
            self.v,
            self.nk_len,
            self.thr_flat,
            sc,
            out=(self.max1g, self.qside_t, self.qsk_t, self.qres),
            partials=(self.pm, self.ps, self.pt),
        )
        qside_t, qsk_t, qres, max1g = self.qside_t, self.qsk_t, self.qres, self.max1g
        P, Am = self.hit.shape
        _scan_batched_kernel[(triton.cdiv(Am, D.SCAN_BLOCK_A), P)](
            qside_t,
            qsk_t,
            qres,
            max1g,
            self.side,
            self.csk,
            self.rho,
            self.a_len,
            self.cc,
            self.hit,
            Am,
            sc,
            H=self.q_heads,
            DD=D.SIDECAR_DIM,
            R=self.v.shape[1],
            BLOCK_A=D.SCAN_BLOCK_A,
            num_warps=D.SCAN_NUM_WARPS,
        )
        # Deterministic two-phase Triton compaction: the torch chain's int64
        # cumsum alone cost 157 us/step at 128k (nsys, 1.5x the scan kernel).
        compact_fired(
            self.hit, self.arch, self.a_len, self.li, self.slot,
            self.fetch_buf, self.fetch_len,
            (self.c_counts, self.c_offsets, self.c_total),
        )
