"""VestigeKV MLA attention backend (training-free NoPE-MLA KV-cache compression).

Wraps a base MLA backend (flashinfer_mla / trtllm_mla / ...). Every metadata,
cuda-graph, and prefill call delegates to the base unchanged; only decode is
intercepted, to attend over the kept latent rows plus a small per-step recalled
set instead of the whole cache. Disabled, it is the base backend call-for-call
(auditable kill-switch).

Seam (mapped against DeepseekV2AttentionMLA absorb path, deepseek_v2.py +
deepseek_common/attention_forward_methods/forward_mla.py):
  forward_decode receives q = q_nope_out [T, num_local_heads, kv_lora_rank] and
  q_rope = q_pe [T, ., qk_rope_head_dim]. The base kernel attends over kv_indices
  gathered from req_to_token_pool.req_to_token[req_pool_indices, :seq_len] against
  the MLATokenToKVPool latent rows [N, 1, kv_lora_rank + qk_rope_head_dim]. NoPE
  holds natively (skip_rope=True -> rotary_emb is None). VestigeKV never touches
  w_kc/w_vc, rope, or the model: eviction drops slots from that gathered set.
  See VESTIGEKV_PORT.md for the full contract.

The tier-2 recall is a required part of the algorithm and has no switch at
all: the method is the partition plus the recall over it, and without recall
quality collapses at long context. There is exactly one serving
configuration. Wiring: each step records its expanded query in-graph (qbuf); the
next step's out-of-graph refresh scans it against the per-slot index
(stale-by-one recall -- a declared deviation from the same-step reference,
gated on the needle/quality validation), writes fired pool rows into the
fixed-width fetch_buf, and the packed CSR splices kept + fetched segments.
The index is built twice per request: provisionally on the first decode step so
tier 2 is never off, then rebuilt once n_cal REAL decode queries have
accumulated (prefill queries are not usable -- MLA prefill runs the un-absorbed
path, [tokens, H, 192], and absorbing needs W_kc, a model weight the backend
never sees; see _collect_calibration). Graph capture
needs only the fixed fetch WIDTH; the uncapped algorithm runs unchanged under a
safety width.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.eviction import (
    blockwise_sigma,
    select_kept,
    sidecar_sigma,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
    from sglang.srt.layers.attention.verify_mask import VerifyMask
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner

# Recent decode/prefill tail always kept regardless of sigma; arbitrary, tune by
# quality. The growing decode tail past prefix_len is attended separately.
# Decode steps a request must spend on one unchanged scan shape before its
# tier-2 scan is captured. Capture costs a warmup plus the capture itself, so a
# short generation would otherwise pay for a graph it replays a handful of
# times. 8 keeps that below 1% for the 128-token benchmark and shorter.
# Extra kept-table columns baked into a capture. kept_len grows by one per
# decode step, so this is how many steps one capture survives before the graph
# would start dropping the newest rows and must be rebuilt.
# Preallocated decode-tail slots per (req, layer) index buffer; grown 2x on
# overflow, so this only sizes the common case.
_TAIL_CAPACITY = 8192


class VestigeKVMLABackend(AttentionBackend):
    """Wrap a base MLA backend; compress the latent cache on decode.

    topj is the per-head fetch cap: default -1 = UNCAPPED (fetch the full
    fired recall set). Set topj > 0 explicitly (SGLANG_VESTIGEKV_TOPJ; 16
    recommended) to enable the bounded-fetch guarantee (fetch <= topj *
    num_heads rows / step / layer). rho is the tier-1 eviction ratio;
    index_rank is the tier-2 sketch rank r.
    """

    # pack-content epoch: class-level defaults so __new__-constructed test
    # fixtures see them; bumped on every host-side tier mutation. When
    # unchanged, the replay fast path skips per-step pairs/fits bookkeeping.
    _pack_epoch = 0
    _pack_epoch_synced = -1
    # ---- in-graph scan (SGLANG_ENABLE_VESTIGEKV_INGRAPH_SCAN) ----
    # Class-level defaults (unit fakes build the backend via __new__): one
    # worst-case-capacity pack whose kernels the decode model graph bakes;
    # update() refreshes contents in place, so the model graph never
    # recaptures for VestigeKV reasons. _trash_slot is the extra row on every
    # slot-indexed buffer that absorbs padded/placeholder lanes.
    _ingraph_pack = None
    _trash_slot: int | None = None
    _ingraph_steps = 0
    _ingraph_full_armed = False
    _ingraph_dead = False
    # capture-reason counters (diagnostic; printed by VKSTATS when stats on)
    _cap_keymiss = 0
    _cap_kmax = 0
    _cap_fits = 0
    _cap_lru_hit = 0
    _scan_cache = None  # OrderedDict[key -> {graph, pack, kmax}], LRU cap 2

    def __init__(
        self,
        base: AttentionBackend,
        model_runner: ModelRunner,
        *,
        rho: float = D.RHO,
        topj: int = -1,
        index_rank: int = D.INDEX_RANK,
    ):
        self.base = base
        self.attn_backend_list = [base]  # let generic snapshot/restore reach the child
        # Capability flags are class attributes, not methods, so delegation has
        # to be explicit: inheriting the AttentionBackend defaults instead of
        # the wrapped backend's values silently changes the runtime's fast
        # paths. Missing needs_cpu_seq_lens alone cost 137 -> 51 tok/s (the
        # scheduler synced seq_lens to host every step).
        for _flag in (
            "needs_cpu_seq_lens",
            "extend_dummy_seqs_capped_by_req_pool",
            "supports_ragged_verify_graph",
            "supports_full_cuda_graph_chunked_prefix",
            "use_captured_forward_metadata_for_breakable_cuda_graph",
            "prefill_attention_backend_str",
            "decode_attention_backend_str",
        ):
            setattr(self, _flag, getattr(base, _flag))
        self.rho = rho
        self.topj = topj
        self.index_rank = index_rank
        # Pool handles shared with the base (the latent rows VestigeKV compresses).
        self.token_to_kv_pool = base.token_to_kv_pool
        self.req_to_token_pool = base.req_to_token_pool
        self.kv_index_translator = base.kv_index_translator
        # Per-request tier-2 index, keyed by req_pool_index; built at prefill end.
        self._tier2: dict = {}
        # Per-layer capture-stable index buffers for CUDA-graph decode: the graph
        # records these addresses; the out-graph hook refreshes contents per step.
        self._graph_bufs: dict = {}
        self._graph_max_bs = 0
        self._num_layers = model_runner.model_config.num_hidden_layers
        # 1-indexed in the checkpoint config -> 0-indexed layer ids
        self._mla_lids_static = {
            lid1 - 1
            for lid1 in model_runner.model_config.hf_config.linear_attn_config[
                "full_attn_layers"
            ]
        }
        # this PP rank's MLA layers, from the hybrid pool's authoritative map
        # (HybridLinearKVPool.full_attention_layer_id_mapping)
        self._local_mla_lids = sorted(
            base.token_to_kv_pool.full_attention_layer_id_mapping
        )
        # MLA layer ids actually routed to this backend, learned at capture time
        # (the hybrid never sends KDA layers here); replay refreshes only these.
        self._mla_lids: set = set()
        # GPU-side per-(pool-slot, layer) kept-index tables: built at prefill
        # (one sync there is free); decode refresh is then pure GPU ops, no
        # host sync on the critical path. bs=1 graphs read kept_buf directly.
        self._kept_buf: dict = {}  # lid -> [max_reqs, cap] fm.kv_indices dtype
        self._kept_len: dict = {}
        self._kmax: dict = {}  # lid -> host-side upper bound on kept_len
        self._vcache: dict = {}  # lid -> last calibrated sketch basis
        self._close_state: dict = {}  # (slot, lid) -> {closed, sigma}
        import queue as _q

        self._build_queue: _q.Queue = _q.Queue()
        self._build_jobs: list = []
        self._build_worker = None
        self._build_stream = None
        self._qbuf_stack = self._fetch_stack = self._fetch_len_stack = None
        self._li_map: dict = {}
        self._indptr1: dict = {}  # lid -> [2] capture-stable CSR for bs=1
        # ---- recall tier (REQUIRED component; no production off-switch) ----
        # Expanded-query dims from the model config (q_nope@W_kc | q_rope).
        cfg = model_runner.model_config.hf_config
        self._q_heads = model_runner.model_config.num_attention_heads
        self._q_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim  # 512 + 64
        # Fixed fetch-buffer width: graph capture needs a fixed WIDTH, not a
        # cap. topj>0 sets it to the capped bound; the uncapped algorithm gets
        # a safety width above the observed worst fire (3983); overflow is
        # truncated and counted in the histogram.
        self._fetch_w = (
            self.topj * self._q_heads
            if self.topj and self.topj > 0
            else D.FETCH_WIDTH_UNCAPPED
        )
        self._qbuf: dict = {}  # lid -> [max_reqs, H, 576] fp32, in-graph updated
        self._fetch_buf: dict = {}  # lid -> [max_reqs, W] fm.kv_indices dtype
        self._fetch_len: dict = {}  # lid -> [max_reqs] int64
        self._recall: dict = {}  # (slot, lid) -> {"tier", "built_at"}
        self._stats = dict.fromkeys(
            ("steps", "scan_calls", "fetched", "kept", "seq", "replays"), 0
        )
        self._scan_graph = None
        self._scan_key_cur = self._scan_key_seen = None
        self._scan_steps = 0
        self._scan_capture_failed = False
        self._scan_fails = 0
        self._scan_batched = None
        self._capture_asap = False
        self._scan_kmax: dict = {}  # lid -> gather width baked into the capture
        self._stage_slots = self._stage_loc = None
        self._step_cache = None  # ((bs, real_bs), reqs, scan key) -- see _step_slots
        self._collecting = False  # any request still gathering calibration queries
        self._needs_recapture = False  # an install awaits its coalesced recapture
        self._stats["replays"] = self._stats["n_build"] = 0
        self._stats["n_capture"] = 0
        self._stats["t_build"] = self._stats["t_capture"] = 0.0
        for _k in ("t_replay", "t_replay_host", "t_eager"):
            self._stats[_k] = 0.0
        for _k in ("t_scan", "t_pack", "t_launch"):
            self._stats[_k] = 0.0
        self._nstep_hist: dict = {}  # fired-count -> occurrences (telemetry)

    # ---- metadata / cuda-graph / properties: delegate to base ----

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.base.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        if (
            not in_capture
            and self._ingraph_pack is not None
            and forward_batch.forward_mode.is_decode()
            and self._out_graph_metadata_lite(forward_batch)
        ):
            pass  # lite path succeeded; the full base call is skipped
        else:
            self.base.init_forward_metadata_out_graph(forward_batch, in_capture)
        # in_capture=True runs BEFORE `with graph.capture()`: allocate + pre-fill
        # the per-layer fixed buffers here (copies/syncs are legal outside
        # capture); inside capture forward_decode only swaps pointers to them.
        if in_capture and forward_batch.forward_mode.is_decode():
            self._prefill_graph_bufs_for_capture(forward_batch)
            return
        # Graph replay path: forward_decode's python never runs, so refresh the
        # buffers here (outside the graph) with this step's compressed indices.
        if (
            not in_capture
            and self._graph_bufs
            and forward_batch.forward_mode.is_decode()
        ):
            if envs.SGLANG_DEBUG_VESTIGEKV_ROWS.get():
                self._check_row_invariant(forward_batch)
            reqs, key = self._step_slots(forward_batch)
            self._maybe_close_blocks(forward_batch, reqs)
            # Collecting a calibration query is five 74 KB clones; it does not
            # need the eager scan path, and tying it to that path cost 16 eager
            # steps per request at 11.3 ms each against 1.5 ms for a replay.
            if self._collecting:
                self._collect_calibration(forward_batch, reqs)
            if self._ingraph_pack is not None:
                # In-graph mode: the model graph itself replays the scan and
                # the CSR pack; the host only refreshes what the graph reads.
                self._ingraph_host_step(forward_batch, reqs)
                return
            if envs.SGLANG_DEBUG_VESTIGEKV_STATS.get():
                self._step_with_stats(forward_batch, reqs, key)
                return
            if self._replay_scan(forward_batch, reqs, key):
                return
            for lid in self._mla_lids:
                self._recall_step(lid, forward_batch, reqs)
                self._refresh_graph_bufs(lid, forward_batch, reqs)

    def _out_graph_metadata_lite(self, forward_batch) -> bool:
        """Replay-prep without the dense kv_indices fill (in-graph decode).

        The base's decode replay-prep spends ~99 us/step building the DENSE
        row list (fill_packed_read_stream -> create_flashinfer_kv_indices),
        and this wrapper then pointer-swaps every MLA layer's CSR to its own
        packed buffers -- the fill's output is discarded whole. This lite
        fork keeps the two pieces the step still needs, through the base's
        own methods (no stock change): num_kv_splits sizing from the DENSE
        lens (split count changes accumulation grouping, so resizing from
        compressed lens would break bit-parity) and the unified-pool write
        locs. Returns False (caller falls back to the full base hook) on any
        base shape it does not recognize.
        """
        # self.base IS the full-attn TritonAttnBackend (the hybrid wrapper
        # sits OUTSIDE this backend and calls each child's hook itself, so
        # the linear side is not this method's concern).
        fa = self.base
        if (
            getattr(fa, "dcp_size", 1) > 1
            or getattr(fa, "use_sliding_window_kv_pool", False)
            or forward_batch.spec_info is not None
        ):
            return False
        try:
            bs = forward_batch.batch_size
            seq_lens = forward_batch.seq_lens[:bs]
            kv_indptr = fa.kv_indptr[: bs + 1]
            kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
            fa.get_num_kv_splits(fa.cuda_graph_num_kv_splits[:bs], seq_lens)
            fa._fill_cuda_graph_write_locs(forward_batch, bs)
        except AttributeError:
            return False
        return True

    def _step_slots(self, forward_batch):
        """This step's pool slots and capture key, without a device readback.

        `req_pool_indices.tolist()` is a D2H sync, and it sits at the top of the
        step, so it drained the pipeline before any of this backend's work was
        even queued -- the host could never run ahead of the device. During
        decode the value is constant: a request can only JOIN the batch through
        extend (which rebuilds a tier and invalidates this cache), and can only
        LEAVE by shrinking out_cache_loc, which is a shape and therefore free to
        read. So (same shapes, no prefill since) implies the same slots in the
        same order, and the list is cached across steps.
        """
        sig = (forward_batch.seq_lens.shape[0], forward_batch.out_cache_loc.shape[0])
        cached = self._step_cache
        if cached is not None and cached[0] == sig:
            return cached[1], cached[2]
        reqs = forward_batch.req_pool_indices.tolist()
        key = self._scan_key(forward_batch, reqs)
        self._step_cache = (sig, reqs, key)
        return reqs, key

    # ---- tier-2 scan capture: 215 launches/step -> 1 ----

    def _scan_key(self, forward_batch, reqs):
        # The capture is valid only while every ADDRESS and SHAPE CLASS it
        # baked still holds: the same (layer, pool-slot) pairs at the same
        # batch size. Archive/kept SIZES deliberately stay out of the key:
        # the kernels read a_len/nk_len per pair and the grid is sized by the
        # pack's capacity, so content growth is the epoch fast path's job
        # (fits() -> update() in place; a tier outgrowing capacity fails
        # fits() and recaptures). Keying on the exact archive size made every
        # install/close a key change -- the capture churn behind the bs4
        # collapse (136 captures / 26 s despite the epoch path).
        real = forward_batch.out_cache_loc.shape[0]
        pairs = []
        for lid in self._mla_lids:
            if lid not in self._qbuf:
                continue
            for i in range(real):
                st = self._recall.get((reqs[i], lid))
                if st is None or st.get("tier") is None:
                    return None  # not all tiers built yet; stay eager
                pairs.append((lid, reqs[i]))
        if not pairs:
            return None
        # Canonical order: the scheduler reorders running requests step to
        # step, and the pack rebinds (li, slot) tensors through update(), so
        # slot order is not part of the baked shape -- sorting keeps a mere
        # reshuffle from reading as a new shape class (bs4: 137/137 captures
        # were key misses).
        return ("bs", forward_batch.seq_lens.shape[0], real) + tuple(sorted(pairs))

    def _kmax_exhausted(self) -> bool:
        # The capture bakes one gather width per layer. kept_len grows by one
        # every decode step, so a graph replayed past its baked headroom would
        # silently drop the newest rows -- the exact silent-wrong-row-set class
        # the row invariant exists to catch. Recapture one step early instead.
        return any(
            self._kmax.get(lid, 0) + 1 > baked for lid, baked in self._scan_kmax.items()
        )

    def _replay_scan(self, forward_batch, reqs, key) -> bool:
        """Run this step's tier-2 scan from a captured graph. Returns False if
        the caller must fall back to the eager per-layer path.

        query_fixed is capturable because every shape it touches is frozen for
        the life of a request: the archive is fixed at prefill (a decoded token
        joins the recent window, which is tier-1 kept, never the archive), and
        both its input (_qbuf) and its outputs (_fetch_buf, _fetch_len) are
        preallocated fixed-address buffers indexed by pool slot. So the capture
        reads and writes them in place -- no staging copies, one launch per
        step instead of 43 per (layer, request). Measured in isolation: 1.69 ->
        0.75 ms/step for 5 layers, results bit-identical."""
        if key is None or self._full_arm() or self._scan_capture_failed:
            return False
        if key == self._scan_key_cur:
            if self._kmax_exhausted():
                # Same shape, just out of baked headroom: recapture at once
                # rather than serving D.SCAN_CAPTURE_AFTER eager steps for a
                # stability the key has already demonstrated.
                self._cap_kmax += 1
                return self._capture_scan(key, forward_batch, reqs)
            real = forward_batch.out_cache_loc.shape[0]
            if self._pack_epoch != self._pack_epoch_synced:
                # Slow path only when a host-side tier mutation happened
                # (install/close/prefill): rebuild the pair list and resync
                # the pack. Steady-state decode skips all of this -- the
                # per-step tuple build over id()/version was measurable host
                # time for bookkeeping that could not have changed.
                tiers, pairs = [], []
                for lid in self._mla_lids:
                    if lid in self._qbuf:
                        for i in range(real):
                            pairs.append((self._li_map[lid], reqs[i]))
                            tiers.append(self._recall[(reqs[i], lid)]["tier"])
                order = sorted(range(len(pairs)), key=lambda k: pairs[k])
                pairs = [pairs[k] for k in order]
                tiers = [tiers[k] for k in order]
                if self._scan_batched.tier_ids != tuple(
                    (id(t), getattr(t, "version", 0)) for t in tiers
                ):
                    if not self._scan_batched.fits(pairs, tiers):
                        return self._capture_scan(key, forward_batch, reqs)
                    self._scan_batched.update(pairs, tiers)
                self._pack_epoch_synced = self._pack_epoch
            self._stage_slots[:real].copy_(
                forward_batch.req_pool_indices[:real], non_blocking=True
            )
            self._stage_loc[:real].copy_(forward_batch.out_cache_loc, non_blocking=True)
            for lid in self._mla_lids:
                if lid in self._kept_buf:
                    self._kmax[lid] = self._kmax.get(lid, 0) + 1
            self._scan_graph.replay()
            return True
        # LRU graph cache: chunked-prefill interleave oscillates the decode
        # composition (e.g. bs 1 <-> 4), and a single graph slot thrashed on
        # the ping-pong (VKKEYMISS showed two shape classes evicting each
        # other 137 times). A hit swaps the active references; capture puts
        # the outgoing entry into the cache instead of discarding it.
        ent = self._scan_cache.get(key) if self._scan_cache is not None else None
        if ent is not None:
            self._scan_cache.move_to_end(key)
            self._stash_active()
            (
                self._scan_graph,
                self._scan_batched,
                self._scan_kmax,
                self._scan_key_cur,
            ) = ent["graph"], ent["pack"], dict(ent["kmax"]), key
            # Epoch sync is global but pack contents are per-entry: force one
            # resync pass so a reactivated pack refreshes via fits()/update()
            # before its first replay (stale tiers otherwise).
            self._pack_epoch_synced = self._pack_epoch - 1
            self._cap_lru_hit += 1
            return self._replay_scan(forward_batch, reqs, key)
        # Amortization guard: capturing costs a warmup plus the capture itself,
        # so a short generation must not pay for a graph it replays twice.
        self._scan_steps = self._scan_steps + 1 if key == self._scan_key_seen else 1
        self._scan_key_seen = key
        # _capture_asap is set when an index build just completed: the deferral
        # would otherwise be reset by the key change the build itself caused.
        if not self._capture_asap and self._scan_steps < D.SCAN_CAPTURE_AFTER:
            return False
        self._capture_asap = False
        self._cap_keymiss += 1
        if envs.SGLANG_DEBUG_VESTIGEKV_STATS.get() and self._cap_keymiss % 20 == 1:
            import logging

            logging.getLogger(__name__).info(
                "VKKEYMISS old=%s new=%s", self._scan_key_cur, key
            )
        return self._capture_scan(key, forward_batch, reqs)

    def _capture_scan(self, key, forward_batch, reqs) -> bool:
        import time as _t

        _t0 = _t.perf_counter()
        ok = self._capture_scan_timed(key, forward_batch, reqs)
        self._stats["t_capture"] += _t.perf_counter() - _t0
        self._stats["n_capture"] += 1
        if envs.SGLANG_DEBUG_VESTIGEKV_STATS.get():
            import logging

            logging.getLogger(__name__).info(
                "VKCAP total=%.1fms pack=%.1f warm=%.1f rec=%.1f",
                1e3 * (_t.perf_counter() - _t0),
                1e3 * self._stats.pop("_ph_pack", 0.0),
                1e3 * self._stats.pop("_ph_warm", 0.0),
                1e3 * self._stats.pop("_ph_rec", 0.0),
            )
        return ok

    def _capture_scan_timed(self, key, forward_batch, reqs) -> bool:
        real = forward_batch.out_cache_loc.shape[0]
        bs = forward_batch.seq_lens.shape[0]
        # slots and out_cache_loc arrive in a fresh ForwardBatch every step, so
        # the capture cannot read them where they land; stage them into fixed
        # buffers the graph can bake (two copies per step).
        if self._stage_slots is None or self._stage_slots.shape[0] < bs:
            dev = forward_batch.req_pool_indices.device
            self._stage_slots = torch.zeros(max(bs, 1), dtype=torch.int64, device=dev)
            self._stage_loc = torch.zeros(
                max(bs, 1), dtype=forward_batch.out_cache_loc.dtype, device=dev
            )
        self._stage_slots[:real].copy_(forward_batch.req_pool_indices[:real])
        self._stage_loc[:real].copy_(forward_batch.out_cache_loc)
        slots = self._stage_slots[:real]
        loc = self._stage_loc[:real]
        # Bake one gather width per layer, with headroom so the graph survives
        # D.SCAN_KMAX_HEADROOM more decode steps before it must be recaptured.
        baked = {
            lid: min(
                self._kmax.get(lid, 0) + D.SCAN_KMAX_HEADROOM,
                self._kept_buf[lid].shape[1],
            )
            for lid in self._mla_lids
            if lid in self._kept_buf
        }

        # Stack every (layer, slot) pair once, outside capture. The batched
        # step then runs ~34 kernels where the per-pair loop ran 180 -- the
        # per-node dispatch floor was both the replay time and most of the
        # ~40 ms capture cost. Fire sets are bit-identical (registered test).
        from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack

        pack_pairs, pack_tiers = [], []
        for lid in self._mla_lids:
            if lid in self._qbuf:
                for i in range(real):
                    pack_pairs.append((self._li_map[lid], reqs[i]))
                    pack_tiers.append(self._recall[(reqs[i], lid)]["tier"])
        import time as _t

        # Stash the outgoing graph into the LRU (the ping-pong partner will
        # want it back) instead of discarding; evict beyond capacity to keep
        # the pack-memory footprint bounded, then free before building the
        # replacement (the OOM transient lesson).
        self._stash_active()
        if self._scan_cache is not None and len(self._scan_cache) > 1:
            # The evicted pack's LAST replay may still be executing: freeing
            # its buffers lets the allocator hand those addresses to the new
            # pack while the in-flight GEMM still reads them (coredump: warp
            # illegal address inside the skept cutlass bmm -- a use-after-free
            # introduced by the free-before-rebuild OOM fix). Synchronize
            # before dropping the references; eviction is rare (LRU miss
            # beyond 2 shape classes), so the sync is off the steady path.
            torch.cuda.synchronize()
            while len(self._scan_cache) > 1:
                _, old = self._scan_cache.popitem(last=False)
                old["graph"] = old["pack"] = None
        self._scan_graph = None
        self._scan_batched = None
        torch.cuda.empty_cache()

        _p0 = _t.perf_counter()
        batched = BatchedScanPack(
            pack_pairs,
            pack_tiers,
            self._qbuf_stack,
            self._fetch_stack,
            self._fetch_len_stack,
            self._q_heads,
        )
        torch.cuda.synchronize()
        self._stats["_ph_pack"] = _t.perf_counter() - _p0

        def _run():
            batched.run()
            for lid in self._mla_lids:
                if lid in self._kept_buf:
                    self._pack_csr(lid, slots, loc, real, bs, baked[lid])

        self._scan_key_cur = None
        try:
            _p1 = _t.perf_counter()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    _run()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            self._stats["_ph_warm"] = _t.perf_counter() - _p1
            _p2 = _t.perf_counter()
            # capture_begin/capture_end rather than the torch.cuda.graph context
            # manager: that manager calls torch.cuda.empty_cache() on entry,
            # which drops every cached allocator block. Under a serving
            # mem-fraction that costs ~97 ms per capture and then makes the
            # model re-acquire memory from the driver. sglang's own graph runner
            # takes this same low-level path for the same reason.
            # Each capture takes its own private pool. Sharing one across
            # captures (either graph.pool() from the first capture or a handle
            # from graph_pool_handle()) makes capture_begin trip
            # "use_count > 0 INTERNAL ASSERT FAILED" in the caching allocator
            # once the graph that created it is replaced; the old graph's pool
            # is released when the graph itself is, so a private pool per
            # capture costs one graph's intermediates, not a leak.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(side):
                graph.capture_begin(capture_error_mode="thread_local")
                try:
                    _run()
                finally:
                    graph.capture_end()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            self._stats["_ph_rec"] = _t.perf_counter() - _p2
        except (RuntimeError, torch.OutOfMemoryError) as e:
            # Capture is an optimization, never a correctness requirement: on
            # any refusal (OOM for the graph pool, a stream still busy) fall
            # back to the eager path permanently for this key and say so once.
            import logging

            logging.getLogger(__name__).warning(
                "VestigeKV: tier-2 scan graph capture failed (%s); "
                "falling back to the eager scan path",
                e,
            )
            self._scan_graph = None
            # A single refusal can be transient (a busy stream, a momentary
            # OOM), and latching on the first one costs every later step of the
            # process; three in a row is a property of the deployment.
            self._scan_fails += 1
            self._scan_capture_failed = self._scan_fails >= D.SCAN_CAPTURE_MAX_FAILS
            return False
        self._scan_graph, self._scan_key_cur, self._scan_kmax = graph, key, baked
        self._pack_epoch_synced = self._pack_epoch
        if self._scan_cache is None:
            import collections

            self._scan_cache = collections.OrderedDict()
        self._scan_cache[key] = {
            "graph": graph,
            "pack": batched,
            "kmax": dict(baked),
        }
        self._scan_cache.move_to_end(key)
        while len(self._scan_cache) > 2:
            _, old = self._scan_cache.popitem(last=False)
            old["graph"] = old["pack"] = None
        self._scan_batched = batched  # the graph reads/writes its tensors
        self._scan_fails = 0
        # _run executed three times so far (two warmups plus the capture), and
        # each execution APPENDED this step's row to every request's kept table.
        # Left alone, that leaves duplicate rows in the attended set -- which
        # changes softmax weights, NoPE's multiset invariance notwithstanding --
        # and desynchronizes the host-side kmax bound, which under-counts and
        # eventually makes the pack gather fewer columns than kept_len holds.
        # Rewind the three extra appends; the replay below then performs this
        # step's single real append.
        for lid in baked:
            kl = self._kept_len[lid]
            kl.scatter_(0, slots, (kl.gather(0, slots) - 3).clamp_min_(0))
            self._kmax[lid] = self._kmax.get(lid, 0) + 1
        graph.replay()
        return True

    def _stash_active(self):
        """Park the active graph/pack in the LRU under its key (no-op when
        nothing is active). Contents may be stale; the epoch check resyncs
        via fits()/update() on reactivation."""
        if self._scan_graph is None or self._scan_key_cur is None:
            return
        if self._scan_cache is None:
            import collections

            self._scan_cache = collections.OrderedDict()
        self._scan_cache[self._scan_key_cur] = {
            "graph": self._scan_graph,
            "pack": self._scan_batched,
            "kmax": dict(self._scan_kmax),
        }
        self._scan_cache.move_to_end(self._scan_key_cur)

    def _invalidate_scan(self):
        self._pack_epoch += 1
        if self._scan_cache is not None:
            torch.cuda.synchronize()  # in-flight replay may read these packs
            for old in self._scan_cache.values():
                old["graph"] = old["pack"] = None
            self._scan_cache.clear()
        # Contents (a new request's tiers, a calibration install) no longer
        # drop the capture: the replay path compares tier identities and
        # refreshes the pack in place. What must still reset here is the
        # deferral state and the step cache (prefill changes the batch
        # composition _step_slots keys on). Shape changes and exhausted
        # gather headroom recapture via the key/kmax checks at replay.
        self._scan_key_seen = None
        self._scan_steps = 0
        self._step_cache = None

    def _step_with_stats(self, forward_batch, reqs, key):
        # SGLANG_DEBUG_VESTIGEKV_STATS=1: attribute the per-step host-side cost
        # between the tier-2 scan and the CSR repack, and count how much work
        # each actually does. Synchronizes twice per layer -- measurement only,
        # never on a quoted serving run.
        import time as _t

        st = self._stats
        real = forward_batch.out_cache_loc.shape[0]
        st["steps"] += 1
        # Must take the SAME branch production takes: measuring _recall_step
        # directly would price the eager path on a step that actually replays a
        # captured graph, i.e. instrument a path the server no longer runs.
        torch.cuda.synchronize()
        t0 = _t.perf_counter()
        replayed = self._replay_scan(forward_batch, reqs, key)
        if not replayed:
            for lid in self._mla_lids:
                self._recall_step(lid, forward_batch, reqs)
        tl = _t.perf_counter()  # queue drained on entry -> t0..tl is pure
        torch.cuda.synchronize()  # host dispatch; tl..t1 is GPU-bound tail
        t1 = _t.perf_counter()
        st["t_launch"] += tl - t0
        st["t_scan"] += t1 - t0
        st["replays"] += int(replayed)
        # Separate the replay steps from the eager ones: their averages differ
        # by more than an order of magnitude, so a blended number attributes
        # nothing.
        if replayed:
            st["t_replay"] += t1 - t0
            st["t_replay_host"] += tl - t0
        else:
            st["t_eager"] += t1 - t0
        # The captured graph already packed the CSR. Running it again here
        # appended this step's row to kept_len a SECOND time, so every number
        # this instrument produced after the pack was captured described a
        # corrupted configuration -- the same "instrument takes a different
        # branch than production" defect as before.
        if not replayed:
            for lid in self._mla_lids:
                self._refresh_graph_bufs(lid, forward_batch, reqs)
            torch.cuda.synchronize()
            st["t_pack"] += _t.perf_counter() - t1
        for lid in self._mla_lids:
            if lid in self._qbuf:
                st["scan_calls"] += real
                fl = self._fetch_len.get(lid)
                kl = self._kept_len.get(lid)
                if fl is not None:
                    slots = forward_batch.req_pool_indices[:real].to(torch.int64)
                    st["fetched"] += int(fl.gather(0, slots).sum())
                    st["kept"] += int(kl.gather(0, slots).sum())
                    st["seq"] += int(forward_batch.seq_lens[:real].sum())
        if st["steps"] % 50 == 0:
            self._dump_stats()

    def _dump_stats(self):
        import logging

        st = self._stats
        n = max(st["steps"], 1)
        c = max(st["scan_calls"], 1)
        logging.getLogger(__name__).info(
            "VKSTATS steps=%d layers=%d replay=%.0f%% build=%.1fms x%d cap=%.1fms x%d "
            "caps[key=%d kmax=%d fits=%d] "
            "replay=%.3fms(host %.3f) eager=%.2fms scan=%.2fms/step (dispatch %.2f) "
            "pack=%.2fms/step "
            "scan_calls=%.1f/step fetched=%.0f/call kept=%.0f/call seq=%.0f/call "
            "attended_frac=%.4f",
            st["steps"],
            len(self._mla_lids),
            100.0 * st["replays"] / n,
            1e3 * st["t_build"],
            st["n_build"],
            1e3 * st["t_capture"],
            st["n_capture"],
            self._cap_keymiss,
            self._cap_kmax,
            self._cap_fits,
            1e3 * st["t_replay"] / max(st["replays"], 1),
            1e3 * st["t_replay_host"] / max(st["replays"], 1),
            1e3 * st["t_eager"] / max(n - st["replays"], 1),
            1e3 * st["t_scan"] / n,
            1e3 * st["t_launch"] / n,
            1e3 * st["t_pack"] / n,
            st["scan_calls"] / n,
            st["fetched"] / c,
            st["kept"] / c,
            st["seq"] / c,
            (st["fetched"] + st["kept"]) / max(st["seq"], 1),
        )

    def _check_row_invariant(self, forward_batch):
        # SGLANG_DEBUG_VESTIGEKV_ROWS=1: per-step loud assertion that the attended row
        # set is the intended one. Catches the silent wrong-row-set class (stale
        # slot state, never-built tables, chunk-local seq_len) that unit tests
        # and short-generate smoke tests cannot see. Syncs; debug/CI only.
        full_arm = self._full_arm()
        real = forward_batch.out_cache_loc.shape[0]
        slots = forward_batch.req_pool_indices[:real].to(torch.int64)
        seq = forward_batch.seq_lens[:real]
        for lid in self._local_mla_lids:
            if lid not in self._kept_len:
                raise AssertionError(
                    f"VESTIGE CHECK: layer {lid} kept table never built "
                    f"(prefill did not reach this backend on this rank)"
                )
            n = self._kept_len[lid].gather(0, slots)
            if int((n <= 0).sum()):
                raise AssertionError(
                    f"VESTIGE CHECK: layer {lid} kept_len==0 for an active slot "
                    f"(slot state missing); slots={slots.tolist()}"
                )
            if full_arm:
                # check precedes this step's append: kept_len lags seq_lens by 1
                bad = (n < seq - 1).sum()
                if int(bad):
                    raise AssertionError(
                        f"VESTIGE CHECK: FULL arm but kept_len < seq_len-1 on "
                        f"layer {lid}: n={n.tolist()} seq={seq.tolist()}"
                    )
            else:
                # compressed: kept (+ bounded fetch) well below seq past window
                if lid in self._fetch_len:
                    n = n + self._fetch_len[lid].gather(0, slots)
                mask = seq > D.CHECK_MIN_SEQ_BLOCKS * D.CLOSE_BLOCK
                bad = ((n > (seq * D.CHECK_MAX_KEPT_FRACTION).to(n.dtype)) & mask).sum()
                if int(bad):
                    raise AssertionError(
                        f"VESTIGE CHECK: VESTIGE arm but kept_len ~ seq_len on "
                        f"layer {lid} (compression not applied): "
                        f"n={n.tolist()} seq={seq.tolist()}"
                    )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self.base.init_forward_metadata_in_graph(forward_batch)
        if self._ingraph_pack is not None and forward_batch.forward_mode.is_decode():
            self._ingraph_device_step(forward_batch.seq_lens.shape[0])
        # Graph-recordable per-step refresh (bs=1): append this step's slot into
        # each local MLA layer's kept table and point the CSR at that row. All
        # operands are fixed-address GPU tensors, so this captures and replays
        # with zero python on the decode path. bs>1 uses the out-graph CPU pack.
        if (
            self._kept_buf
            and forward_batch.forward_mode.is_decode()
            and forward_batch.seq_lens.shape[0] == 1
        ):
            # 1-D gather/scatter only: extracting a 0-dim scalar (t[0]) does an
            # internal .item() sync, which invalidates stream capture.
            slot = forward_batch.req_pool_indices[:1].to(torch.int64)
            loc = forward_batch.out_cache_loc[:1]
            for lid in self._local_mla_lids:
                cap = self._kept_buf[lid].shape[1]
                n = self._kept_len[lid].gather(0, slot)
                flat = slot * cap + n
                self._kept_buf[lid].view(-1).scatter_(
                    0, flat.to(torch.int64), loc.to(self._kept_buf[lid].dtype)
                )
                self._kept_len[lid].scatter_(0, slot, n + 1)
                self._indptr1[lid][0:1].copy_(slot * cap)
                self._indptr1[lid][1:2].copy_(flat + 1)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.base.init_cuda_graph_state(max_bs, max_num_tokens)
        self._graph_max_bs = max_bs

    def get_cuda_graph_seq_len_fill_value(self):
        return self.base.get_cuda_graph_seq_len_fill_value()

    def on_after_cuda_graph_warmup(self):
        self.base.on_after_cuda_graph_warmup()

    def shared_read_ends(self, fm: ForwardMode) -> SharedReadEnds:
        return self.base.shared_read_ends(fm)

    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        return self.base.get_indexer_metadata(layer_id, forward_batch)

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs):
        self.base.update_verify_buffers_to_fill_after_draft(spec_info, cuda_graph_bs)

    @property
    def verify_mask(self) -> Optional[VerifyMask]:
        return self.base.verify_mask

    @property
    def data_type(self):
        return self.base.data_type

    @property
    def kv_cache_dtype(self):
        return self.base.kv_cache_dtype

    # ---- prefill: delegate; the kept set is built lazily on first decode ----

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        out = self.base.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )
        self._build_gpu_state(layer, forward_batch)
        return out

    def _build_gpu_state(self, layer, forward_batch):
        # Prefill-time build (sync here is off the decode critical path): the
        # sidecar-residual kept set for every request in this extend batch.
        # Every request passes through extend before decode, so decode can
        # assume the slot state exists (slot reuse re-extends).
        lid = layer.layer_id
        fm = self.base.forward_metadata
        r2t = self.req_to_token_pool.req_to_token
        if lid not in self._kept_buf:
            max_reqs = r2t.shape[0]
            cap = self.base.max_context_len
            dt = fm.kv_indices.dtype if fm is not None else torch.int64
            dev = r2t.device
            self._ensure_kept_stacks(max_reqs, cap, dt, dev)
            self._alloc_recall_bufs(lid, max_reqs, dt, dev)
        kbuf = self.token_to_kv_pool.get_key_buffer(lid)
        kbuf = kbuf.reshape(-1, kbuf.shape[-1])
        slots = forward_batch.req_pool_indices.tolist()
        lens = (
            forward_batch.seq_lens_cpu.tolist()
            if forward_batch.seq_lens_cpu is not None
            else forward_batch.seq_lens.tolist()
        )
        for slot, seq_len in zip(slots, lens):
            seq_len = int(seq_len)
            row_slots = r2t[slot, :seq_len]
            kept = self._arm_aware_kept(row_slots, kbuf, seq_len, layer.v_head_dim)
            n = kept.numel()
            self._kept_buf[lid][slot, :n] = kept.to(self._kept_buf[lid].dtype)
            self._kept_len[lid][slot] = n
            # Arm decode-time closes: the prefix counts as closed, and its
            # sigma record seeds the global rebalance (recomputed here rather
            # than plumbed out of _arm_aware_kept; prefill-path cost).
            closed0 = (seq_len // D.CLOSE_BLOCK) * D.CLOSE_BLOCK
            self._close_state[(slot, lid)] = {
                "closed": closed0,
                "sigma": blockwise_sigma(
                    kbuf[row_slots][:, D.KV_LORA_RANK :], D.CLOSE_BLOCK
                ),
            }
            # Host-side upper bound on kept_len, so the per-step CSR pack needs
            # no `int(lens.max())` readback. Exact by construction: every decode
            # step appends one row to every active request, so bumping the bound
            # by one per step keeps it >= the true max; a new request can only
            # raise it here, where n is already known on the host.
            self._kmax[lid] = max(self._kmax.get(lid, 0), n)
            # Slot reuse: drop the bs>1 tier-2 index state built for the slot's
            # previous occupant; the first decode step rebuilds it for this
            # request. Stale state made bs>1 decode attend the prior request's
            # row set (profiler: VESTIGE attn == FULL attn at bs=16).
            self._tier2.pop((slot, lid), None)
            # Build the recall index HERE, at prefill, alongside the kept table:
            # RecallTier.build costs ~85 ms at S=65k (fp32 prefix gather, SVD,
            # chunked full-cache calibration). Deferring it to decode billed
            # that to the token path (measured: 5 builds x 85 ms inside a
            # 32-step window = 138 -> 50 tok/s). Prefill is where the reference
            # does it and where the sync is free. Rebuilt per chunk-close:
            # the last chunk's build is the one decode uses.
            # Build the tier-2 index exactly once, on the FINAL prefill chunk
            # (prefix complete, and still off the token path). Building on every
            # chunk costs 9x for a 9-chunk prefill; deferring to the first decode
            # step bills the SVD to decode (trace: aten::linalg_svd 83 ms inside
            # the decode window). Both were measured and rejected.
            self._recall[(slot, lid)] = {
                "tier": None,
                "built_at": 0,
                "qcal": [],
                "qpos": [],
                "target": D.N_CAL_START,
            }
            self._collecting = True
            self._invalidate_scan()
            if lid in self._fetch_len:
                self._fetch_len[lid][slot] = 0
            # Deliberately NOT built here: under chunked prefill seq_lens is the
            # running total, not the request length, so "is this the last chunk?"
            # is not decidable from the ForwardBatch (measured: the obvious
            # predicate is always true -> 9x redundant builds). The index is
            # built once on the first decode step for this slot; that is
            # prefill-phase work by semantics, so serving reports must bill it to
            # TTFT, not to steady-state decode throughput.

    def _ensure_kept_stacks(self, max_reqs, cap, dt, dev):
        # Stacked kept tables ([L, R1, CAP] etc.) with the per-lid dict
        # entries as views (the _qbuf_stack pattern): eager call sites keep
        # their dict interface, while the in-graph CSR pack addresses every
        # layer from two kernels (see vestigekv/pack_csr.py).
        if getattr(self, "_kept_stack", None) is not None:
            return
        n = len(self._local_mla_lids)
        li_map = {L: i for i, L in enumerate(self._local_mla_lids)}
        self._kept_stack = torch.zeros(n, max_reqs + 1, cap, dtype=dt, device=dev)
        self._kept_len_stack = torch.zeros(
            n, max_reqs + 1, dtype=torch.int64, device=dev
        )
        for lid, i in li_map.items():
            self._kept_buf[lid] = self._kept_stack[i]
            self._kept_len[lid] = self._kept_len_stack[i]
            if lid not in self._indptr1:
                self._indptr1[lid] = torch.zeros(2, dtype=dt, device=dev)
                self._indptr1[lid][1] = 1  # capture attends one padded row

    def _alloc_recall_bufs(self, lid, max_reqs, idx_dtype, dev):
        if lid in self._qbuf:
            return
        # One stacked allocation per buffer kind, with the per-layer dict
        # entries as views into it. The dict interface stays (the CSR pack and
        # the eager path read it), while the batched step can gather and
        # scatter across every (layer, slot) pair with single indexed ops.
        if self._qbuf_stack is None:
            n = len(self._local_mla_lids)
            self._li_map = {L: i for i, L in enumerate(self._local_mla_lids)}
            # +1 trash row: padded replay lanes and placeholder pack pairs
            # point here, so their reads see zeros and their writes land where
            # nothing is consumed.
            self._trash_slot = max_reqs
            max_reqs = max_reqs + 1
            # bf16, matching q as the model produces it: the fp32 upconvert
            # used to happen INSIDE the model's captured graph (an allocation
            # plus twice the traffic, per MLA layer per step); bf16 -> fp32 is
            # exact, so converting at the scan's read site instead is
            # bit-identical and halves the in-graph copy.
            self._qbuf_stack = torch.zeros(
                n,
                max_reqs,
                self._q_heads,
                self._q_dim,
                dtype=torch.bfloat16,
                device=dev,
            )
            self._fetch_stack = torch.zeros(
                n, max_reqs, self._fetch_w, dtype=idx_dtype, device=dev
            )
            self._fetch_len_stack = torch.zeros(
                n, max_reqs, dtype=torch.int64, device=dev
            )
        li = self._li_map[lid]
        self._qbuf[lid] = self._qbuf_stack[li]
        self._fetch_buf[lid] = self._fetch_stack[li]
        self._fetch_len[lid] = self._fetch_len_stack[li]

    def _full_arm(self) -> bool:
        # Benchmark-only A/B switch (SGLANG_TEST_VESTIGEKV_FULL_ARM_FLAG names a
        # flag file; present -> FULL arm). Unset in production: no file stat.
        path = envs.SGLANG_TEST_VESTIGEKV_FULL_ARM_FLAG.get()
        return bool(path) and os.path.exists(path)

    def _arm_aware_kept(self, row_slots, kbuf, seq_len, v_dim):
        # The kept row set for one request, shared by the bs==1 kept-table build
        # and the bs>1 tier-2 build so both honor the same arm. Benchmark arm
        # switch, read per prefill: /tmp/vestige_full present -> FULL prefix (arm
        # A, == baseline), else sidecar-residual sigma top-m + 4 sinks + recent
        # window (arm B). Both arms then run the identical graph/refresh path.
        if self._full_arm() or self._ingraph_dead:
            # FULL arm, or the in-graph pack was disabled (capacity defect):
            # dense is the only safe row set when nothing will recall.
            return row_slots
        # BLOCKWISE sigma (reference policy semantics): only full CLOSE_BLOCK
        # windows are closed and ranked; the remainder is the unclosed tail,
        # unconditionally attended -- it subsumes the recent-window role
        # (RECENT_WINDOW survives only as the row-invariant checker's scale).
        # See eviction.blockwise_sigma for why the whole-prefix transform was
        # wrong: an S-dependent cutoff, and the only formulation that would
        # ever have needed an incremental FFT.
        closed = (seq_len // D.CLOSE_BLOCK) * D.CLOSE_BLOCK
        if closed == 0:
            return row_slots  # too short to close anything: attend everything
        sigma = blockwise_sigma(kbuf[row_slots][:, v_dim:], D.CLOSE_BLOCK)
        keep = select_kept(sigma, rho=self.rho, closed=closed, sinks=D.SINKS)
        kept_closed = row_slots[:closed][keep.nonzero(as_tuple=True)[0]]
        return torch.cat([kept_closed, row_slots[closed:]])

    # ---- decode: evict (tier-1) + recall (tier-2), both unconditional ----

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        from sglang.srt.model_executor.runner import get_is_capture_mode

        # Reuse the base's fused MLA-decode kernel and KV write verbatim; only
        # swap the attended index set it reads (kept prefix + growing decode
        # tail) for this layer, then restore. num_kv_splits is left as the base
        # sized it (>= the compressed length -> correct, empty splits merge out).
        fm = self.base.forward_metadata
        lid = layer.layer_id
        # In-graph: record this step's expanded query into the fixed-address
        # qbuf so the NEXT step's out-graph recall scan can read it
        # (stale-by-one recall; fixed-shape index_copy_, capture-safe). Padded
        # lanes may overwrite a real slot's qbuf with a padded lane's query;
        # this only perturbs calibration/scan inputs, never row correctness.
        # The FULL benchmark arm attends every row and never scans, so it must
        # not pay for the recall query copy -- otherwise the A/B baseline is
        # polluted by the compressed arm's machinery (measured: FULL 123.7 ->
        # 72.8 tok/s with this copy unconditionally enabled).
        if lid in self._qbuf and not self._full_arm():
            # q arrives already in the absorbed expanded form
            # [bs, H, kv_lora_rank + rope] (see the base's qk_head_dim view).
            bs_q = forward_batch.seq_lens.shape[0]
            slots_q = forward_batch.req_pool_indices[:bs_q].to(torch.int64)
            q_exp = q.view(bs_q, self._q_heads, self._q_dim)
            self._qbuf[lid].index_copy_(0, slots_q, q_exp)
        if get_is_capture_mode():
            # Capture: pure pointer swap to this layer's fixed-address buffers
            # (pre-filled outside capture by the in_capture out-graph hook; any
            # copy or sync here would invalidate stream capture). One uniform
            # binding for every bs: the packed (kept + fetched) CSR buffers.
            self._mla_lids.add(lid)
            bs = forward_batch.seq_lens.shape[0]
            bufs = self._graph_bufs[lid]
            indptr, indices = bufs["indptr"][: bs + 1], bufs["indices"]
        elif lid in self._graph_bufs:
            bs = forward_batch.seq_lens.shape[0]
            bufs = self._graph_bufs[lid]
            indptr, indices = bufs["indptr"][: bs + 1], bufs["indices"]
        else:
            indptr, indices = self._compressed_indices(layer, forward_batch)
        saved = (fm.kv_indptr, fm.kv_indices)
        fm.kv_indptr, fm.kv_indices = indptr, indices
        try:
            return self.base.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        finally:
            fm.kv_indptr, fm.kv_indices = saved

    # ---- VestigeKV core: tier-1 sidecar-residual eviction over the latent pool ----

    def _prefill_graph_bufs_for_capture(self, forward_batch):
        # All layers get a buffer (only the MLA layers' entries are used); the
        # dummy batch's base metadata is copied in so capture records sane
        # contents at these fixed addresses.
        fm = self.base.forward_metadata
        bs = forward_batch.seq_lens.shape[0]
        n = int(fm.kv_indptr[bs])
        r2t = self.req_to_token_pool.req_to_token
        for lid in self._local_mla_lids:
            if lid not in self._kept_buf:
                cap_row = self.base.max_context_len
                self._ensure_kept_stacks(
                    r2t.shape[0], cap_row, fm.kv_indices.dtype, r2t.device
                )
            self._alloc_recall_bufs(lid, r2t.shape[0], fm.kv_indices.dtype, r2t.device)
            bufs = self._graph_bufs.get(lid)
            if bufs is None:
                # Stacked across layers (views per lid): the in-graph CSR pack
                # addresses all layers from one kernel launch. +1: the legacy
                # torch pack scatters masked-out entries into a trash slot at
                # the end (see _refresh_graph_bufs); the Triton pack masks.
                if getattr(self, "_gb_indices_stack", None) is None:
                    nl = len(self._local_mla_lids)
                    mb1 = max(self._graph_max_bs, 1) + 1
                    cap = max(self._graph_max_bs, 1) * self.base.max_context_len + 1
                    self._gb_indptr_stack = fm.kv_indptr.new_zeros(nl, mb1)
                    self._gb_indices_stack = fm.kv_indices.new_zeros(nl, cap)
                li_ = self._li_map[lid]
                bufs = {
                    "indptr": self._gb_indptr_stack[li_],
                    "indices": self._gb_indices_stack[li_],
                }
                self._graph_bufs[lid] = bufs
            bufs["indptr"][: bs + 1].copy_(fm.kv_indptr[: bs + 1])
            bufs["indptr"][bs + 1 :].fill_(fm.kv_indptr[bs])
            if n > 0:
                bufs["indices"][:n].copy_(fm.kv_indices[:n])
        if (
            envs.SGLANG_ENABLE_VESTIGEKV_INGRAPH_SCAN.get()
            and self._ingraph_pack is None
        ):
            self._build_ingraph_pack()

    # ---- in-graph scan: the recall step as nodes of the decode model graph ----
    #
    # Gate 19 isolated the second per-step cudaGraphLaunch (the scan graph) as
    # a ~0.9 ms batch-independent fixed cost -- the sole reason bs=1 sat below
    # 1.0x while bs>=2 cleared it. These paths bake the scan + CSR pack into
    # the model graph itself: one launch per step, no scan-graph lifecycle.
    #
    # Correctness rests on three invariants:
    # 1. Capacity, not recapture: the pack and every buffer the baked kernels
    #    read are sized for the worst case at startup; content refreshes go
    #    through update()/copy_ at fixed addresses.
    # 2. Placeholder no-ops: unoccupied pack pairs and padded batch lanes
    #    carry zero lengths and point at the trash slot, so the always-running
    #    kernels fire nothing and write where nothing reads.
    # 3. Stale-by-one is preserved: the scan sits at the top of the captured
    #    step, before any layer overwrites qbuf.

    def _build_ingraph_pack(self):
        from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack

        dev = self._qbuf_stack.device
        maxbs = max(self._graph_max_bs, 1)
        if self._stage_slots is None or self._stage_slots.shape[0] < maxbs:
            self._stage_slots = torch.full(
                (maxbs,), self._trash_slot, dtype=torch.int64, device=dev
            )
            self._stage_loc = torch.zeros(maxbs, dtype=torch.int64, device=dev)
        max_ctx = self.base.max_context_len
        # Kept rows are bounded by tier-1's keep rate plus the un-closed tail
        # (blocks close every CLOSE_BLOCK); the slack absorbs close latency.
        nkm = int(D.RHO * max_ctx) + 3 * D.CLOSE_BLOCK
        self._ingraph_pack = BatchedScanPack.at_capacity(
            len(self._local_mla_lids) * maxbs,
            max(1, min(nkm, max_ctx)),
            max_ctx,
            self.index_rank,
            self._q_heads,
            self._qbuf_stack,
            self._fetch_stack,
            self._fetch_len_stack,
            self._trash_slot,
        )

    def _ingraph_device_step(self, bs):
        # Runs inside run_once during capture, so the whole recall step --
        # scan + fetch + the all-layer CSR pack -- replays as nodes of the
        # ONE decode graph launch. No host code runs here at replay time.
        from sglang.srt.layers.attention.vestigekv.pack_csr import pack_csr_all_layers

        self._ingraph_pack.run(p_live=len(self._local_mla_lids) * bs)
        pack_csr_all_layers(
            self._stage_slots[:bs],
            self._stage_loc[:bs],
            self._kept_stack,
            self._kept_len_stack,
            self._fetch_len_stack,
            self._fetch_stack,
            self._gb_indices_stack,
            self._gb_indptr_stack,
        )

    def _ingraph_host_step(self, forward_batch, reqs):
        bs = forward_batch.seq_lens.shape[0]
        real = forward_batch.out_cache_loc.shape[0]
        self._stage_slots[:real].copy_(
            forward_batch.req_pool_indices[:real].to(torch.int64), non_blocking=True
        )
        self._stage_loc[:real].copy_(
            forward_batch.out_cache_loc.to(torch.int64), non_blocking=True
        )
        if real < bs:
            self._stage_slots[real:bs].fill_(self._trash_slot)
            self._stage_loc[real:bs].zero_()
        if self._full_arm():
            if not self._ingraph_full_armed:
                # FULL arm: kept_buf already holds every row (_arm_aware_kept),
                # so silence the scan and clear stale fires -- a fired row
                # spliced next to its dense copy would be attended twice.
                self._ingraph_pack.update([], [])
                self._fetch_len_stack.zero_()
                self._ingraph_full_armed = True
        elif (
            self._pack_epoch != self._pack_epoch_synced or self._ingraph_full_armed
        ) and not self._ingraph_dead:
            self._ingraph_full_armed = False
            pairs, tiers = [], []
            for lid in self._local_mla_lids:
                if lid not in self._qbuf:
                    continue
                for i in range(real):
                    st = self._recall.get((reqs[i], lid))
                    tier = st.get("tier") if st is not None else None
                    if tier is not None:
                        pairs.append((self._li_map[lid], reqs[i]))
                        tiers.append(tier)
            if not self._ingraph_pack.fits(pairs, tiers):
                self._ingraph_disable(forward_batch, reqs)
            else:
                self._ingraph_pack.update(pairs, tiers)
            self._pack_epoch_synced = self._pack_epoch
        # The captured pack appends one row per padded lane per layer to the
        # trash slot's kept table; reset it before it can reach capacity
        # (growth <= layers per step, capacity is max_context_len).
        self._ingraph_steps += 1
        if self._ingraph_steps % 1024 == 0:
            self._kept_len_stack[:, self._trash_slot] = 0

    def _ingraph_disable(self, forward_batch, reqs):
        # A tier outgrew the capacity pack -- impossible under the sizing
        # invariant, so treat it as a defect signal, but stay CORRECT: the
        # baked kernels keep replaying, so silence every pair and fall back to
        # serving each live request its dense row set through kept_buf.
        import logging

        logging.getLogger(__name__).error(
            "VestigeKV: in-graph pack capacity exceeded; degrading to dense "
            "serving (report this -- the sizing invariant is broken)"
        )
        self._ingraph_dead = True
        self._ingraph_pack.update([], [])
        self._fetch_len_stack.zero_()
        r2t = self.req_to_token_pool.req_to_token
        lens = self._seq_lens_host(forward_batch)
        real = forward_batch.out_cache_loc.shape[0]
        for lid in self._local_mla_lids:
            if lid not in self._kept_buf:
                continue
            for i in range(real):
                slot, n = reqs[i], int(lens[i])
                self._kept_buf[lid][slot, :n] = r2t[slot, :n].to(
                    self._kept_buf[lid].dtype
                )
                self._kept_len[lid][slot] = n

    def _recall_step(self, lid, forward_batch, reqs):
        # Per-step recall: a required part of the algorithm, with no switch.
        # Stale-by-one: scans with the PREVIOUS step's query recorded in-graph
        # into qbuf; fired pool rows land in the fixed-address fetch_buf that
        # the packed CSR splices next.
        if lid not in self._qbuf or self._full_arm():
            return
        real = forward_batch.out_cache_loc.shape[0]
        for i in range(real):
            slot = reqs[i]
            st = self._recall.get((slot, lid))
            if st is None:
                continue  # slot never prefilled through this backend
            if st["tier"] is None:
                continue
            # Device-only fire+fetch: no host sync on the decode path (a
            # per-layer .numel()/bool() readback serialized the pipeline and
            # cost ~9 ms/step -- profiler: GPU busy 6.4 ms, gap 279 ms).
            st["tier"].query_fixed(
                self._qbuf[lid][slot],
                self._fetch_buf[lid],
                self._fetch_len[lid],
                slot,
            )

    def _seq_lens_host(self, forward_batch):
        """Host-side seq_lens without a device readback. The scheduler ships
        seq_lens_cpu alongside the device tensor on the decode path; falling
        back to the device tensor is a per-request D2H sync, acceptable only
        because it is the exceptional path."""
        cpu = getattr(forward_batch, "seq_lens_cpu", None)
        return cpu if cpu is not None else forward_batch.seq_lens

    def _maybe_close_blocks(self, forward_batch, reqs):
        """Decode-time compression events (reference policy semantics).

        Every D.CLOSE_BLOCK decoded tokens per request, the newest block is
        closed: sigma over its sidecars joins the request's global sigma
        record, tier-1 re-selects top-(rho * closed) over ALL closed rows
        (global rebalance -- rows move both ways, matching the reference
        engine's constant-m schedule), the kept table is rewritten to
        [selected | tail], and the recall index refreshes its archive through
        the live projection caches so every evicted row stays recallable.

        Synchronous by design: ~1 ms every 4096 steps. The pack picks up the
        new membership through the tier's version bump (an in-place refresh
        would otherwise be invisible to the (id, version) sync key), and the
        archive growth trips pack.fits into a recapture when the headroom is
        out -- both existing mechanisms.
        """
        real = forward_batch.out_cache_loc.shape[0]
        seq_lens = self._seq_lens_host(forward_batch)
        for i in range(real):
            slot = reqs[i]
            seq_len = int(seq_lens[i])
            for lid in self._local_mla_lids:
                cl = self._close_state.get((slot, lid))
                if cl is None:
                    continue
                while seq_len - cl["closed"] >= D.CLOSE_BLOCK:
                    self._close_one_block(slot, lid, cl, seq_len)

    def _close_one_block(self, slot, lid, cl, seq_len):
        kbuf = self.token_to_kv_pool.get_key_buffer(lid)
        kbuf = kbuf.reshape(-1, kbuf.shape[-1])
        r2t = self.req_to_token_pool.req_to_token
        c0 = cl["closed"]
        c1 = c0 + D.CLOSE_BLOCK
        block_slots = r2t[slot, c0:c1].to(torch.int64)
        sigma = sidecar_sigma(kbuf[block_slots][:, D.KV_LORA_RANK :])
        cl["sigma"] = torch.cat([cl["sigma"], sigma])
        cl["closed"] = c1
        # global rebalance over every closed row
        m = max(1, round(self.rho * c1))
        keep = torch.zeros(c1, dtype=torch.bool, device=sigma.device)
        keep[cl["sigma"].topk(min(m, c1)).indices] = True
        keep[: D.SINKS] = True
        closed_slots = r2t[slot, :c1].to(torch.int64)
        kept_slots = closed_slots[keep]
        tail_slots = r2t[slot, c1:seq_len].to(torch.int64)
        new_kept = torch.cat([kept_slots, tail_slots])
        n = new_kept.shape[0]
        self._kept_buf[lid][slot, :n] = new_kept.to(self._kept_buf[lid].dtype)
        self._kept_len[lid][slot] = n
        # exact host-side bound: the close is the one place kept_len is fully
        # known on the host again, so the +1-per-step estimate resets here
        self._kmax[lid] = max(self._kmax.get(lid, 0), n)
        st = self._recall.get((slot, lid))
        tier = st.get("tier") if st else None
        if tier is not None and tier.built:
            cached = 0 if tier._pos_all is None else tier._pos_all.shape[0]
            if cached < c1:
                delta = closed_slots[cached:c1]
                tier.extend_closed(kbuf[delta], delta)
            tier.refresh_membership(keep, kbuf[kept_slots])
            self._pack_epoch += 1

    def _collect_calibration(self, forward_batch, reqs):
        """Keep tier 2 serving from the first decode step while calibrating it
        on real queries.

        Calibration needs queries whose best-scoring row was evicted -- that is
        the `hard` set the z-ladder and the entropy gate are fitted on. The
        reference engine takes strided PREFILL queries, but an sglang attention
        backend cannot: MLA prefill runs the un-absorbed path, so q arrives as
        [tokens, H, 192] and absorbing 128 -> 512 needs W_kc, a model weight the
        backend never sees. Decode queries ARE absorbed ([bs, H, 576]) and the
        in-graph hook writes them into _qbuf every step, so a request's first
        n_cal decode steps supply them at no extra cost.

        Waiting n_cal steps to build anything would leave tier 2 off at the start
        of every request, and tier 2 has no off state -- at long context the kept
        tier alone is exactly what the method claims is insufficient. So the
        index is built twice: a provisional one on the first decode step from
        cache-row proxies, and the real one once n_cal live queries exist.

        The proxies are degenerate as calibration -- a cache row used as its own
        query has itself as causal argmax, and its own position is always inside
        the recent window, hence always kept. Measured, they returned zp anywhere
        from 0.0 to 8.0 for the same layer across consecutive requests, and
        zp=0 means no certificate inflation at all, i.e. firing too little. So
        the provisional index does not pretend to calibrate: it takes the most
        conservative rung with the gate open, which over-fetches, costing
        latency and never recall.
        """
        real = forward_batch.out_cache_loc.shape[0]
        pending = False
        for lid in self._mla_lids:
            if lid not in self._qbuf:
                continue
            for i in range(real):
                slot = reqs[i]
                st = self._recall.get((slot, lid))
                if st is None or st.get("qcal") is None:
                    continue
                prefix_len = int(self._seq_lens_host(forward_batch)[i]) - 1
                st["qcal"].append(self._qbuf[lid][slot].clone())
                st["qpos"].append(prefix_len)
                if st["tier"] is None:
                    # The provisional build stays SYNCHRONOUS: it is ~0.7 ms
                    # (sketch basis cached) and tier 2 must serve from the
                    # first decode step.
                    self._build_index(slot, lid, prefix_len + 1, st, proxy=True)
                    self._capture_asap = True
                    pending = True
                elif len(st["qcal"]) >= st["target"] and "job" not in st:
                    # The CALIBRATED build's label GEMM is O(n_cal*H*S) -- 7 ms
                    # at 64k, 28 ms at 256k, ten builds per request -- and on
                    # the decode critical path it was the whole difference
                    # between the measured and the theoretical speedup. It goes
                    # to the side-stream worker; the provisional index keeps
                    # serving until the result is installed (next step, main
                    # thread, atomic).
                    st["job"] = self._enqueue_build(slot, lid, prefix_len + 1, st)
                    pending = True
                else:
                    pending = True
        if self._install_finished_builds():
            # Sticky: the install may land on a step whose collection scan
            # already counted this layer as pending; the recapture then fires
            # on the first later step with nothing left in flight.
            self._needs_recapture = True
        self._collecting = pending or bool(self._build_jobs)
        if self._needs_recapture and not self._collecting:
            # ONE recapture per request, once every layer's calibration has
            # settled. Firing per-install (or on a momentary empty job queue
            # between predictive-window rounds) cost ~4 captures/request at
            # 256k, ~60 ms each -- most of the residual S-slope.
            self._needs_recapture = False
            # Content swap, not a shape change: bump the epoch so the replay
            # fast path resyncs via fits()->update() in place. Dropping the
            # graph here caused capture churn under concurrency (bs4: 139
            # captures / 27 s over 4050 steps -- the whole ITL collapse);
            # update() cannot handle a shape-class change, and fits() failing
            # falls back to a real recapture on its own.
            self._pack_epoch += 1
            # The step cache may hold key=None from before this tier existed
            # (the recovery _invalidate_scan used to provide): drop it so the
            # next step recomputes the key. Without this, a stable bs=1
            # signature cached None forever and decode stayed eager for the
            # whole request (bs1 ITL 8.2 -> 17.6 regression).
            self._step_cache = None
            self._capture_asap = True

    def _enqueue_build(self, slot, lid, seq_len, st):
        """Snapshot the build inputs on the MAIN stream and hand them to the
        side-stream worker.

        The snapshot is what makes this safe: the kept table row is appended
        in-graph every step, so the worker gets a CLONE taken here (ordered
        after this step's replay on the main stream, fenced by an event), and
        the pool rows it reads (row_slots[:seq_len]) are prefix rows, immutable
        after prefill. The worker touches nothing on the backend; installation
        happens back on the main thread in _install_finished_builds.
        """
        import threading

        r2t = self.req_to_token_pool.req_to_token
        n_kept = int(self._kept_len[lid][slot])
        job = {
            "slot": slot,
            "lid": lid,
            "seq_len": seq_len,
            "row_slots": r2t[slot, :seq_len].to(torch.int64).clone(),
            "kept": self._kept_buf[lid][slot, :n_kept].to(torch.int64).clone(),
            "st": st,  # identity token: a re-prefill REPLACES the state dict
            "qcal": list(st["qcal"]),
            "qpos": list(st["qpos"]),
            "v_init": self._vcache.get(lid),
            "ready": torch.cuda.Event(),
            "done": threading.Event(),
            "tier": None,
            "stats": None,
            "error": None,
        }
        job["ready"].record()  # snapshot visible to the side stream after this
        if self._build_worker is None:
            self._build_stream = torch.cuda.Stream()
            self._build_worker = threading.Thread(
                target=self._build_worker_loop, daemon=True
            )
            self._build_worker.start()
        self._build_jobs.append(job)
        self._build_queue.put(job)
        return job

    def _build_worker_loop(self):
        from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

        while True:
            job = self._build_queue.get()
            try:
                with torch.cuda.stream(self._build_stream):
                    self._build_stream.wait_event(job["ready"])
                    kbuf = self.token_to_kv_pool.get_key_buffer(job["lid"])
                    kbuf = kbuf.reshape(-1, kbuf.shape[-1])
                    keep = torch.isin(job["row_slots"], job["kept"])
                    q_cal = torch.stack(job["qcal"]).float()
                    q_pos = torch.tensor(
                        job["qpos"], device=kbuf.device, dtype=torch.long
                    )
                    tier = RecallTier(r=self.index_rank, topj=self.topj)
                    stats = tier.build(
                        kbuf,
                        job["row_slots"],
                        keep,
                        q_cal.contiguous(),
                        q_pos,
                        conservative=False,
                        v_init=job["v_init"],
                    )
                    tier.arch = job["row_slots"][tier.arch]
                self._build_stream.synchronize()
                job["tier"], job["stats"] = tier, stats
            except Exception as e:  # provisional keeps serving; never crash
                job["error"] = e
            job["done"].set()

    def _install_finished_builds(self):
        """Main-thread half of the async build: adopt finished tiers.

        Runs once per decode step. Installation is what the capture lifecycle
        keys off (_invalidate_scan drops the old graph; the next step's
        _capture_scan restacks the batched pack from the NEW tier), so it must
        happen here and never in the worker.
        """
        if not self._build_jobs:
            return False
        installed = False
        remaining = []
        for job in self._build_jobs:
            if not job["done"].is_set():
                remaining.append(job)
                continue
            slot, lid = job["slot"], job["lid"]
            st = self._recall.get((slot, lid))
            if st is not job["st"] or st.get("qcal") is None:
                # The slot was re-prefilled while the build ran: prefill
                # replaces the state dict, so an identity mismatch means this
                # tier belongs to the slot's PREVIOUS occupant. Installing it
                # would hand the new request the old request's archive.
                continue
            st.pop("job", None)
            if job["error"] is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "VestigeKV: async calibrated build failed (%s); the "
                    "provisional index keeps serving (over-fetches, never "
                    "under-recalls)",
                    job["error"],
                )
                st["qcal"] = st["qpos"] = None
                continue
            stats = job["stats"]
            if lid not in self._vcache:
                self._vcache[lid] = job["tier"].V
            if envs.SGLANG_DEBUG_VESTIGEKV_STATS.get():
                import logging

                logging.getLogger(__name__).info(
                    "VKCAL lid=%s slot=%s seq=%s proxy=False async=True %s",
                    lid,
                    slot,
                    job["seq_len"],
                    stats,
                )
            window = len(job["qcal"])
            need_more = stats.get("need_more_hard") and window < D.N_CAL_MAX
            if need_more:
                # Predict the window from the measured hard-rate instead of
                # blind doubling: at S=256k the rate is ~0.3%, so 8->16->32->64
                # ran four expensive rebuilds only to end at the same Z_MAX
                # clamp the provisional index already served with. If the
                # conformal requirement is not reachable within N_CAL_MAX,
                # install the clamped-but-real-basis tier NOW and stop.
                import math

                n_hard = max(int(stats.get("n_hard") or 0), 1)
                needed = math.ceil(D.min_hard() * window / n_hard)
                if needed > D.N_CAL_MAX:
                    need_more = False  # unreachable: adopt the clamped tier
                else:
                    st["target"] = min(
                        max(2 ** math.ceil(math.log2(needed)), window * 2),
                        D.N_CAL_MAX,
                    )
            if not need_more:
                # Install silently: the captured graph replays against the
                # batched pack's own stacked COPIES, so serving stays on the
                # previous parameters until the single coalesced recapture
                # below -- one capture for all layers instead of one each
                # (5 extra ~100 ms captures per request at 256k).
                st["tier"], st["built_at"] = job["tier"], job["seq_len"]
                st["qcal"] = st["qpos"] = None
                installed = True
        self._build_jobs = remaining
        return installed

    def _build_index(self, slot, lid, seq_len, st, proxy: bool):
        import time as _t

        _t0 = _t.perf_counter()
        stats = self._build_index_timed(slot, lid, seq_len, st, proxy)
        self._stats["t_build"] += _t.perf_counter() - _t0
        self._stats["n_build"] += 1
        return stats

    def _build_index_timed(self, slot, lid, seq_len, st, proxy: bool):
        """Build (or rebuild) the tier-2 index. Runs off the token path, where
        a sync costs nothing. See _collect_calibration for what `proxy` means."""
        from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

        kbuf = self.token_to_kv_pool.get_key_buffer(lid)
        kbuf = kbuf.reshape(-1, kbuf.shape[-1])
        r2t = self.req_to_token_pool.req_to_token
        row_slots = r2t[slot, :seq_len].to(torch.int64)
        n_kept = int(self._kept_len[lid][slot])
        kept = self._kept_buf[lid][slot, :n_kept].to(torch.int64)
        keep = torch.isin(row_slots, kept)
        if proxy:
            n_cal = min(D.N_CAL_START, max(seq_len - 1, 1))
            q_cal = (
                kbuf[row_slots[-n_cal:]]
                .float()
                .unsqueeze(1)
                .expand(n_cal, self._q_heads, D.LATENT_DIM)
            )
            q_pos = torch.arange(
                seq_len - n_cal, seq_len, device=row_slots.device, dtype=torch.long
            )
        else:
            q_cal = torch.stack(st["qcal"]).float()  # [n_cal, H, 576]
            q_pos = torch.tensor(st["qpos"], device=row_slots.device, dtype=torch.long)
        tier = RecallTier(r=self.index_rank, topj=self.topj)
        stats = tier.build(
            kbuf,
            row_slots,
            keep,
            q_cal.contiguous(),
            q_pos,
            conservative=proxy,
            # Every build after a layer's first calibrated one reuses that
            # basis. The certificate is a Cauchy-Schwarz bound on the sketch
            # truncation error and is sound for ANY orthonormal basis -- a
            # stale basis can only loosen rho (fire slightly more), never miss
            # -- while the eigendecomposition it replaces was ~3 ms per layer
            # per request, the largest single item left in the build.
            v_init=self._vcache.get(lid),
        )
        if not proxy and lid not in self._vcache:
            self._vcache[lid] = tier.V
        if envs.SGLANG_DEBUG_VESTIGEKV_STATS.get():
            import logging

            logging.getLogger(__name__).info(
                "VKCAL lid=%s slot=%s seq=%s proxy=%s %s",
                lid,
                slot,
                seq_len,
                proxy,
                stats,
            )
        tier.arch = row_slots[tier.arch]
        # Update in place: replacing the dict would drop the qcal/qpos keys the
        # calibration collector uses, so the provisional index would install
        # itself and then never be replaced (observed: proxy=True builds only).
        st["tier"], st["built_at"] = tier, seq_len
        self._pack_epoch += 1  # content swap; see _install_finished_builds
        self._step_cache = None  # may hold key=None from the pre-tier step
        return stats

    def _refresh_graph_bufs(self, lid, forward_batch, reqs):
        bs = forward_batch.seq_lens.shape[0]
        loc = forward_batch.out_cache_loc
        real_bs = loc.shape[0]
        slots = forward_batch.req_pool_indices[:real_bs].to(torch.int64)
        self._pack_csr(lid, slots, loc, real_bs, bs, self._kmax_step(lid))

    def _kmax_step(self, lid):
        # Bound used by this step's pack, advanced by one because the step
        # appends one row to every active request. Kept host-side so the pack
        # never reads lens.max() back from the device.
        kmax = min(self._kmax.get(lid, 0) + 1, self._kept_buf[lid].shape[1])
        self._kmax[lid] = kmax
        return kmax

    def _pack_csr(self, lid, slots, loc, real_bs, bs, kmax):
        # bs > 1 graph replay: append this step's slot to each request's kept
        # table, then repack the CSR into the fixed-address buffers the captured
        # graph reads. Vectorized (~12 tensor ops per layer); the previous
        # per-request python loop cost ~4 ms/step at bs=16.
        # seq_lens is padded to the captured graph bs; out_cache_loc carries only
        # the real (unpadded) requests -- see build_replay_fb_view. Padded CSR
        # slots get one reserved pad row (slot 0); their output is discarded.
        bufs = self._graph_bufs[lid]
        kept_buf, kept_len = self._kept_buf[lid], self._kept_len[lid]
        cap = kept_buf.shape[1]
        # batched append of this step's slot (same pattern as the bs=1 in-graph
        # hook): kept_buf[slot, n_slot] = loc; kept_len[slot] += 1
        dev = kept_buf.device
        n = kept_len.gather(0, slots)
        kept_buf.view(-1).scatter_(
            0, (slots * cap + n).to(torch.int64), loc.to(kept_buf.dtype)
        )
        lens = n + 1
        kept_len.scatter_(0, slots, lens)
        # pack: gather each request's kept rows [slot, :len] plus its fired
        # recall rows [slot, :fetch_len] into one contiguous CSR. Intra-request
        # order is irrelevant (NoPE multiset invariance), so the two segments
        # concatenate per request.
        #
        # Every shape here is a host-side constant and every offset is computed
        # on the device: the previous form read back lens.max(), f_lens.max()
        # and lens_tot.sum(), and selected with a boolean mask (a fourth,
        # implicit readback), which is 4 pipeline drains per layer -- 3.5 of the
        # 4.5 ms/step of host-side cost once the tier-2 scan was captured
        # (VKSTATS, S=64k bs=1). Masked-out entries scatter into the trash slot
        # reserved at the end of the indices buffer.
        if lid in self._fetch_len:
            f_lens = self._fetch_len[lid].gather(0, slots)
            fw = self._fetch_buf[lid].shape[1]
            fetch_flat = self._fetch_buf[lid].view(-1)
        else:  # recall buffers not allocated (unit fakes): zero-fetch
            f_lens = torch.zeros_like(lens)
            fw = 0
            fetch_flat = None
        lens_tot = lens + f_lens
        starts = torch.zeros_like(lens_tot)
        starts[1:] = lens_tot.cumsum(0)[:-1]
        total = lens_tot.sum()
        trash = bufs["indices"].shape[0] - 1

        col = torch.arange(kmax, device=dev)
        src = (
            kept_buf.view(-1)
            .gather(0, (slots[:, None] * cap + col[None, :]).reshape(-1))
            .view(real_bs, kmax)
        )
        dst = torch.where(
            col[None, :] < lens[:, None], starts[:, None] + col[None, :], trash
        )
        if fw:
            fcol = torch.arange(fw, device=dev)
            src_f = fetch_flat.gather(
                0, (slots[:, None] * fw + fcol[None, :]).reshape(-1)
            ).view(real_bs, fw)
            dst_f = torch.where(
                fcol[None, :] < f_lens[:, None],
                starts[:, None] + lens[:, None] + fcol[None, :],
                trash,
            )
            src = torch.cat([src.reshape(-1), src_f.reshape(-1)])
            dst = torch.cat([dst.reshape(-1), dst_f.reshape(-1)])
        bufs["indices"].scatter_(0, dst.reshape(-1), src.reshape(-1))
        n_pad = bs - real_bs
        if n_pad > 0:
            pad_dst = total + torch.arange(n_pad, device=dev)
            bufs["indices"].scatter_(
                0, pad_dst, torch.zeros(n_pad, dtype=kept_buf.dtype, device=dev)
            )
        indptr = torch.zeros(bs + 1, dtype=torch.int64, device=dev)
        indptr[1 : real_bs + 1] = lens_tot.cumsum(0)
        if n_pad > 0:
            indptr[real_bs + 1 :] = total + torch.arange(1, n_pad + 1, device=dev)
        bufs["indptr"][: bs + 1].copy_(indptr)
        bufs["indptr"][bs + 1 :] = total + n_pad

    def _compressed_indices(self, layer, forward_batch):
        # Incremental per-(req, layer) index buffer: [kept..., tail...] built once
        # (one-time sync), then each step appends this step's slot from
        # out_cache_loc (no GPU->CPU sync). bs=1 returns a zero-copy narrow.
        lid = layer.layer_id
        fm = self.base.forward_metadata
        bs = forward_batch.seq_lens.shape[0]
        parts, lens = [], []
        for i in range(bs):
            req = int(forward_batch.req_pool_indices[i])
            st = self._tier2.get((req, lid))
            if st is None:
                st = self._build_index_buffer(req, lid, forward_batch, i, fm)
            buf, n = st["buf"], st["n"]
            if n >= buf.numel():
                buf = torch.cat([buf, buf.new_zeros(buf.numel())])
                st["buf"] = buf
            buf[n] = forward_batch.out_cache_loc[i]
            st["n"] = n = n + 1
            parts.append(buf[:n])
            lens.append(n)
        if bs == 1:
            kv_indptr = fm.kv_indptr.new_zeros(2)
            kv_indptr[1] = lens[0]
            return kv_indptr, parts[0]
        kv_indices = torch.cat(parts)
        kv_indptr = fm.kv_indptr.new_zeros(bs + 1)
        kv_indptr[1:] = torch.as_tensor(lens, device=kv_indptr.device).cumsum(0)
        return kv_indptr, kv_indices

    def _build_index_buffer(self, req, lid, forward_batch, i, fm):
        # One-time per (req, layer). Source the kept set from the prefill-built
        # kept_buf/kept_len tables (single source of truth; pure GPU copy) --
        # recomputing sigma here put 7 x bs rFFTs over the whole prefix into the
        # FIRST decode step, which at bs=16/S=64k ate the entire speedup.
        if lid in self._kept_buf and int(self._kept_len[lid][req]) > 0:
            n_kept = int(self._kept_len[lid][req])
            head = self._kept_buf[lid][req, :n_kept].to(fm.kv_indices.dtype)
        else:
            # Fallback (slot never extended through this backend): compute once.
            kbuf = self.token_to_kv_pool.get_key_buffer(lid)
            kbuf = kbuf.reshape(-1, kbuf.shape[-1])
            r2t = self.req_to_token_pool.req_to_token
            seq_len = int(forward_batch.seq_lens[i])
            kept_st = self._build_kept(
                req, lid, seq_len, kbuf, r2t, kbuf.shape[-1] - 64
            )
            head = torch.cat(
                [kept_st["kept"], r2t[req, kept_st["prefix_len"] : seq_len]]
            ).to(fm.kv_indices.dtype)
        buf = head.new_zeros(head.numel() + _TAIL_CAPACITY)
        buf[: head.numel()] = head
        st = {"buf": buf, "n": head.numel() - 1}  # -1: current step re-appends its slot
        self._tier2[(req, lid)] = st
        return st

    def _build_kept(self, req, lid, seq_len, kbuf, r2t, v_dim):
        # keep the m=round(rho*seq_len) most anomalous rows by sidecar-residual
        # sigma, plus sinks and a recent window; the decode tail is appended at
        # query time so it is always attended. Same arm-aware selector as the
        # bs==1 path, so the FULL benchmark arm is honored here too.
        slots = r2t[req, :seq_len]
        kept = self._arm_aware_kept(slots, kbuf, seq_len, v_dim)
        st = {"kept": kept, "prefix_len": seq_len}
        self._tier2[(req, lid)] = st
        return st
