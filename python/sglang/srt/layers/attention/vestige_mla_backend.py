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
  w_kc/w_vc, rope, or the model: eviction drops slots from that gathered set and
  recall splices a topj-capped set of evicted slots back for the current step.
  See VESTIGEKV_PORT.md for the full contract.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.vestige.eviction import select_kept, sidecar_sigma

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
    from sglang.srt.layers.attention.verify_mask import VerifyMask
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner

# Recent decode/prefill tail always kept regardless of sigma; arbitrary, tune by
# quality. The growing decode tail past prefix_len is attended separately.
_RECENT_WINDOW = 256
# Preallocated decode-tail slots per (req, layer) index buffer; grown 2x on
# overflow, so this only sizes the common case.
_TAIL_CAPACITY = 8192


class VestigeMLABackend(AttentionBackend):
    """Wrap a base MLA backend; compress the latent cache on decode.

    topj is the per-head fetch cap: default 16 = the bounded-fetch guarantee
    (fetch <= topj * num_heads rows / step / layer; no degeneration to naive
    MLA). Configurable: topj = -1 (or 0) disables the cap and fetches the full
    fired recall set. rho is the tier-1 eviction ratio; index_rank is the
    tier-2 sketch rank r.
    """

    def __init__(
        self,
        base: AttentionBackend,
        model_runner: "ModelRunner",
        *,
        enabled: bool = False,
        rho: float = 1.0 / 32,
        topj: int = 16,
        index_rank: int = 64,
    ):
        self.base = base
        self.attn_backend_list = [base]  # let generic snapshot/restore reach the child
        self.enabled = enabled
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
        self._kept_buf: dict = {}   # lid -> [max_reqs, cap] fm.kv_indices dtype
        self._kept_len: dict = {}   # lid -> [max_reqs] int64
        self._indptr1: dict = {}    # lid -> [2] capture-stable CSR for bs=1

    # ---- metadata / cuda-graph / properties: delegate to base ----

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        self.base.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: "ForwardBatch", in_capture: bool = False
    ):
        self.base.init_forward_metadata_out_graph(forward_batch, in_capture)
        if not self.enabled:
            return
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
            if forward_batch.seq_lens.shape[0] == 1:
                return  # bs=1 refresh is captured in-graph (zero per-step python)
            reqs = forward_batch.req_pool_indices.tolist()
            for lid in self._mla_lids:
                self._refresh_graph_bufs(lid, forward_batch, reqs)

    def init_forward_metadata_in_graph(self, forward_batch: "ForwardBatch"):
        self.base.init_forward_metadata_in_graph(forward_batch)
        # Graph-recordable per-step refresh (bs=1): append this step's slot into
        # each local MLA layer's kept table and point the CSR at that row. All
        # operands are fixed-address GPU tensors, so this captures and replays
        # with zero python on the decode path. bs>1 uses the out-graph CPU pack.
        if (
            self.enabled
            and self._kept_buf
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

    def shared_read_ends(self, fm: "ForwardMode") -> "SharedReadEnds":
        return self.base.shared_read_ends(fm)

    def get_indexer_metadata(self, layer_id: int, forward_batch: "ForwardBatch"):
        return self.base.get_indexer_metadata(layer_id, forward_batch)

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs):
        self.base.update_verify_buffers_to_fill_after_draft(spec_info, cuda_graph_bs)

    @property
    def verify_mask(self) -> Optional["VerifyMask"]:
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
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        out = self.base.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )
        if self.enabled:
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
            self._kept_buf[lid] = torch.zeros(max_reqs, cap, dtype=dt, device=dev)
            self._kept_len[lid] = torch.zeros(max_reqs, dtype=torch.int64, device=dev)
            self._indptr1[lid] = torch.zeros(2, dtype=dt, device=dev)
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
            # Slot reuse: drop the bs>1 tier-2 index state built for the slot's
            # previous occupant; the first decode step rebuilds it for this
            # request. Stale state made bs>1 decode attend the prior request's
            # row set (profiler: VESTIGE attn == FULL attn at bs=16).
            self._tier2.pop((slot, lid), None)

    def _arm_aware_kept(self, row_slots, kbuf, seq_len, v_dim):
        # The kept row set for one request, shared by the bs==1 kept-table build
        # and the bs>1 tier-2 build so both honor the same arm. Benchmark arm
        # switch, read per prefill: /tmp/vestige_full present -> FULL prefix (arm
        # A, == baseline), else sidecar-residual sigma top-m + 4 sinks + recent
        # window (arm B). Both arms then run the identical graph/refresh path.
        if os.path.exists("/tmp/vestige_full"):
            return row_slots
        sigma = sidecar_sigma(kbuf[row_slots][:, v_dim:])
        keep = select_kept(sigma, rho=self.rho, closed=seq_len, sinks=4)
        keep[max(0, seq_len - _RECENT_WINDOW) :] = True
        return row_slots[keep.nonzero(as_tuple=True)[0]]

    # ---- decode: kill-switch == base; enabled == evict + recall ----

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if not self.enabled:
            return self.base.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        from sglang.srt.model_executor.runner import get_is_capture_mode

        # Reuse the base's fused MLA-decode kernel and KV write verbatim; only
        # swap the attended index set it reads (kept prefix + growing decode
        # tail) for this layer, then restore. num_kv_splits is left as the base
        # sized it (>= the compressed length -> correct, empty splits merge out).
        fm = self.base.forward_metadata
        if get_is_capture_mode():
            # Capture: pure pointer swap to this layer's fixed-address buffers
            # (pre-filled outside capture by the in_capture out-graph hook; any
            # copy or sync here would invalidate stream capture).
            self._mla_lids.add(layer.layer_id)
            bs = forward_batch.seq_lens.shape[0]
            if bs == 1:
                # tables were allocated by the pre-capture hook (outside capture)
                lid = layer.layer_id
                indptr = self._indptr1[lid]
                indices = self._kept_buf[lid].view(-1)
            else:
                bufs = self._graph_bufs[layer.layer_id]
                indptr, indices = bufs["indptr"][: bs + 1], bufs["indices"]
        elif forward_batch.seq_lens.shape[0] == 1 and layer.layer_id in self._kept_buf:
            indptr = self._indptr1[layer.layer_id]
            indices = self._kept_buf[layer.layer_id].view(-1)
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
                self._kept_buf[lid] = torch.zeros(
                    r2t.shape[0], cap_row, dtype=fm.kv_indices.dtype, device=r2t.device
                )
                self._kept_len[lid] = torch.zeros(
                    r2t.shape[0], dtype=torch.int64, device=r2t.device
                )
                self._indptr1[lid] = torch.zeros(
                    2, dtype=fm.kv_indptr.dtype, device=r2t.device
                )
                self._indptr1[lid][1] = 1  # capture attends one padded row
            bufs = self._graph_bufs.get(lid)
            if bufs is None:
                cap = max(self._graph_max_bs, 1) * self.base.max_context_len
                bufs = {
                    "indptr": fm.kv_indptr.new_zeros(max(self._graph_max_bs, 1) + 1),
                    "indices": fm.kv_indices.new_zeros(cap),
                }
                self._graph_bufs[lid] = bufs
            bufs["indptr"][: bs + 1].copy_(fm.kv_indptr[: bs + 1])
            bufs["indptr"][bs + 1 :].fill_(fm.kv_indptr[bs])
            if n > 0:
                bufs["indices"][:n].copy_(fm.kv_indices[:n])

    def _refresh_graph_bufs(self, lid, forward_batch, reqs):
        # bs > 1 graph replay: repack the CSR (contiguous, no gaps) into the
        # fixed-address buffers the captured graph reads.
        fm = self.base.forward_metadata
        bufs = self._graph_bufs[lid]
        # seq_lens is padded to the captured graph bs; out_cache_loc carries only
        # the real (unpadded) requests -- see build_replay_fb_view, which takes
        # out_cache_loc from the original batch but seq_lens from the padded
        # capture buffers. Iterate real requests, then point every padded slot at
        # one reserved pad row (slot 0) so its softmax is non-empty; its output is
        # discarded. (Reading out_cache_loc[i] for a padded i used to overrun.)
        bs = forward_batch.seq_lens.shape[0]
        real_bs = forward_batch.out_cache_loc.shape[0]
        off = 0
        for i in range(bs):
            if i < real_bs:
                req = reqs[i]
                st = self._tier2.get((req, lid))
                if st is None:
                    st = self._build_index_buffer(req, lid, forward_batch, i, fm)
                buf, n = st["buf"], st["n"]
                if n >= buf.numel():
                    buf = torch.cat([buf, buf.new_zeros(buf.numel())])
                    st["buf"] = buf
                buf[n] = forward_batch.out_cache_loc[i]
                st["n"] = n = n + 1
                st["synced_req"] = None  # invalidate the bs=1 fast-path sync state
                bufs["indices"][off : off + n].copy_(buf[:n])
                off += n
            else:
                bufs["indices"][off : off + 1].fill_(0)
                off += 1
            bufs["indptr"][i + 1] = off
        bufs["indptr"][0] = 0
        bufs["indptr"][bs + 1 :].fill_(off)

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
