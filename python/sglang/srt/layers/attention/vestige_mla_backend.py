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

from typing import TYPE_CHECKING, Optional

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
    from sglang.srt.layers.attention.verify_mask import VerifyMask
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner


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

    # ---- metadata / cuda-graph / properties: delegate to base ----

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        self.base.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: "ForwardBatch", in_capture: bool = False
    ):
        self.base.init_forward_metadata_out_graph(forward_batch, in_capture)

    def init_forward_metadata_in_graph(self, forward_batch: "ForwardBatch"):
        self.base.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.base.init_cuda_graph_state(max_bs, max_num_tokens)

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

    # ---- prefill: delegate, then build the tier-2 index for this request ----

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
            self._build_tier2(layer, forward_batch)
        return out

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
        return self._compressed_decode(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    # ---- VestigeKV core (GPU seam; port RecallTier/VestigePolicy) ----

    def _build_tier2(self, layer: "RadixAttention", forward_batch: "ForwardBatch"):
        # TODO(vestigekv): at prefill end, from the latent rows just written to
        # MLATokenToKVPool (get_key_buffer(layer.layer_id) indexed by
        # req_to_token[req_pool_index, :seq_len]), compute the sidecar-residual
        # eviction set (sigma over the 64-dim decoupled branch) and the tier-2
        # recall index (exact 64-dim summand + rank-index_rank sketch + per-row
        # certificate, self-calibrated z + entropy gate). Store under
        # self._tier2[req_pool_index]. Port from mini-sglang RecallTier.build.
        raise NotImplementedError("VestigeKV tier-2 build seam pending; see VESTIGEKV_PORT.md")

    def _compressed_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        # TODO(vestigekv): restrict the attended latent set to kept rows + a
        # topj-capped recalled set for this step, then call self.base.forward_decode.
        # Surgical point (per interface map): the paged kv_indices the base kernel
        # gathers from req_to_token[req_pool_indices, :seq_len]; drop evicted slots
        # and splice recalled slots, optionally staging recalled rows with
        # token_to_kv_pool.get_mla_kv_buffer(layer, recall_loc). Port the query
        # scoring + fetch from mini-sglang RecallTier.query / VestigePolicy.
        raise NotImplementedError("VestigeKV decode seam pending; see VESTIGEKV_PORT.md")
