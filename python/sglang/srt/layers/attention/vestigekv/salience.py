"""Salience key for rope-less MLA models under VestigeKV.

GLM-5.3-Flash stores no un-roped branch in its latent row (qk_rope_head_dim
== 0), so tier-1 needs another query-independent per-token channel. The DSA
indexer's key, k_norm(wk(x)), is never rotated when rope is absent and was
trained to rank tokens for retrieval; once the indexer's top-k is switched off
(index_topk=None) this module keeps that key alive and the backend holds it in
a per-layer side pool, row-aligned with the latent KV pool.
"""

from __future__ import annotations

import torch
from torch import nn

from sglang.srt.layers.layernorm import LayerNorm, RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.utils.common import add_prefix


class SalienceKey(nn.Module):
    """k_norm(wk(x)), exactly IndexerKPool._get_k_bf16 with rope skipped."""

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        k_norm_type: str,
        prefix: str,
    ):
        super().__init__()
        self.wk = ReplicatedLinear(
            hidden_size, head_dim, bias=False, prefix=add_prefix("wk", prefix)
        )
        # Same norm the indexer builds for this config (fp32 LayerNorm by default).
        self.k_norm = (
            RMSNorm(head_dim)
            if k_norm_type == "rms"
            else LayerNorm(head_dim, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key, _ = self.wk(x)
        return self.k_norm(key)


FP8_MAX = 448.0  # e4m3 range act_quant clamps to


def quantize_salience(key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The DSA index-cache format: fp8 e4m3 with one ue8m0 (power-of-two) fp32
    scale per token, i.e. act_quant(key, block_size=head_dim, scale_fmt="ue8m0").
    key ~= q.float() * scale[:, None]."""
    key = key.float().contiguous()
    if key.is_cuda:
        from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant

        q, scale = act_quant(key, block_size=key.shape[-1], scale_fmt="ue8m0")
        return q, scale.squeeze(-1)
    # Same arithmetic as _act_quant_kernel with round_scale, for non-CUDA callers.
    amax = key.abs().amax(dim=-1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    q = (key / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def dequantize_salience(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale[:, None]


def vestigekv_backend_of(backend):
    """The VestigeKV backend serving full-attention layers, or None.

    Hybrid models hand the model a HybridLinearAttnBackend whose full-attention
    half is the one VestigeKV wraps.
    """
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.vestigekv_mla_backend import VestigeKVMLABackend

    if isinstance(backend, HybridLinearAttnBackend):
        backend = backend.full_attn_backend
    return backend if isinstance(backend, VestigeKVMLABackend) else None
