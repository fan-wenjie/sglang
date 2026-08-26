"""The host as an attention service: layers from the manifest, never a model.

A span-routed host executes almost none of the family's code -- the routing
replaces every layer's forward, the pool holds every projection, and what this
side actually runs is the attention core, a convolution ring, and one final
norm. What the family's model class still contributed was its module TREE: a
place for `layer.attn` to hang, a conv1d whose storage the pool's push fills,
and the numbers to size them. This module builds that tree from the pool's
ATTENTION MANIFEST alone -- which layers exist, which of the two state
algebras each speaks, and how wide its pieces are -- so a host neither knows
nor cares which model it serves, and none of a checkpoint's private field
names appear here. The pool says `host_model: skeleton` in the pushed
configuration, and the loader builds this instead of resolving a family class.

The tokenizer is untouched by any of this: it comes through the papers exactly
as before, because text in and text out is the part of serving that IS the
host's own business.
"""

from __future__ import annotations

import logging
import types

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def skeleton_wanted() -> bool:
    """Whether the pool told this host to build the skeleton. Safe anywhere.

    Reads the adopted configuration, adopting first if this process has not yet
    -- class resolution happens before the build context that usually adopts.
    Any failure to answer is a "no": a process without a published runtime
    (a unit test, a tool) is not an AFD host mid-boot.
    """
    try:
        from sglang.srt.afd.pushed_config import adopt_from_the_pool, adopted_value
        from sglang.srt.runtime_context import get_disagg

        if get_disagg().afd_mode != "host":
            return False
        adopt_from_the_pool()
        return adopted_value("host_model") == "skeleton"
    except Exception:  # noqa: BLE001 -- no runtime, no adoption: not a skeleton host
        return False


class AttentionServiceLinearLayer(nn.Module):
    """A linear-attention stop: the convolution's storage, and nothing else.

    The class NAME is load-bearing: `layer_kinds` reads the kind off it.
    """

    def __init__(self, spec: dict, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        conv_dim = 2 * spec["k_heads"] * spec["dk"] + spec["v_heads"] * spec["dv"]
        conv = nn.Module()
        conv.weight = nn.Parameter(
            torch.empty(conv_dim, 1, spec["conv_taps"]), requires_grad=False
        )
        conv.bias = None
        self.linear_attn = nn.Module()
        self.linear_attn.conv1d = conv

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            f"skeleton linear layer {self.layer_id} ran its own forward: the span "
            f"routing installs a pass-through over it, so a call reaching here "
            f"means the install did not."
        )


class AttentionServiceLayer(nn.Module):
    """A standard-attention stop: the cache-facing attention core, and nothing else."""

    def __init__(self, spec: dict, layer_id: int, prefix: str = ""):
        super().__init__()
        from sglang.srt.layers.radix_attention import RadixAttention

        self.layer_id = layer_id
        self.attn = RadixAttention(
            spec["heads"],
            spec["head_dim"],
            spec["scaling"],
            num_kv_heads=spec["kv_heads"],
            layer_id=layer_id,
            v_head_dim=spec.get("v_head_dim", -1),
            prefix=f"{prefix}.attn",
            quant_config=None,
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            f"skeleton attention layer {self.layer_id} ran its own forward: the "
            f"span routing installs the head over it, so a call reaching here "
            f"means the install did not."
        )


class _NoEmbedding(nn.Module):
    """Stands where an embedding table would: the ids ride to the pool instead.

    The zero-row meta weight is what the install-time embedding check reads --
    a skeleton host is absent-embedding by construction, in the same sense the
    family's released table is.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(0, 0, device="meta"), requires_grad=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # the install wraps this forward to record the ids and hand back the
        # NaN placeholder; reaching the original body means it did not
        raise RuntimeError(
            "a skeleton host holds no embedding table; the token ids ride the "
            "span ENTER and the pool looks them up -- and the routing's install "
            "did not wrap this module, so nothing is recording them."
        )


class AttentionServiceStack(nn.Module):
    def __init__(self, manifest: dict):
        super().__init__()
        from sglang.srt.afd.manifest import KIND_GATED_DELTA, KIND_SOFTMAX_KV
        from sglang.srt.layers.layernorm import GemmaRMSNorm

        self.embed_tokens = _NoEmbedding()
        built = []
        for index, spec in enumerate(manifest["layers"]):
            if spec["kind"] == KIND_SOFTMAX_KV:
                built.append(
                    AttentionServiceLayer(spec, index, prefix=f"model.layers.{index}")
                )
            elif spec["kind"] == KIND_GATED_DELTA:
                built.append(AttentionServiceLinearLayer(spec, index))
            else:
                raise RuntimeError(
                    f"layer {index} speaks {spec['kind']!r}, which is no state "
                    f"algebra this host knows. A layer served by guess would be "
                    f"fluent and wrong; teach the kind before serving it."
                )
        self.layers = nn.ModuleList(built)
        self.norm = GemmaRMSNorm(manifest["hidden_size"], eps=manifest["norm_eps"])

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **_):
        # THROUGH the embedding stub: the install wrapped it to record the ids
        # (the span ENTER sends them) and hand back the NaN placeholder
        hidden_states = (
            input_embeds if input_embeds is not None else self.embed_tokens(input_ids)
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )
        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class AttentionServiceForCausalLM(nn.Module):
    """What the loader builds when the pool says `host_model: skeleton`."""

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        from sglang.srt.afd.manifest import adopted_manifest
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        manifest = adopted_manifest()
        if not manifest:
            raise RuntimeError(
                "the pool said `host_model: skeleton` and pushed no attention "
                "manifest. A skeleton built from a config would need the "
                "checkpoint's own field names, which is the coupling this host "
                "exists to not have; align the pool's code."
            )
        self.manifest = manifest
        # everything the installer and the head shim read is served from the
        # manifest, through this plain record -- no hf config is consulted
        self.config = types.SimpleNamespace(
            vocab_size=manifest["vocab_size"],
            hidden_size=manifest["hidden_size"],
        )
        self.model = AttentionServiceStack(manifest)
        # the family swaps in the (3, N) M-RoPE positions before the layer loop;
        # the pool's rotation is keyed to that shape, so the position scheme is
        # part of the manifest (v carries no rope, which is how a guess here was
        # caught: k drifted from row 1 on, v matched to the bit)
        self.is_mrope_enabled = manifest.get("positions") == "mrope"
        # meta by the absent-class build context: the pool computes the logits
        self.lm_head = ParallelLMHead(manifest["vocab_size"], manifest["hidden_size"])
        self.logits_processor = LogitsProcessor(self.config)
        logger.info(
            "afd host: attention service built from the manifest -- %d layer(s), "
            "no family class, no embedding, no head",
            len(self.model.layers),
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, **kwargs):
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions
        hidden_states = self.model(input_ids, positions, forward_batch, **kwargs)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights):
        # a skeleton has no checkpoint weights; the pool pushes the residual few
        for _ in weights:
            pass
