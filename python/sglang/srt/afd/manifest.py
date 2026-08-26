"""The attention manifest: what a host must know, named without any model's words.

The skeleton host asks three questions -- which layers exist, which of the two
state algebras each one speaks, and how wide its pieces are -- and nothing about
the answers is a checkpoint's private vocabulary. This module is where the
pool's answer is DERIVED and where its field names are fixed: derived from the
pool's own BUILT modules, not from its config, for the same reason
`layer_kinds` reads class names -- a config says what was requested, the module
list says what was constructed. The manifest rides the pushed configuration,
so a host adopts it with everything else and never opens a config to build.

The position scheme is part of the contract. The family swaps in (3, N)
M-RoPE positions before its layer loop, the pool's rotation is keyed to that
shape, and a host that guessed flat positions rotated every key differently
from row 1 on -- found on a live pair, by v (which carries no rope) matching
to the bit while k drifted. What a host must SEND is therefore something the
pool must SAY.
"""

from __future__ import annotations

MANIFEST_KEY = "attention_manifest"

# the two state algebras a host speaks; a manifest naming any other kind is
# refused at build, by name, rather than served through a guess
KIND_SOFTMAX_KV = "softmax_kv"
KIND_GATED_DELTA = "gated_delta"



def _first_of(module, *names):
    """The first of these attributes the module has. Two families, two spellings of one width.

    `conv_kernel_size` on one, `conv_size` on the other, and the manifest wants the number, not
    the family's word for it. Named alternatives rather than a `getattr` chain with a default so
    a checkpoint carrying neither raises here, where the reason is legible, instead of putting a
    zero-tap convolution on the wire.
    """
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(
        f"a {type(module).__name__} has none of {names}: the manifest cannot state a width it "
        f"cannot read, and a guessed one is a host built to the wrong shape."
    )


def _the_sweep_in(attn):
    """The `RadixAttention` this block sweeps its cache with.

    One family hangs it on the decoder layer itself, so the attention module IS the sweep; a
    latent-attention block holds two -- `attn_mqa` for decode and `attn_mha` for prefill -- and
    the manifest describes what the HOST does, which is the decode sweep. Preferred by name and
    then by position, because a block that grows a third would otherwise be described by whichever
    one `modules()` happened to yield first.
    """
    from sglang.srt.layers.radix_attention import RadixAttention

    # By what it EXPOSES first, not by what it is: a family that hangs the sweep on the decoder
    # layer hands this the sweep itself, and so does any stand-in built to the same shape. The
    # manifest reads five fields and this is the test for all five of them.
    if all(
        hasattr(attn, f)
        for f in ("tp_q_head_num", "head_dim", "tp_k_head_num", "v_head_dim", "scaling")
    ):
        return attn
    for name in ("attn_mqa", "attn", "attn_mha"):
        held = getattr(attn, name, None)
        if isinstance(held, RadixAttention):
            return held
    if hasattr(attn, "modules"):
        for module in attn.modules():
            if isinstance(module, RadixAttention):
                return module
    raise AttributeError(
        f"a {type(attn).__name__} was sorted as cache-keeping and holds no RadixAttention. The "
        f"two answers disagree, and the manifest is not the place to decide which is right."
    )


def manifest_of(model) -> dict:
    """The manifest, read off the pool's built modules. The pool's half."""
    from sglang.srt.afd.layer_kinds import attention_module, layer_types_of

    layers = []
    for index, kind in enumerate(layer_types_of(model)):
        layer = model.model.layers[index]
        attn = attention_module(layer)
        if kind == "full_attention":
            sweep = _the_sweep_in(attn)
            layers.append(
                {
                    "kind": KIND_SOFTMAX_KV,
                    "heads": int(sweep.tp_q_head_num),
                    "head_dim": int(sweep.head_dim),
                    "kv_heads": int(sweep.tp_k_head_num),
                    "v_head_dim": int(sweep.v_head_dim),
                    "scaling": float(sweep.scaling),
                }
            )
        else:
            layers.append(
                {
                    "kind": KIND_GATED_DELTA,
                    "k_heads": int(attn.num_k_heads),
                    "v_heads": int(attn.num_v_heads),
                    "dk": int(attn.head_k_dim),
                    "dv": int(attn.head_v_dim),
                    "conv_taps": int(_first_of(attn, "conv_kernel_size", "conv_size")),
                }
            )
    norm = model.model.norm
    lm_head = getattr(model, "lm_head", None)
    return {
        "layers": layers,
        "hidden_size": int(norm.weight.shape[0]),
        "norm_eps": float(norm.variance_epsilon),
        "vocab_size": (int(lm_head.weight.shape[0]) if lm_head is not None else None),
        # what the host must SEND as positions; see the module docstring
        "positions": ("mrope" if getattr(model, "is_mrope_enabled", False) else "flat"),
    }


def adopted_manifest() -> dict | None:
    """The manifest this host adopted, or None on a pool that pushes none."""
    from sglang.srt.afd.pushed_config import adopted_value

    return adopted_value(MANIFEST_KEY)
