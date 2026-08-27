"""What kind of layer each layer is, asked of the built model rather than of a config.

It lives here because nothing here concerns a read point. Modules on the shared side of the
arrangement -- `roles`, `linear_routing`, `remote_attention` -- need to know which layers hold a
KV cache and which hold a recurrent state, and the question is older than any arrangement built on
top of it. It used to be answered from a module that has since left this package for a derived
one; asking it there made the shared half depend on the derived one, which is the direction that
must not exist if standard AFD is to stand on its own.

The functions are unchanged from where they were.
"""

from __future__ import annotations


def full_attention_layers(layer_types: list[str]) -> tuple[int, ...]:
    """The layers whose attention is a sweep over a cache: the softmax ones.

    This answers the DEPLOYMENT question -- which layers can start their attention before the
    feed-forward that precedes them, because a sweep over cached keys needs only the query. A
    linear-attention layer has no cache to sweep; its state update is sequential.

    It is NOT the answer to "which layers' queries move". Those are two questions and they were
    one function here for a while, which silently capped coverage at the softmax layers.
    """
    return tuple(i for i, t in enumerate(layer_types) if t == "full_attention")


def is_full_attention(layer) -> bool:
    """Whether this layer sweeps a KV cache, decided in the one place that decides it.

    Asking `hasattr(layer, "attn")` at each call site is the same question answered separately in
    each module, and the day a layer grows an `attn` attribute for another reason they answer it
    differently. This routes through the same class-name rule `layer_types_of` uses, so a stack
    whose layers are neither kind is refused once rather than silently sorted twice.
    """
    return _kind_of(layer) == "full_attention"


# Where a decoder layer hangs the thing that does its attention. Three spellings, because
# `Qwen3.5` splits the two kinds into `linear_attn` and `attn` while `kimi_linear` gives both
# kinds one attribute and one decoder class. Ordered so the split families answer on their own
# name and only a single-attribute family falls through to the last.
_ATTENTION_ATTRS = ("linear_attn", "attn", "self_attn")


def attention_module(layer):
    """The module that does this layer's attention, whatever its family calls it.

    Resolved once here so that the twenty-odd places which reach for `layer.linear_attn` or
    `layer.attn` stop each deciding the question separately. A family that hangs both kinds on
    one attribute -- `kimi_linear` does -- answers the same way as one that splits them.
    """
    for name in _ATTENTION_ATTRS:
        module = getattr(layer, name, None)
        if module is not None:
            return module
    return None


def _kind_of(layer) -> str:
    """Which of the two kinds this layer is, asked of the layer and then of what it holds.

    The class name is tried first because it is what the skeleton host builds to
    (`afd/skeleton.py` says the name is load-bearing) and what the split families already say.
    A family with ONE decoder class for both kinds -- `KimiDecoderLayer`, which decides by what
    it hangs on `self_attn` -- says nothing in its own name, so the question goes to the built
    attention module. It is still the built module being asked, never a config.

    A block is recognised by the TYPE it holds, not by what it is called and not by an attribute
    name. Neither `KimiDeltaAttention` nor `DeepseekV2AttentionMLA` has "Linear" in its name, so
    a rule on names would sort the delta block into the cache-keeping half. An attribute name is
    no better: this rule first looked for `q_conv1d`, which is what the checkpoint's own modelling
    file calls the convolution and is not what sglang's implementation of the same block calls it
    (`qkv_conv1d`, and the weights then live inside the kernel wrapper).

    What does hold still is the pair of types sglang gives the two kinds of state:
    `RadixLinearAttention` for a recurrence and `RadixAttention` for a cache. They are siblings,
    not parent and child, so holding one says nothing about the other -- which is what makes them
    a discriminator rather than a guess.
    """
    name = type(layer).__name__
    if "Linear" in name:
        return "linear_attention"
    if "Attention" in name:
        return "full_attention"

    attn = attention_module(layer)
    if attn is not None:
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.layers.radix_linear_attention import RadixLinearAttention

        held = set(type(m) for m in attn.modules())
        # The linear one first. The two are siblings rather than parent and child -- neither is
        # an instance of the other -- so the order is not a correctness fix; it is that a block
        # holding both would be one whose STATE is recurrent, and the state is the question.
        if any(issubclass(t, RadixLinearAttention) for t in held):
            return "linear_attention"
        if any(issubclass(t, RadixAttention) for t in held):
            return "full_attention"

    held = type(attn).__name__ if attn is not None else "nothing this knows to look at"
    raise RuntimeError(
        f"a {name} holding {held} is neither of the two kinds this arm knows how to treat. "
        f"Name it before converting it, or give it a mark this can read: a layer handled by "
        f"accident is a layer nothing has checked."
    )


def layer_types_of(model) -> list[str]:
    """Read the layer kinds off the LOADED stack, not off a config field.

    The checkpoint's config.json carries `layer_types`, and sglang's own config class does not:
    it derives the pattern from `full_attention_interval`. Asking the built model which class each
    layer is answers the question both ways round and cannot disagree with what is actually there
    -- a config field says what was requested, the module list says what was constructed.
    """
    if not hasattr(model, "model"):
        raise TypeError(
            f"layer_types_of takes the loaded model, not a config: got "
            f"{type(model).__name__}. The whole point of this function is that it asks the built "
            f"module list rather than a config field, because a config says what was requested "
            f"and the module list says what was constructed."
        )
    kinds = []
    for index, layer in enumerate(model.model.layers):
        try:
            kinds.append(_kind_of(layer))
        except RuntimeError as e:
            raise RuntimeError(f"layer {index}: {e}") from e
    return kinds


def convertible_layers(layer_types: list[str]) -> tuple[int, ...]:
    """The layers whose query read point moves: every layer that has a query.

    A linear-attention layer has one too. Its query is the first slice of a fused projection and
    it meets a recurrent state rather than a cache, but the rewiring -- read the query from an
    earlier point of the residual stream -- is the same rewiring, and the quality it costs is a
    property of the whole stack rather than of the softmax layers alone.

    The study measured both coverages on this model: 16 of 64 layers costs +0.0181 bits per byte
    and 63 of 64 costs +0.0211, a factor of 1.17 for four times the coverage. Reporting the
    cheaper coverage's cost under the fuller coverage's name is the error this function exists to
    make impossible.
    """
    return tuple(range(len(layer_types)))


def stateful_layers(layer_types: list[str]) -> tuple[int, ...]:
    """The layers that hold per-request state and therefore cannot leave the host.

    A softmax layer holds a KV cache; a linear-attention layer holds a recurrent state. Both are
    the request's, and a pool that held either would stop being stateless -- it could no longer be
    released between a request's own calls, which is the property the arrangement is built on.
    Only the feed-forward, which reads the same weights whatever the caller's history, may leave.
    """
    return tuple(range(len(layer_types)))


def linear_widths(config) -> dict:
    """The recurrent block's widths, from whichever vocabulary this family speaks.

    Two of them. Qwen3.5 puts flat fields on the config -- `linear_num_key_heads` and its four
    companions -- and `kimi_linear` does not: its widths live inside `linear_attn_config` and are
    exposed through `mamba2_cache_params`, the normalised accessor sglang already builds every
    hybrid state pool from. Both are read here so that no caller has to know which it is holding.

    The flat fields are tried FIRST and returned unchanged where they exist. That ordering is not
    a preference: it means a family that has them keeps the numbers it has always been given, so
    this cannot move a deployment that was working. The derived path runs only where the direct
    one raises.

    Returns the manifest's own field names, which is what the callers want to put on the wire.
    """
    flat = (
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    )
    if all(hasattr(config, name) for name in flat):
        return {
            "k_heads": config.linear_num_key_heads,
            "v_heads": config.linear_num_value_heads,
            "dk": config.linear_key_head_dim,
            "dv": config.linear_value_head_dim,
            "conv_taps": config.linear_conv_kernel_dim,
        }

    params = getattr(config, "mamba2_cache_params", None)
    shape = getattr(params, "shape", None)
    if shape is None:
        raise AttributeError(
            f"a {type(config).__name__} states its recurrent widths neither as "
            f"{flat[0]} and its companions nor through `mamba2_cache_params`. The arrangement "
            f"cannot size a state it cannot measure, and a guessed width is a host built to the "
            f"wrong shape."
        )
    # `num_heads`/`head_dim` are the VALUE side on both shapes; a shape that separates the key
    # side says so, and one that does not is telling us the two are the same.
    return {
        "k_heads": getattr(shape, "num_k_heads", shape.num_heads),
        "v_heads": shape.num_heads,
        "dk": getattr(shape, "head_k_dim", shape.head_dim),
        "dv": shape.head_dim,
        "conv_taps": shape.conv_kernel,
    }
