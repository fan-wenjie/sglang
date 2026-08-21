"""Where a decoder layer keeps the three things this wiring needs, family by family.

The wiring wraps names it does not own. Written against one model file, it hardcoded where that
file happens to put them, and `python -m sglang.srt.afd.portability` counts the result: of 216
model files in this checkout, one has all four `forward_prepare_*` variants and 32 have a
`layer_communicator` at all. The contract was not too strict; it was stated at the wrong
granularity.

Three things are needed, and every family has all three somewhere:

    the attention module     what the sweep and the join are issued against
    the query projection     the variants that turn a hidden state into q, k and v
    h_l                      the residual AFTER attention and BEFORE the feed-forward

Two families, the same three things:

                          Qwen3.5                          Llama
    attention module      layer.attn                       layer.self_attn.attn
    query projection      layer.forward_prepare_* (4)      layer.self_attn.forward_prepare_* (2)
    h_l                   layer_communicator.prepare_mlp   layer.post_attention_layernorm
                          returns it as `residual`         returns it as the second output

The `post_attention_layernorm` route is not a workaround. That module is a FUSED add-and-normalise:
handed the attention output and the pre-attention residual, it returns the normalised value and
their sum, and their sum is the definition of h_l. sglang's own HuggingFace installer in
`wiring.py` already reads h_l there, so this makes one route out of what were two.

## Why classification is structural here and nominal in read_point

`read_point._kind_of` sorts a layer by whether its class name contains "Linear" or "Attention".
That works on a file whose two layer classes are named for their kinds and silently fails on one
whose are not -- `LlamaDecoderLayer` contains neither word, so a stack of them classified as
neither kind and the check reported sixteen copies of that before reporting anything else.

A name is evidence about a layer; a `linear_attn` attribute IS the layer being a linear-attention
layer. So the kind is decided here by what the layer holds, and the name is used only to describe
what was found. A family that renames its classes keeps working; a family that renames its
submodules gets a message naming the submodules that were looked for.

## What this deliberately does not do

It does not search. Each lookup is a short, ordered list of names that a family is known to use,
and an unknown family raises with that list in the message. A resolver that recursed until it
found something shaped right would eventually find something shaped right on a layer that means
something else, and the failure would be a converted layer that nothing checked -- the exact
outcome `supported.py` exists to prevent.
"""

from __future__ import annotations

from typing import NamedTuple

# Where a family may keep the attention submodule. `None` means the layer itself, which is where
# Qwen3.5 keeps `attn`; the named ones are the wrappers other families put around it.
ATTENTION_HOLDERS = (None, "self_attn", "attention", "attn_block")

# The variants a layer may dispatch its query/key/value projection through. A family need not have
# all of them -- Llama has two, Qwen3.5 has four -- because only the one its own dispatch picks is
# ever called. Requiring all four was a statement about one model file, not about the contract.
PREPARE_METHODS = (
    "forward_prepare_cuda_fused",
    "forward_prepare_fused_gate",
    "forward_prepare_native",
    "forward_prepare_npu",
)

# What the split reads off the attention module to shape its two halves.
ATTENTION_GEOMETRY = ("tp_q_head_num", "tp_k_head_num", "qk_head_dim", "v_head_dim", "scaling")

FULL, LINEAR = "full_attention", "linear_attention"


class HSource(NamedTuple):
    """Where h_l can be captured, and which element of that call's result carries it.

    `stage` is the name the stash records. `EarlyQStash` refuses a write from a stage it does not
    expect, so the two routes to h_l have to be distinguishable in the record -- a stash that
    accepted any stage would accept the layer's output, which is x_(l+1), and that substitution
    produces a correct model with no overlap and nothing to notice it by.
    """

    stage: str
    owner_name: str      # attribute on the layer holding the callable, "" when it is the layer
    attr: str            # the callable's own name
    index: int           # which element of the returned tuple is h_l


class LayerPorts(NamedTuple):
    """One layer's three things, resolved."""

    kind: str
    holder_name: str            # "" when the attention module hangs off the layer itself
    attn_name: str              # attribute on the holder that is the attention module
    prepare: tuple[str, ...]    # the projection variants this family actually has
    h: HSource

    @property
    def is_full(self) -> bool:
        return self.kind == FULL


def _has(obj, name: str) -> bool:
    """Presence of a name this module does not own.

    hasattr is the right tool for the reason it usually is not: absence is the answer being
    computed, not an error being swallowed. These attributes live on model files this code cannot
    change, so there is no construction site at which they could be set to None instead.
    """
    return obj is not None and hasattr(obj, name)


def _holder_of(layer):
    """The module carrying the attention, and the attribute path that found it."""
    for holder_name in ATTENTION_HOLDERS:
        holder = layer if holder_name is None else getattr(layer, holder_name, None)
        if holder is None:
            continue
        for attn_name in ("attn",):
            if _has(holder, attn_name):
                return holder, ("" if holder_name is None else holder_name), attn_name
    return None, "", ""


def _h_source_of(layer) -> HSource | None:
    """The two known routes to h_l, in the order a family is asked for them.

    `prepare_mlp` first: where a family has a layer communicator, that communicator is what walks
    the residual stream, and reading h_l anywhere else would read a value the communicator has not
    finished with. `post_attention_layernorm` is the general route -- a fused add-and-normalise
    returns h_l as its second output -- and it is what a family without a communicator uses.
    """
    communicator = getattr(layer, "layer_communicator", None)
    if _has(communicator, "prepare_mlp"):
        return HSource(stage="post_attention_pre_mlp", owner_name="layer_communicator",
                       attr="prepare_mlp", index=1)
    if _has(layer, "post_attention_layernorm"):
        return HSource(stage="post_attention_pre_mlp", owner_name="",
                       attr="post_attention_layernorm", index=1)
    return None


def gaps(layer, *, coverage: str = "all") -> list[str]:
    """Everything this layer does not provide, as messages, without raising.

    Returning a list rather than raising is what lets one caller report every layer's every gap in
    one message. A resolver that raised on the first problem would hand back the first problem,
    and a family would learn its port one launch at a time.
    """
    found: list[str] = []
    linear = getattr(layer, "linear_attn", None)
    holder, holder_name, attn_name = _holder_of(layer)

    if linear is None and holder is None:
        found.append(
            f"holds neither a linear_attn nor an attention module. Looked for attn on the layer "
            f"and on {', '.join(n for n in ATTENTION_HOLDERS if n)}. One of those is what the "
            f"sweep is issued against, and a layer with neither is not a layer this converts"
        )
    if linear is not None and holder is not None:
        found.append(
            "holds both linear_attn and an attention module, so which kind it is depends on which "
            "attribute is read first. Name the kind rather than letting the order decide it"
        )

    if holder is not None:
        attn = getattr(holder, attn_name)
        absent = [f for f in ATTENTION_GEOMETRY if not _has(attn, f)]
        if absent:
            found.append(f"its attention module lacks {', '.join(absent)}, which the partition "
                         f"needs to shape its two halves")
        if not [m for m in PREPARE_METHODS if _has(holder, m)]:
            where = holder_name or "the layer"
            found.append(
                f"none of {', '.join(PREPARE_METHODS)} on {where}. One of them is the query's own "
                f"projection, and with none of them there is nothing to run on the early stream"
            )
    elif linear is not None and coverage == "all" and not _has(linear, "_forward_input_proj"):
        found.append("its linear_attn has no _forward_input_proj, which is where the query slice "
                     "is spliced. coverage=softmax does not need it")

    if not _has(layer, "input_layernorm"):
        found.append("no input_layernorm, which is the norm the early stream has to pass through "
                     "to be the layer's own input")
    if _h_source_of(layer) is None:
        found.append(
            "no layer_communicator.prepare_mlp and no post_attention_layernorm, so there is "
            "nowhere to read h_l. h_l is the residual after attention and before the feed-forward; "
            "a family that forms it inline needs the module that produces it named here"
        )
    return found


def resolve(layer, *, coverage: str = "all") -> LayerPorts:
    """This layer's three things, or a RuntimeError listing what was looked for.

    Callers that report many layers at once should ask `gaps` first; this raises on the first
    layer that cannot be resolved, which is right at the point of use and wrong for a report.
    """
    problems = gaps(layer, coverage=coverage)
    if problems:
        raise RuntimeError(f"a {type(layer).__name__}: " + "; ".join(problems))

    linear = getattr(layer, "linear_attn", None)
    holder, holder_name, attn_name = _holder_of(layer)
    return LayerPorts(
        kind=FULL if holder is not None else LINEAR,
        holder_name=holder_name,
        attn_name=attn_name,
        prepare=tuple(m for m in PREPARE_METHODS if holder is not None and _has(holder, m)),
        h=_h_source_of(layer),
    )


def holder_of(layer, ports: LayerPorts):
    """The module the prepare variants and the attention module hang off."""
    return layer if not ports.holder_name else getattr(layer, ports.holder_name)


def h_owner_of(layer, ports: LayerPorts):
    """The module whose call returns h_l."""
    return layer if not ports.h.owner_name else getattr(layer, ports.h.owner_name)
