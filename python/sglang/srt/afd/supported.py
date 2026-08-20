"""What a model has to expose for this wiring to work, checked once and stated in full.

This installs by wrapping methods the model file happens to have. On the stack it was written
against that is fine; on the next one it is not, and the failure it would otherwise produce is the
bad kind -- an AttributeError from three calls inside a wrapper, during a forward pass, naming one
attribute out of a contract nobody wrote down. Somebody bringing a new model would then discover
the requirements one exception at a time.

So the contract is checked at install and reported whole:

    the stack          model.model.layers, non-empty
    every layer        classifies as softmax or linear attention, and carries input_layernorm and
                       layer_communicator.prepare_mlp
    softmax layers     attn, and the four forward_prepare_* variants, and a geometry the split can
                       read: query heads, key-value heads, head dimensions, the softmax scale
    linear layers      linear_attn._forward_input_proj

One message, every missing piece, and the model's own class names. A check that reported the first
problem would be the same discovery process with extra steps.

## Why this cannot be an isinstance check

The pieces are spread over a model file that is not this one's to change, and the same shapes
appear under different class names in every family. What is actually required is structural, so
that is what is verified -- and the check is run against a stack that has already been BUILT, so
it describes what is there rather than what a config asked for.
"""

from __future__ import annotations

import logging

from sglang.srt.afd.read_point import is_full_attention

logger = logging.getLogger(__name__)

PREPARE_METHODS = (
    "forward_prepare_cuda_fused",
    "forward_prepare_fused_gate",
    "forward_prepare_native",
    "forward_prepare_npu",
)
ATTENTION_GEOMETRY = (
    "tp_q_head_num",
    "tp_k_head_num",
    "qk_head_dim",
    "v_head_dim",
    "scaling",
)


class ModelNotSupported(RuntimeError):
    """Raised at install, listing everything the stack does not provide."""


def _layers_of(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) == 0:
        raise ModelNotSupported(
            f"a {type(model).__name__} exposes no model.model.layers to convert. This wiring "
            f"wraps methods on a built decoder stack; there is nothing here to wrap, and "
            f"installing on it would convert nothing while reporting a conversion."
        )
    return layers


def check_supported(model, *, coverage: str = "all") -> dict:
    """Verify the structural contract. Returns a summary, or raises listing every gap.

    `coverage="softmax"` skips the linear-attention requirements, because that arm never touches
    those layers and a stack whose linear layers are shaped differently can still serve it.
    """
    layers = _layers_of(model)
    missing: list[str] = []
    kinds = {"full_attention": 0, "linear_attention": 0}

    for index, layer in enumerate(layers):
        name = type(layer).__name__
        try:
            softmax = is_full_attention(layer)
        except RuntimeError as e:
            missing.append(f"layer {index} ({name}): {e}")
            continue
        kinds["full_attention" if softmax else "linear_attention"] += 1

        if not hasattr(layer, "input_layernorm"):
            missing.append(f"layer {index} ({name}): no input_layernorm, which is the norm the "
                           f"early stream has to pass through to be the layer's own input")
        communicator = getattr(layer, "layer_communicator", None)
        if communicator is None or not hasattr(communicator, "prepare_mlp"):
            missing.append(f"layer {index} ({name}): no layer_communicator.prepare_mlp, which is "
                           f"where h_l is read from")

        if softmax:
            attn = getattr(layer, "attn", None)
            if attn is None:
                missing.append(f"layer {index} ({name}): classified as softmax attention and has "
                               f"no attn to sweep with")
            else:
                absent = [f for f in ATTENTION_GEOMETRY if not hasattr(attn, f)]
                if absent:
                    missing.append(f"layer {index} ({name}): attn lacks {', '.join(absent)}, "
                                   f"which the partition needs to shape its two halves")
            absent = [m for m in PREPARE_METHODS if not hasattr(layer, m)]
            if absent:
                missing.append(f"layer {index} ({name}): no {', '.join(absent)}. The query is "
                               f"projected by whichever variant the layer's own dispatch picks, "
                               f"and a missing one means that dispatch is not the one mirrored "
                               f"here")
        elif coverage == "all":
            linear = getattr(layer, "linear_attn", None)
            if linear is None or not hasattr(linear, "_forward_input_proj"):
                missing.append(f"layer {index} ({name}): no linear_attn._forward_input_proj, "
                               f"which is where the query slice is spliced. coverage=softmax "
                               f"does not need it")

    if missing:
        shown = missing[:12]
        more = (f"\n  ... and {len(missing) - len(shown)} more"
                if len(missing) > len(shown) else "")
        raise ModelNotSupported(
            f"this stack does not provide what the Early-Q wiring wraps:\n  "
            + "\n  ".join(shown) + more
            + f"\n\nAll of it is structural: the wiring wraps methods on the model file rather "
              f"than forking it, so a family that spells these differently needs those names "
              f"mapped, not this check relaxed. Installing anyway would convert some layers and "
              f"not others, and report the deeper shift's cost under the shallower one's name."
        )

    summary = {"layers": len(layers), **kinds}
    logger.info(
        "afd: stack checked -- %s layer(s), %s softmax and %s linear attention, all providing "
        "what the wiring wraps",
        summary["layers"], summary["full_attention"], summary["linear_attention"],
    )
    return summary
