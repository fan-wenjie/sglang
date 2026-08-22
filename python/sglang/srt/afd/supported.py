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

from sglang.srt.afd.layer_kinds import is_full_attention

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



def _spans(indices: list[int]) -> str:
    """"0-15" rather than sixteen numbers; consecutive runs collapsed."""
    runs, start = [], indices[0]
    for a, b in zip(indices, indices[1:] + [None]):
        if b != a + 1:
            runs.append(f"{start}" if start == a else f"{start}-{a}")
            start = b
    return ", ".join(runs)


def _kind_independent(layer) -> list[str]:
    """What every layer needs, whatever kind it turned out to be.

    hasattr is the right tool here for the reason it usually is not: absence is the answer this
    function returns, not an error it is swallowing. The names belong to a model file this one does
    not own, so there is no construction site at which they could be set to None instead.
    """
    gaps = []
    if not hasattr(layer, "input_layernorm"):
        gaps.append("no input_layernorm, which is the norm the early stream has to pass through "
                    "to be the layer's own input")
    communicator = getattr(layer, "layer_communicator", None)
    if communicator is None or not hasattr(communicator, "prepare_mlp"):
        gaps.append("no layer_communicator.prepare_mlp, which is where h_l is read from")
    return gaps


def _softmax_side(layer) -> list[str]:
    gaps = []
    attn = getattr(layer, "attn", None)
    if attn is None:
        gaps.append("classified as softmax attention and has no attn to sweep with")
    else:
        absent = [f for f in ATTENTION_GEOMETRY if not hasattr(attn, f)]
        if absent:
            gaps.append(f"attn lacks {', '.join(absent)}, which the partition needs to shape its "
                        f"two halves")
    absent = [m for m in PREPARE_METHODS if not hasattr(layer, m)]
    if absent:
        gaps.append(f"no {', '.join(absent)}. The query is projected by whichever variant the "
                    f"layer's own dispatch picks, and a missing one means that dispatch is not "
                    f"the one mirrored here")
    return gaps


def _linear_side(layer) -> list[str]:
    linear = getattr(layer, "linear_attn", None)
    if linear is None or not hasattr(linear, "_forward_input_proj"):
        return ["no linear_attn._forward_input_proj, which is where the query slice is spliced. "
                "coverage=softmax does not need it"]
    return []


def _probe(layer, coverage: str) -> tuple[str | None, list[str]]:
    """One layer's gaps, and its kind if it has one.

    A layer this arm cannot classify used to end the probe there, so a new family saw only the
    first gate of several and learned the rest one launch at a time -- which is the discovery
    process this module exists to replace. An unclassified layer is therefore probed against BOTH
    sides, and the report says which side's names it does have. That is what tells a porter whether
    they are mapping a softmax layer or a linear one.
    """
    gaps = _kind_independent(layer)
    try:
        softmax = is_full_attention(layer)
    except RuntimeError as e:
        both = {"softmax": _softmax_side(layer), "linear": _linear_side(layer)}
        has = [side for side, missing in both.items() if not missing]
        gaps.append(
            f"{e} Probed against both kinds anyway: it satisfies the "
            + (f"{' and '.join(has)} names" if has
               else "names of neither side, so the port is both a classification and a mapping")
        )
        gaps.extend(f"if it is meant to be softmax: {g}" for g in both["softmax"])
        if coverage == "all":
            gaps.extend(f"if it is meant to be linear: {g}" for g in both["linear"])
        return None, gaps

    gaps.extend(_softmax_side(layer) if softmax
                else (_linear_side(layer) if coverage == "all" else []))
    return ("full_attention" if softmax else "linear_attention"), gaps


def check_supported(model, *, coverage: str = "all") -> dict:
    """Verify the structural contract. Returns a summary, or raises listing every gap.

    `coverage="softmax"` skips the linear-attention requirements, because that arm never touches
    those layers and a stack whose linear layers are shaped differently can still serve it.

    Findings are grouped by message rather than listed per layer. A 64-layer stack with one problem
    used to print that problem 64 times and then truncate at twelve, so a second and third problem
    were hidden behind the first one's repetitions -- a check promising every missing piece at once
    that delivered the same piece twelve times.
    """
    layers = _layers_of(model)
    kinds = {"full_attention": 0, "linear_attention": 0}
    found: dict[tuple[str, str], list[int]] = {}

    for index, layer in enumerate(layers):
        kind, gaps = _probe(layer, coverage)
        if kind is not None:
            kinds[kind] += 1
        for gap in gaps:
            found.setdefault((type(layer).__name__, gap), []).append(index)

    if found:
        listed = "\n  ".join(
            f"layer(s) {_spans(indices)} ({name}): {gap}"
            for (name, gap), indices in found.items()
        )
        raise ModelNotSupported(
            f"this stack does not provide what the Early-Q wiring wraps:\n  {listed}"
            f"\n\nAll of it is structural: the wiring wraps methods on the model file rather "
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
