"""Install the Early-Q read point on a loaded Qwen3.5 stack, without forking the model file.

Three hooks, each at a point the model already names:

    prepare_mlp        its second return value IS h_l -- the residual after this layer's attention
                       and before its feed-forward. Wrapped on the layers some other layer reads,
                       so the stack does not hold activations nobody wants.
    forward_prepare_*  the four variants all take `hidden_states` and return (q, k, v, gate). The
                       wrapper runs the same one twice when a source is set: once on this layer's
                       own normalised input for the key and value, once on the source's for the
                       query. Wrapping all four uniformly avoids re-implementing the dispatch,
                       which would drift from upstream the first time a branch is added.
    forward            sets the source before the attention runs and clears it after, so a layer
                       that is not converted cannot inherit the previous one's.

The query and the key/value therefore come from different points of the residual stream, and
everything else -- the norms, the rotation, the qk-norm, the gate, the parameter count -- is the
model's own.

## What this costs, and why it is written this way first

`qkv_proj` is fused: one projection emits q, gate, k and v together. Reading the query from a
different stream therefore runs that projection twice on a converted layer. Slicing the q rows out
of the fused weight would avoid it, and under FP8 that means carrying the scales too -- worth doing,
and not worth doing before the arrangement is known to run at all. The extra projection is
arithmetic on the compute side, which is the side this arrangement is trying to keep busy.

## The stash is keyed by layer, not by request

One forward pass is one batch: `h_l` is a single tensor covering every token in it. There is no
per-request loop here to confuse, and the (request, layer) keying that `pool_client` needs is a
property of the WIRE, where two requests really are in flight at once, not of this.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from sglang.srt.afd.read_point import ReadPlan, full_attention_layers, plan_read_points

logger = logging.getLogger(__name__)

PREPARE_METHODS = (
    "forward_prepare_cuda_fused",
    "forward_prepare_fused_gate",
    "forward_prepare_native",
    "forward_prepare_npu",
)


class LayerStash:
    """`h_l` for the layers some converted layer reads, for the duration of one forward pass."""

    def __init__(self) -> None:
        self._h: dict[int, torch.Tensor] = {}

    def put(self, layer: int, h: torch.Tensor) -> None:
        self._h[layer] = h

    def get(self, layer: int) -> torch.Tensor | None:
        return self._h.get(layer)

    def clear(self) -> None:
        self._h.clear()

    def __len__(self) -> int:
        return len(self._h)


def _wrap_prepare_mlp(layer, layer_id: int, stash: LayerStash) -> Callable:
    original = layer.layer_communicator.prepare_mlp

    def wrapped(*args, **kwargs):
        hidden_states, residual = original(*args, **kwargs)
        # residual here is h_l. The layer's OUTPUT would be x_(l+1), which is the standard read
        # point with extra steps: a correct model, no overlap, and nothing to notice it by.
        stash.put(layer_id, residual)
        return hidden_states, residual

    layer.layer_communicator.prepare_mlp = wrapped
    return original


def _wrap_prepare(layer, name: str) -> Callable:
    """Run the same prepare twice when a source is set: q from it, k and v from this layer.

    `self_attention` calls every variant by keyword, so the wrapper does too rather than
    guessing at positions.
    """
    original = getattr(layer, name)

    def wrapped(positions, hidden_states, **kwargs):
        q, k, v, gate = original(positions=positions, hidden_states=hidden_states, **kwargs)
        source = layer._afd_q_hidden
        if source is None:
            return q, k, v, gate
        q_early, _, _, gate_early = original(
            positions=positions, hidden_states=source, **kwargs
        )
        return q_early, k, v, gate_early

    setattr(layer, name, wrapped)
    return original


def _wrap_layer_forward(layer, layer_id: int, source_of: dict[int, int],
                        stash: LayerStash, norm) -> Callable:
    original = layer.forward

    def wrapped(*args, **kwargs):
        src = source_of.get(layer_id)
        h = stash.get(src) if src is not None and src >= 0 else None
        # LN1 belongs with the attention sub-layer, so the early stream gets the same
        # normalisation this layer's own input would have got.
        layer._afd_q_hidden = norm(h) if h is not None else None
        try:
            return original(*args, **kwargs)
        finally:
            layer._afd_q_hidden = None

    layer.forward = wrapped
    return original


class InstalledWiring:
    """What was changed, so a run can record it and a test can put it back."""

    def __init__(self, plan: ReadPlan, stash: LayerStash, undo: list[Callable]):
        self.plan = plan
        self.stash = stash
        self._undo = undo

    def record(self) -> dict:
        return self.plan.as_record()

    def remove(self) -> None:
        for fn in self._undo:
            fn()
        self._undo.clear()
        self.stash.clear()


def install_early_q(model, shift_layers: int, layer_types: list[str]) -> InstalledWiring:
    """Move the query's read point on every full-attention layer the shift can reach.

    `layer_types` names which layers are softmax attention: a linear-attention layer has no cache
    to sweep, and converting one would be a different intervention than the one that was measured.
    """
    layers = model.model.layers
    convertible = full_attention_layers(layer_types)
    plan = plan_read_points(shift_layers, len(layers), convertible=convertible)
    if shift_layers > 0 and not plan.moved:
        raise RuntimeError(
            f"--afd-q-shift-layers={shift_layers} moves no layer of this {len(layers)}-layer "
            f"stack. A conversion that hooks nothing costs nothing, and a cost of zero reads as "
            f"tolerance rather than as a wiring that never installed."
        )

    stash, undo = LayerStash(), []
    source_of = {p.layer: p.source for p in plan.points if not p.clamped}
    needed = sorted({s for s in source_of.values()})

    for layer_id in needed:
        layer = layers[layer_id]
        original = _wrap_prepare_mlp(layer, layer_id, stash)
        undo.append(lambda ly=layer, o=original: setattr(ly.layer_communicator, "prepare_mlp", o))

    for layer_id in sorted(source_of):
        layer = layers[layer_id]
        layer._afd_q_hidden = None
        for name in PREPARE_METHODS:
            # Accessed directly, not guarded: all four exist on this class, and a missing one
            # means the model file changed under this patch, which should be loud.
            original = _wrap_prepare(layer, name)
            undo.append(lambda ly=layer, n=name, o=original: setattr(ly, n, o))
        original = _wrap_layer_forward(layer, layer_id, source_of, stash, layer.input_layernorm)
        undo.append(lambda ly=layer, o=original: setattr(ly, "forward", o))

    logger.info(
        "afd early-q installed: shift=%s (%s half-layers), %s layer(s) moved, %s clamped, "
        "%s layer(s) stashed",
        plan.shift_layers,
        plan.half_layers,
        len(plan.moved),
        len(plan.clamped),
        len(needed),
    )
    return InstalledWiring(plan, stash, undo)
