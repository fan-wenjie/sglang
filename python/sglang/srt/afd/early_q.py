"""Wire a decoder stack so a layer's query is read from an earlier point of the residual stream.

The intervention is one tensor. In sglang's Qwen3.5 attention layer the residual stream is walked
by three named stages:

    hidden, residual = layer_communicator.prepare_attn(...)   # hidden = LN1(x_l),  residual = x_l
    hidden           = self.self_attention(...)
    hidden, residual = layer_communicator.prepare_mlp(...)    # hidden = LN2(h_l),  residual = h_l
    hidden           = self.mlp(hidden)
    hidden, residual = layer_communicator.postprocess_layer(...)

`residual` after `prepare_mlp` is `h_l`. `EarlyQStash` captures it there and hands it to the layer
whose read point names it. Nothing else about the block moves: the key and value still project
from `x_l`, the normalisations stay with their own sub-layers, and the parameter count, the
arithmetic and the serial depth are unchanged.

Two failures this file is built to make loud rather than quiet:

  * **stashing the wrong tensor.** The layer's OUTPUT is `x_{l+1}`, not `h_l`. Reading it produces
    a correct model with no overlap and no error, which is the worst of the three outcomes. The
    stash therefore records WHICH stage wrote it and refuses a read from a stage it did not.
  * **converting nothing.** A conversion that hooks no layer costs nothing, and a cost of zero
    reads as tolerance rather than as a hook that never installed. `install` returns the plan and
    raises when a non-zero shift converts no layer at all.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch

from sglang.srt.afd.read_point import ReadPlan, plan_read_points

logger = logging.getLogger(__name__)


class EarlyQStash:
    """Per-request, per-layer `h_l`, written where `prepare_mlp` returns it."""

    STAGE = "post_attention_pre_mlp"

    def __init__(self) -> None:
        self._h: dict[tuple[int, int], torch.Tensor] = {}
        self._stage: dict[tuple[int, int], str] = {}

    def put(self, request_id: int, layer: int, h: torch.Tensor, stage: str) -> None:
        if stage != self.STAGE:
            raise ValueError(
                f"the stash holds h_l, written at {self.STAGE!r}; got {stage!r}. The layer's "
                f"output is x_(l+1) and reading it would be the standard wiring with extra steps."
            )
        self._h[(request_id, layer)] = h
        self._stage[(request_id, layer)] = stage

    def take(self, request_id: int, layer: int) -> torch.Tensor:
        key = (request_id, layer)
        if key not in self._h:
            raise KeyError(
                f"no h_{layer} stashed for request {request_id}; the layer whose query reads it "
                f"ran before the layer that produces it, which is a plan error, not a race"
            )
        return self._h[key]

    def has(self, request_id: int, layer: int) -> bool:
        return (request_id, layer) in self._h

    def drop_request(self, request_id: int) -> None:
        for key in [k for k in self._h if k[0] == request_id]:
            self._h.pop(key, None)
            self._stage.pop(key, None)

    def __len__(self) -> int:
        return len(self._h)


class EarlyQWiring:
    """The resolved plan plus the stash, and the two calls a layer makes."""

    def __init__(self, plan: ReadPlan, stash: EarlyQStash | None = None):
        self.plan = plan
        self.stash = stash if stash is not None else EarlyQStash()
        self._source_of = {p.layer: p.source for p in plan.points}
        # every layer whose h is named by somebody: only these need stashing, and stashing the
        # rest would hold the whole stack's activations for a shift that reads two of them.
        self._needed = frozenset(p.source for p in plan.points if p.source >= 0)

    @property
    def converted(self) -> tuple[int, ...]:
        return tuple(self._source_of)

    def needs_stash(self, layer: int) -> bool:
        return layer in self._needed

    def query_source(self, request_id: int, layer: int, x_l: torch.Tensor) -> torch.Tensor:
        """The tensor layer `layer` should project its query from.

        Falls back to `x_l` for a layer the plan does not convert, and for a converted layer whose
        source clamped below the stack -- at the bottom there is no earlier point, and pretending
        otherwise would silently read the wrong thing.
        """
        source = self._source_of.get(layer)
        if source is None or source < 0:
            return x_l
        return self.stash.take(request_id, source)


def install(
    shift_layers: int,
    n_layers: int,
    convertible: tuple[int, ...] | None = None,
) -> EarlyQWiring:
    """Resolve the read plan and refuse the silent no-op."""
    plan = plan_read_points(shift_layers, n_layers, convertible)
    if shift_layers > 0 and not plan.points:
        raise RuntimeError(
            f"--afd-q-shift-layers={shift_layers} converts no layer of a {n_layers}-layer stack "
            f"(convertible={convertible}). A conversion that hooks nothing costs nothing, and a "
            f"cost of zero reads as tolerance rather than as a hook that never installed."
        )
    if plan.clamped:
        logger.warning(
            "afd early-q: layers %s clamp -- the stack is shallower than a %s-layer shift asks "
            "for, so they keep the standard read point and are recorded as clamped",
            list(plan.clamped),
            shift_layers,
        )
    logger.info(
        "afd early-q: shift=%s layers (%s half-layers, offset %.1f), %s converted, %s clamped",
        plan.shift_layers,
        plan.half_layers,
        plan.offset_layers,
        len(plan.points),
        len(plan.clamped),
    )
    return EarlyQWiring(plan)
