"""The read point as data: where each layer's query comes from, and where h_l is held.

`wiring.py` installs the same plan onto a live sglang stack by wrapping methods. This is the same
idea with no model attached -- a stash keyed by (request, layer) and a lookup that says which
source a layer reads -- and it is what the plan's behaviour is pinned by, because a rule tested
only through a wrapped model is a rule tested through everything the model also does.

Two failures this file exists to make loud rather than quiet:

  * **stashing the wrong tensor.** The layer's OUTPUT is `x_{l+1}`, not `h_l`. Reading it produces
    a correct model with no overlap and no error, which is the worst of the three outcomes. The
    stash records WHICH stage wrote it and refuses a write from any other.
  * **converting nothing.** A conversion that hooks no layer costs nothing, and a cost of zero
    reads as tolerance rather than as a hook that never installed. `install` raises when a
    non-zero shift converts no layer at all.

## Why a clamped layer keeps its own input

A layer with fewer than `shift_layers` beneath it has no earlier residual to read. It could be
refused, or clamped to the bottom, or left alone; it is left alone, and `query_source` returns the
layer's own `x_l`. The plan still lists it under `considered` and marks it `clamped`, so a run
reports the coverage it achieved rather than the coverage it asked for -- those differ by one
layer on this model, and a number quoted under the wrong one is a shallower shift's cost wearing
a deeper shift's name.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.afd.read_point import ReadPlan, plan_read_points

logger = logging.getLogger(__name__)


class EarlyQStash:
    """Per-request, per-layer `h_l`, and the stage that produced it."""

    # `h_l` is the residual AFTER a layer's attention and BEFORE its feed-forward. In sglang that
    # is what `layer_communicator.prepare_mlp` returns as `residual`, and on a family without a
    # communicator it is the second output of the fused add-and-normalise.
    STAGE = "post_attention_pre_mlp"

    def __init__(self) -> None:
        self._h: dict[tuple[int, int], torch.Tensor] = {}

    def put(self, request_id: int, layer: int, h: torch.Tensor, stage: str) -> None:
        if stage != self.STAGE:
            raise ValueError(
                f"the stash holds h_l, written at {self.STAGE!r}; got {stage!r}. The layer's "
                f"output is x_(l+1) and reading it would be the standard wiring with extra steps: "
                f"a correct model, no overlap, and nothing to notice it by."
            )
        # keyed by BOTH, because a decode batch carries one token from each of several requests
        # and a stash keyed by layer alone would have them overwrite each other
        self._h[(request_id, layer)] = h

    def has(self, request_id: int, layer: int) -> bool:
        return (request_id, layer) in self._h

    def take(self, request_id: int, layer: int) -> torch.Tensor:
        key = (request_id, layer)
        if key not in self._h:
            raise KeyError(
                f"no h_{layer} stashed for request {request_id}; the layer whose query reads it "
                f"ran before the layer that produces it, which is a plan error, not a race"
            )
        return self._h[key]

    def drop_request(self, request_id: int) -> int:
        """Forget one request's residuals. Returns how many were held."""
        keys = [key for key in self._h if key[0] == request_id]
        for key in keys:
            del self._h[key]
        return len(keys)

    def clear(self) -> None:
        self._h.clear()

    def held(self) -> int:
        return len(self._h)


class EarlyQWiring:
    """A resolved plan, answering the two questions a forward pass asks of it."""

    def __init__(self, plan: ReadPlan, stash: EarlyQStash | None = None) -> None:
        self.plan = plan
        self.stash = stash if stash is not None else EarlyQStash()
        self._source_of = {p.layer: p.source for p in plan.points if not p.clamped}
        self._needed = set(self._source_of.values())

    def needs_stash(self, layer: int) -> bool:
        """Whether any layer reads this one's h. Holding the rest would be holding the stack."""
        return layer in self._needed

    def query_source(self, request_id: int, layer: int, x_l: torch.Tensor) -> torch.Tensor:
        """The tensor this layer's query projects from.

        `x_l` for a layer that was not converted or that clamped; the stashed `h_source` for one
        that was. Returning `x_l` unchanged rather than raising is what makes a clamped layer a
        correct layer: it is the standard wiring, which is what it was always going to be.
        """
        source = self._source_of.get(layer)
        if source is None:
            return x_l
        return self.stash.take(request_id, source)

    def record(self) -> dict:
        return self.plan.as_record()


def install(shift_layers: int, n_layers: int, convertible=None) -> EarlyQWiring:
    """Resolve a plan, refusing one that would convert nothing.

    A shift that hooks no layer costs nothing to run and reads as tolerance -- "we tried it and it
    was free" -- when what happened is that it never installed. A one-layer stack has only the
    exempt layer 0, so asking for any shift on one is asking for a conversion that cannot happen.
    """
    plan = plan_read_points(shift_layers, n_layers, convertible=convertible)
    if shift_layers and not plan.moved:
        raise RuntimeError(
            f"--afd-query-shift-layers={shift_layers} on a {n_layers}-layer stack converts no layer: "
            f"{len(plan.points)} considered, all clamped or exempt. A conversion that hooks "
            f"nothing costs nothing, and a cost of zero would be read as this rewiring being free."
        )
    logger.info("afd early-q plan: %s", plan.as_record())
    return EarlyQWiring(plan)
