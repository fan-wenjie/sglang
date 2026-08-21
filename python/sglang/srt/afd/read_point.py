"""Where each layer's query is read from, and the bookkeeping that keeps that honest.

`--afd-query-shift-layers N` is one number wearing four hats, which is why it is one knob:

    offset      = N - 0.5 layers      how far back the query is read
    source      = h_{l-N}             the residual it is read from
    group       = N layers            what the shift spans
    half-layers = 2N - 1              the study's unit for the same depth

`N = 0` is the standard wiring. `N = 1` reads `h_{l-1}` -- half a layer back, the operating point.

`h_l` is the residual stream AFTER layer l's attention and BEFORE its feed-forward. In sglang that
is exactly the `residual` returned by `layer_communicator.prepare_mlp`. Reading the layer's output
instead gives `x_{l+1}`, which is the standard wiring with extra steps: a correct model, no
overlap, and no error to notice it by.

On a hybrid stack the group reading is the point. Qwen3.8-27B is 64 layers with
`full_attention_interval = 4`, so its full-attention layers are 3, 7, 11, ... -- the last of each
group of four. At `N = 4` layer 7 reads `h_3` and layer 11 reads `h_7`: each softmax layer's sweep
overlaps the whole group behind it.
"""

from __future__ import annotations

from typing import NamedTuple


class ReadPoint(NamedTuple):
    """One layer's resolved source, and whether the stack was too shallow to give it."""

    layer: int
    source: int          # the index whose h is read; -1 means the embedding
    clamped: bool        # the stack was shallower than the shift asked for


class ReadPlan(NamedTuple):
    """Every converted layer's read point, plus the numbers a run record has to carry."""

    shift_layers: int
    points: tuple[ReadPoint, ...]
    n_layers: int

    @property
    def offset_layers(self) -> float:
        return 0.0 if self.shift_layers == 0 else self.shift_layers - 0.5

    @property
    def half_layers(self) -> int:
        return 0 if self.shift_layers == 0 else 2 * self.shift_layers - 1

    @property
    def clamped(self) -> tuple[int, ...]:
        return tuple(p.layer for p in self.points if p.clamped)

    @property
    def moved(self) -> tuple[int, ...]:
        """The layers whose read point ACTUALLY moved.

        A clamped layer is in `points` -- the plan considered it -- but it keeps the standard read
        point, because there is no earlier residual for it to read. Counting it as converted is
        how a run reports a shallower shift's coverage under a deeper shift's name, so `moved` and
        `clamped` are separate numbers and both are in the record.
        """
        return tuple(p.layer for p in self.points if not p.clamped)

    def as_record(self) -> dict:
        return {
            "shift_layers": self.shift_layers,
            "offset_layers": self.offset_layers,
            "half_layers": self.half_layers,
            "n_layers": self.n_layers,
            "considered": [p.layer for p in self.points],
            "moved": list(self.moved),
            "sources": {str(p.layer): p.source for p in self.points if not p.clamped},
            "clamped": list(self.clamped),
            "n_considered": len(self.points),
            "n_moved": len(self.moved),
            "n_clamped": len(self.clamped),
        }


def plan_read_points(
    shift_layers: int,
    n_layers: int,
    convertible: tuple[int, ...] | None = None,
) -> ReadPlan:
    """Resolve the read point of every convertible layer.

    `convertible` names the layers whose query moves -- on a hybrid stack, the full-attention
    layers. None means every layer.

    Layer 0 is exempt at any shift: its query reads the embedding under either wiring, so moving
    it is not a move. A layer with fewer than `shift_layers` layers beneath it CLAMPS to the
    bottom, and the clamp is returned rather than hidden: a run that silently converts fewer
    layers than it was asked for reports a shallower shift's cost under a deeper shift's name, and
    the number looks better for it.
    """
    if not isinstance(shift_layers, int) or isinstance(shift_layers, bool):
        raise TypeError(
            f"--afd-query-shift-layers is a layer count, got {shift_layers!r}. There is no fractional "
            f"setting: a query read between a block's two sub-layers is read from a point where "
            f"the residual stream has no value."
        )
    if shift_layers < 0:
        raise ValueError(f"--afd-query-shift-layers must not be negative, got {shift_layers}")
    if n_layers <= 0:
        raise ValueError(f"a stack has at least one layer, got {n_layers}")
    if shift_layers == 0:
        return ReadPlan(0, (), n_layers)

    candidates = tuple(range(n_layers)) if convertible is None else tuple(convertible)
    for layer in candidates:
        if not 0 <= layer < n_layers:
            raise ValueError(f"convertible layer {layer} is outside a {n_layers}-layer stack")

    points = []
    for layer in candidates:
        if layer == 0:
            continue                      # exempt: reads the embedding either way
        source = layer - shift_layers
        clamped = source < 0
        points.append(ReadPoint(layer=layer, source=max(source, -1), clamped=clamped))
    return ReadPlan(shift_layers, tuple(points), n_layers)


def full_attention_layers(layer_types: list[str]) -> tuple[int, ...]:
    """The layers whose attention is a sweep over a cache: the softmax ones.

    This answers the DEPLOYMENT question -- which layers can start their attention before the
    feed-forward that precedes them, because a sweep over cached keys needs only the query. A
    linear-attention layer has no cache to sweep; its state update is sequential.

    It is NOT the answer to "which layers' queries move". Those are two questions and they were
    one function here for a while, which silently capped coverage at the softmax layers.
    """
    return tuple(i for i, t in enumerate(layer_types) if t == "full_attention")


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


def is_full_attention(layer) -> bool:
    """Whether this layer sweeps a KV cache, decided in the one place that decides it.

    Asking `hasattr(layer, "attn")` at each call site is the same question answered separately in
    each module, and the day a layer grows an `attn` attribute for another reason they answer it
    differently. This routes through the same class-name rule `layer_types_of` uses, so a stack
    whose layers are neither kind is refused once rather than silently sorted twice.
    """
    return _kind_of(layer) == "full_attention"


def _kind_of(layer) -> str:
    name = type(layer).__name__
    if "Linear" in name:
        return "linear_attention"
    if "Attention" in name:
        return "full_attention"
    raise RuntimeError(
        f"a {name} is neither of the two kinds this arm knows how to treat. Name it before "
        f"converting it: a layer handled by accident is a layer nothing has checked."
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


