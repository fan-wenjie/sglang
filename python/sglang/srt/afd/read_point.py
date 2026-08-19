"""Where each layer's query is read from, and the bookkeeping that keeps that honest.

`--afd-q-shift-layers N` is one number wearing four hats, which is why it is one knob:

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
            f"--afd-q-shift-layers is a layer count, got {shift_layers!r}. There is no fractional "
            f"setting: a query read between a block's two sub-layers is read from a point where "
            f"the residual stream has no value."
        )
    if shift_layers < 0:
        raise ValueError(f"--afd-q-shift-layers must not be negative, got {shift_layers}")
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
    """The layers whose query this project moves: the softmax-attention ones.

    A linear-attention layer has no cache to sweep and no query in the sense that matters here;
    its state update is sequential. Converting one would be a different intervention than the one
    that was measured, so the arm names them rather than letting a loop over all layers decide.
    """
    return tuple(i for i, t in enumerate(layer_types) if t == "full_attention")


def layer_types_of(model) -> list[str]:
    """Read the layer kinds off the LOADED stack, not off a config field.

    The checkpoint's config.json carries `layer_types`, and sglang's own config class does not:
    it derives the pattern from `full_attention_interval`. Asking the built model which class each
    layer is answers the question both ways round and cannot disagree with what is actually there
    -- a config field says what was requested, the module list says what was constructed.
    """
    kinds = []
    for layer in model.model.layers:
        name = type(layer).__name__
        if "Linear" in name:
            kinds.append("linear_attention")
        elif "Attention" in name:
            kinds.append("full_attention")
        else:
            raise RuntimeError(
                f"layer {len(kinds)} is a {name}, which is neither of the two kinds this arm "
                f"knows how to treat. Name it before converting it: a layer handled by accident "
                f"is a different intervention than the one that was measured."
            )
    return kinds
