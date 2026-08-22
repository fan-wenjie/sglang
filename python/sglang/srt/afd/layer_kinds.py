"""What kind of layer each layer is, asked of the built model rather than of a config.

Separated from `read_point` because nothing here concerns a read point. Four modules on the shared
side of the arrangement -- `roles`, `supported`, `sweep_ahead`, `remote_attention` -- need to know
which layers hold a KV cache and which hold a recurrent state, and they needed it before any
shift existed. Importing it from the module that moves the query's read point made the shared half
depend on the derived one, which is the direction that must not exist if standard AFD is to stand
on its own.

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


