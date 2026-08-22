"""Strip the weights this cut computes on the pool, once it is installed and only what it routed.

Measured before this existed: the host's resident weights are 19.18 GB under the per-layer cut AND
under the group cut, to the digit. The loader makes exactly one thing absent -- the feed-forward --
so a group-cut host holds a full set of attention projections it never multiplies by anything. The
group cut moves where those are COMPUTED and, until now, changed nothing about what is allocated.

That matters because of what the arrangement is for. A 24 GB card with 13.5 GB of KV cache for a
512K context has about 10 GB left; a host carrying 19.18 GB of weights cannot be one of the many
small cards this whole split exists to make useful.

## Why this strips modules rather than building them on meta

`absent_ffn` wraps a CLASS's `__init__` so the feed-forward is never allocated at all, which also
avoids the construction peak. That works there because `Qwen2MoeMLP` is specific to the thing being
routed. It does not extend here: a host's attention projections are `QKVParallelLinear` and
`RowParallelLinear`, classes shared with every other projection in the model including ones this
host still uses and the vision tower's. Wrapping them would strip storage from modules nobody
routed, and the symptom is a meta tensor reaching a matmul -- a message about devices, from
somewhere that names neither the cut nor the module.

So this runs AFTER the routing is installed and takes the modules the routing itself names. The
peak during construction is unchanged and that is stated rather than glossed: this recovers
steady-state memory, which is what decides how much KV cache fits, and not the peak, which is what
decides whether the model can be built at all. Both matter and they are different numbers.

## What is safe to strip under the group cut, and why

    passenger layers    run entirely on the pool -- the host's forward is a pass-through, so the
                        whole decoder layer's weights are unused
    head layers         the host runs the softmax attention CORE, which has no weights: it reads
                        its own KV cache. The query, key and value arrive already projected in the
                        span's reply, and `run_epilogue` applies `o_proj` ON THE POOL. So the
                        head's own qkv_proj and o_proj are unused too
    norms               kept. They are vectors, they cost nothing, and `post_attention_layernorm`
                        is called by the span's own bookkeeping on this side

What is NOT stripped: anything on a layer the routing does not speak for. The list comes from the
routing's own `heads` and `passengers`, so a half-installed cut strips exactly the half it routed.
"""

from __future__ import annotations

import logging

from sglang.srt.afd.absent_ffn import to_meta

logger = logging.getLogger(__name__)

# On a head layer the host keeps the attention core and the norms and needs neither projection.
HEAD_UNUSED = ("qkv_proj", "o_proj")


def strip_routed_weights(model, routing) -> dict:
    """Release the weights the pool computes. Returns what went, for the record.

    Called with the routing rather than with a flag: what is safe to strip is exactly what was
    routed, and the routing is the only thing that knows which layers those are.
    """
    layers = model.model.layers
    freed, touched = 0, {"passenger_layers": 0, "head_layers": 0}

    for index in sorted(routing.passengers):
        freed += to_meta(layers[index])
        touched["passenger_layers"] += 1

    for index in sorted(routing.heads):
        layer = layers[index]
        for name in HEAD_UNUSED:
            module = getattr(layer, name, None)
            if module is None:
                continue
            freed += to_meta(module)
        # the feed-forward of a head layer runs on the pool too, in `run_epilogue`
        if getattr(layer, "mlp", None) is not None:
            freed += to_meta(layer.mlp)
        touched["head_layers"] += 1

    report = {"gib_freed": freed / 1024 ** 3, **touched}
    logger.info(
        "afd host: released %.2f GiB of weights the pool computes -- %s passenger layer(s) whole "
        "and the projections of %s head layer(s). The construction peak is unchanged; this is "
        "steady-state memory, which is what decides how much KV cache fits.",
        report["gib_freed"], touched["passenger_layers"], touched["head_layers"],
    )
    return report
