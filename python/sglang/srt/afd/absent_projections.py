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
    norms               kept, and the reason first written here was WRONG. It said
                        `post_attention_layernorm` is called on this side; `span_routing.py`
                        contains no norm call at all -- every per-layer norm is applied on the
                        pool, inside `_add_and_norm`, `_finish` and `run_epilogue`. A head
                        layer's norms are therefore held and never used, and they are kept only
                        because releasing 1.2 MiB is not worth a line of code.

                        The norm the host DOES apply is a different one: `model.norm`, the final
                        RMSNorm after the layer loop. `run_epilogue` deliberately returns the
                        un-normalised residual stream so sglang's own forward applies it here --
                        normalising on the pool as well applied it twice, which with this
                        checkpoint's weights running -0.285 to 1.711 squares every channel and
                        flips the sign of the negative ones

What is NOT stripped: anything on a layer the routing does not speak for. The list comes from the
routing's own `heads` and `passengers`, so a half-installed cut strips exactly the half it routed.
"""

from __future__ import annotations

import logging

import torch


def release(module, keep: tuple = ()) -> int:
    """Move a module's parameters to meta and return the bytes THAT WERE ACTUALLY THERE.

    Not `absent_ffn.to_meta`, which counts every parameter it moves. Here the feed-forward has
    already been built on meta by the loader, so counting it again reported 45.36 GiB released on
    a host whose whole checkpoint was 19.18 GB -- a number that would have been quoted.

    `keep` names parameters this side still needs -- dotted, relative to `module`. They are saved,
    the module goes to meta with everything else, and they are put back. Restored by NAME rather
    than by holding the object, because `.to("meta")` replaces the parameter objects and a saved
    reference would point at a tensor nothing reads any more: the convolution would then run
    against a weight the module no longer owns, or not at all, and either way the answer stays
    fluent.
    """
    saved = {}
    for name in keep:
        owner, _, attr = name.rpartition(".")
        parent = module.get_submodule(owner) if owner else module
        value = getattr(parent, attr)
        if value is not None and not value.is_meta:
            saved[name] = (parent, attr, value.detach().clone())
    freed = sum(
        p.numel() * p.element_size()
        for n, p in module.named_parameters()
        if not p.is_meta and n not in saved
    )
    module.to("meta")
    for name, (parent, attr, value) in saved.items():
        held = getattr(parent, attr)
        setattr(
            parent, attr, torch.nn.Parameter(value, requires_grad=held.requires_grad)
        )
    return freed


logger = logging.getLogger(__name__)

# On a head layer the host keeps the attention core and the norms and needs neither projection.
HEAD_UNUSED = ("qkv_proj", "o_proj")


def _runs_convolution_here(routing) -> bool:
    """Whether this host holds the convolution ring, and therefore needs the weight that filters it.

    Read off the history rather than off a flag, because the ring's presence is the fact that
    matters and a flag can disagree with it. A history built without a real ring answers a MIX
    frame by refusing it, so the two are decided in one place.
    """
    history = routing.history
    return history is not None and history.conv_weight is not None


def _convolution_of(layer) -> tuple:
    """One layer's convolution, named the way `release` resolves it."""
    return ("linear_attn.conv1d.weight", "linear_attn.conv1d.bias")


def strip_routed_weights(model, routing, *, remote_embedding: bool) -> dict:
    """Release the weights the pool computes. Returns what went, for the record.

    Called with the routing rather than with a flag: what is safe to strip is exactly what was
    routed, and the routing is the only thing that knows which layers those are.

    `remote_embedding` arrives as an argument rather than being read off the global server args
    HERE. That read was inside this function and it made it uncallable from a unit test -- there
    is no global in one -- and the failure was an exception from `runtime_context` naming neither
    this function nor the flag. It is the fourth time on this line that a global read inside a
    function broke the thing that calls it; the composition root reads it and passes it down.
    """
    layers = model.model.layers
    freed, touched = 0, {"passenger_layers": 0, "head_layers": 0}

    for index in sorted(routing.passengers):
        # A passenger layer runs entirely on the pool -- except for its convolution, when this
        # host is the side that holds the ring. Releasing the whole layer then takes the one weight
        # this end still needs, and the failure is not an error: the convolution would be applied
        # against a freed tensor or skipped, and the answer would still be fluent.
        #
        # 80 KiB a layer, 3.75 MiB for all 48 on this model, measured from the checkpoint -- 0.035%
        # of what the layer weighs, which is what makes holding it here cheap enough to be the
        # design rather than a compromise.
        spare = (
            _convolution_of(layers[index]) if _runs_convolution_here(routing) else ()
        )
        freed += release(layers[index], keep=spare)
        touched["passenger_layers"] += 1

    for index in sorted(routing.heads):
        layer = layers[index]
        for name in HEAD_UNUSED:
            module = getattr(layer, name, None)
            if module is None:
                continue
            freed += release(module)
        # the feed-forward of a head layer runs on the pool too, in `run_epilogue`
        if getattr(layer, "mlp", None) is not None:
            freed += release(layer.mlp)
        touched["head_layers"] += 1

    # The embedding, when the pool does the lookup. Released here rather than named in
    # `absent_classes` because that names a CLASS and `VocabParallelEmbedding` is shared with the
    # vision tower -- naming it would strip storage from an embedding this host still uses.
    #
    # 2.368 GiB on this model: [248320, 5120], and `tie_word_embeddings` is false so it is a
    # distinct tensor from `lm_head`, which stays.
    if remote_embedding:
        before = freed
        freed += release(model.model.embed_tokens)
        touched["embedding_gib"] = (freed - before) / 1024**3

    report = {"gib_freed": freed / 1024**3, **touched}
    logger.info(
        "afd host: released %.2f GiB of weights the pool computes -- %s passenger layer(s) whole "
        "and the projections of %s head layer(s). The construction peak is unchanged; this is "
        "steady-state memory, which is what decides how much KV cache fits.",
        report["gib_freed"],
        touched["passenger_layers"],
        touched["head_layers"],
    )
    return report
