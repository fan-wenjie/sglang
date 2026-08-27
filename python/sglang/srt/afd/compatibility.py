"""Which of sglang's features this arrangement can be run with, decided before a model loads.

The arrangement replaces a layer's feed-forward with a socket call and moves a query's read point
a layer earlier. Both are surgery on the forward pass, and sglang has a great many features that
also touch the forward pass. Most combinations have never been run; a few are known to be wrong.

Every one of them fails in the same bad way if nothing checks: the server starts, the tokens come
out fluent, and the number that was supposed to be measured is a number about something else. That
has happened repeatedly in this arrangement's own history -- a verifier that never covered the
remote path, a construction hook that reached one of sixteen loaders, a module flag that did not
cross a process boundary -- and in each case what was missing was a check, not a fix.

So: everything unsupported is refused by name, at startup, with the reason.

## Three kinds of answer, and why they are separate

    BROKEN     known to produce a wrong result or a hang. Refused; no override, because an
               override here is a foot-gun whose report would be a bug report about sglang
    UNTESTED   no reason to think it is broken and no evidence it works. Refused by default and
               overridable, because somebody has to run it first for it to stop being untested,
               and that person should have to say so
    FINE       run, or orthogonal to everything this touches

The overlap scheduler is in the third group and was briefly in the second, which is worth keeping
as an example of how this list goes wrong. It was listed UNTESTED on the reasoning that a forward
overlapping the previous one's output would disturb per-layer state the router keys by request --
plausible, and false: the two-machine deployment runs with `disable_overlap_schedule=False`, which
is sglang's default, and its tokens match the local split arm exactly. The check was refusing the
configuration it was written inside of. A guess about a feature belongs here only after somebody
has looked at what the running system actually sets.

The override is one environment variable naming the features to allow, so a run that used one says
so in its own command line and a grep of the logs finds it later:

    AFD_ALLOW_UNTESTED=cuda_graph,tp_size python -m sglang.launch_server ...
"""

from __future__ import annotations

import logging

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

BROKEN, UNTESTED = "broken", "untested"


class Incompatible(ValueError):
    """A server configuration this arrangement cannot serve correctly."""


def _cuda_graph_on(args) -> bool:
    """Whether any decode graph will be captured.

    Graphs are ON unless something turns them off, which is sglang's default and therefore the
    right default for a check whose job is to catch the ordinary launch. The first version of this
    read the config object and returned False when it was absent -- so a server started without a
    graph config, which is the common case and has graphs enabled, walked straight past a check
    written to stop exactly that. A predicate that answers "no" when it does not know is a
    predicate that never fires.

    Read from both the switches and the config, because a deployment may set either.

    Read DIRECTLY off the server args, not defensively. Every field named here is one sglang
    always defines, so `getattr(args, name, default)` could only ever hide a rename -- and it
    would hide it as "the default", which for this predicate means "no graphs" and means the check
    silently stops firing. The config object's own attributes stay guarded because it is optional
    and may legitimately be absent.
    """
    if args.disable_cuda_graph:
        return False
    if args.disable_decode_cuda_graph:
        return False
    config = args.cuda_graph_config
    decode = getattr(config, "decode", None) if config is not None else None
    backend = getattr(decode, "backend", None) if decode is not None else None
    if backend == "disabled":
        return False
    return True


# (name, predicate, kind, why). The predicate takes the resolved server args.
CHECKS = (
    (
        "cuda_graph",
        _cuda_graph_on,
        BROKEN,
        "the router replaces a layer's mlp.forward with a blocking socket call, and a captured "
        "CUDA graph cannot contain one. Every measurement of this arrangement was taken with "
        "--disable-cuda-graph, which also means none of them is against sglang's default decode "
        "path. Run with --disable-cuda-graph.",
    ),
    (
        "tp_size",
        lambda a: a.tp_size > 1,
        UNTESTED,
        "each rank would hold a shard of the layer and call the pool separately, and nothing "
        "decides how one departure batches frames from several ranks. The pool would answer each "
        "rank's slice as if it were a whole layer.",
    ),
    (
        "pp_size",
        lambda a: a.pp_size > 1,
        UNTESTED,
        "pipeline stages and a per-layer router have never been run together.",
    ),
    (
        "dp_attention",
        lambda a: bool(a.enable_dp_attention),
        UNTESTED,
        "data-parallel attention changes which rows a rank holds, and the append ledger keys "
        "histories by the row ids of a single rank's batch.",
    ),
    (
        "speculative_decoding",
        lambda a: a.speculative_algorithm is not None,
        UNTESTED,
        "a draft model's layers are not routed, and a verify step's batch is a shape the "
        "departure queue has never been given.",
    ),
    (
        "lora",
        lambda a: bool(a.enable_lora or a.lora_paths),
        BROKEN,
        "an adapter modifies the weights of the layers the pool holds, and the pool has no "
        "adapter. The host would apply the adapter to a feed-forward it does not run.",
    ),
    (
        "multimodal",
        lambda a: bool(a.enable_multimodal),
        UNTESTED,
        "positions are carried on the wire in a form that handles mrope, but an image's prefill "
        "path through the router has never been run.",
    ),
    (
        "weight_update",
        lambda a: a.rl_on_policy_target is not None,
        BROKEN,
        "the pool holds weights this host cannot update, so an update would leave the two ends "
        "serving different models and nothing would say so.",
    ),
)


def _allowed() -> set:
    return set(envs.SGLANG_AFD_ALLOW_UNTESTED.get())


def check(server_args) -> dict:
    """Refuse what cannot work, name what has never been tried. Returns what was checked.

    Called for both roles. A pool serves feed-forwards and holds no cache, so several of these do
    not apply to it -- but a pool started with a draft model or a LoRA is a pool whose weights are
    not the host's, and that is worth refusing at either end.
    """
    allowed = _allowed()
    hit_broken, hit_untested, ran = [], [], []
    for name, predicate, kind, why in CHECKS:
        try:
            engaged = bool(predicate(server_args))
        except Exception:  # an arg this build does not have is not engaged
            engaged = False
        ran.append(name)
        if not engaged:
            continue
        if kind is BROKEN:
            hit_broken.append((name, why))
        elif name in allowed:
            logger.warning(
                "afd: %s is UNTESTED with this arrangement and AFD_ALLOW_UNTESTED permits it. "
                "Nothing here knows whether the result is correct. %s",
                name,
                why,
            )
        else:
            hit_untested.append((name, why))

    if hit_broken or hit_untested:
        lines = []
        for name, why in hit_broken:
            lines.append(f"  {name}: BROKEN with afd. {why}")
        for name, why in hit_untested:
            lines.append(
                f"  {name}: UNTESTED with afd. {why} Set AFD_ALLOW_UNTESTED={name} to run it "
                f"anyway and be the one who finds out."
            )
        raise Incompatible(
            "this configuration combines afd with sglang features it cannot serve correctly:\n"
            + "\n".join(lines)
            + "\n\nEach of these fails the same way when nothing checks: the server starts, the "
            "output is fluent, and the measurement is of something else."
        )
    return {"checked": ran, "allowed_untested": sorted(allowed)}
