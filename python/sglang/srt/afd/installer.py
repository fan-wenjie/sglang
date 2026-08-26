"""Installing the span arrangement, on either end, and announcing it to AFD.

These five functions lived in `afd/roles.py`, which is the arrangement's composition root and must
not know that this arm exists -- AFD ships without this package, and a root that names it would
not start when the directory is absent.

They are unchanged from where they were. What is new is the bottom of the file: an object with the
two entry points the root needs, and one `register` call. The root asks `afd.arms` for an arm by
name, gets this or nothing, and constructs -- so the branch that used to read
`if span_cut_wanted()` now reads `if arm is not None`, and the word "span" does not appear in it.
"""

from __future__ import annotations

import logging

from sglang.srt.afd.arms import register
from sglang.srt.afd.pool_client import PoolClient

# Importing this module claims the span's entries in the pool's departure table. Here
# because `arms.load` imports it in every process, and a POOL process installs nothing
# else -- it holds weights and answers -- yet still has to know what a SPAN frame is.
from sglang.srt.afd import pool_span  # noqa: F401,E402

logger = logging.getLogger(__name__)


def remote_embedding_wanted(server_args) -> bool:
    """Whether the embedding lookup runs on the POOL. Derived, because its one condition is
    already a server setting.

    On, unless the deployment serves images. The host then records token ids and sends them at the
    first span instead of a hidden state -- 8 bytes a row rather than 10 KiB -- and does not hold
    `embed_tokens` at all, which is 2.368 GiB on Qwen3.8-27B.

    Off under multimodal, and not as a preference: `general_mm_embed_routine` mixes an image's
    embedding into the same tensor, and token ids cannot describe that. A pool sent ids would
    embed the text, drop the picture, and answer fluently. There is nothing for a flag to choose
    between -- one arrangement is wrong for images and the other wastes 2.368 GiB without them --
    so the condition is read rather than asked for.
    """
    return not server_args.enable_multimodal


def span_cut_wanted() -> bool:
    """Whether this process serves the span cut, which is this FAMILY's arrangement.

    The FAMILY chooses the cut and the read point stays a separate question. A checkpoint
    whose `layer_types` name a linear attention is served through the span cut whether or
    not anything asked for a shift: the cut at shift 0 is the standard wiring -- the same
    boundary from the output projection to the next key/value, the pool holding every
    weight the sweep does not need, serially -- and it is token-identical to the colocated
    model (pinned by test). What the shift selects is only WHERE the query is read:
    0 reads it at its own layer, 1 reads it one feed-forward early so the sweep overlaps.

    A checkpoint repaired to read `h_(l-1)` must be served at its stated shift, which
    needs the derived early-read package; a measurement override exists on the derived
    branch and is DANGEROUS by construction above 0, warned wherever it is honoured.

    Read from the global server args rather than passed down, for the reason `absent_ffn` reads it
    the same way: the pool role is set up inside the scheduler process, which is spawned, and a
    module-level flag set in the parent does not cross that boundary. That failure was silent once
    already here -- the feed-forward weights were built after the flag said not to -- so the value
    is fetched where it is used.
    """
    from sglang.srt.runtime_context import get_server_args

    return span_cut_wanted_for(get_server_args())


def span_cut_wanted_for(server_args) -> bool:
    """The same question, of args handed in rather than of the process's own.

    Two forms because two callers: the argument check has the args in its hand and runs before
    anything global is set, and everything after install has only the global. One implementation
    so the two cannot answer differently -- which is the failure this arm has had twice.
    """
    from sglang.srt.afd.checkpoint import (
        requested_shift,
        states_a_read_point,
    )

    from sglang.srt.afd.checkpoint import effective_model_path, serves_linear_layers

    if requested_shift(server_args) is not None:
        return True
    path = effective_model_path(server_args)
    if states_a_read_point(path):
        return True
    return serves_linear_layers(path)


def _conv_width(config) -> int:
    """The convolution's channel count: two key blocks and one value block, as the model lays it
    out. Derived from the config rather than from a tensor, because it has to be right before any
    weight is looked at -- the ring is allocated first.

    The widths come through `linear_widths` rather than off the config directly, because not
    every family spells them the same way and a ring built to the wrong width is a host that
    reads garbage rather than a host that refuses.
    """
    from sglang.srt.afd.layer_kinds import linear_widths

    w = linear_widths(config)
    return 2 * w["k_heads"] * w["dk"] + (w["v_heads"] * w["dv"])


def _span_slots() -> int:
    """How many requests the pool can hold recurrent state for.

    Taken from `--max-running-requests` rather than guessed. A slot table that ran out would refuse
    a request mid-generation, and a number invented here would put that cliff somewhere the
    operator never chose. If sglang has not resolved the limit yet there is nothing to derive from,
    and refusing at startup beats picking one.
    """
    from sglang.srt.runtime_context import get_schedule

    limit = get_schedule().max_running_requests
    if limit is None:
        raise ValueError(
            "the span cut needs --max-running-requests to size the pool's recurrent state "
            "table. A recurrent state is the whole history compressed and cannot be rebuilt from "
            "a prefix, so the table refuses rather than evicts -- and where that refusal falls "
            "has to be a number somebody chose."
        )
    return int(limit)


def _span_query_shift() -> int:
    """How far back this pool reads the next group's query. Read HERE, on the pool.

    Under the group cut the query projection moved to the pool with `W_q`, so the pool is the
    side that decides where the query is read from and the host's copy of the setting decides
    nothing at all.

    Asked of `resolved_shift`, which the wiring set when it read the checkpoint, so there is ONE
    resolution in the process and every reader gets the same answer. Resolving again here would
    be a second one, and two resolutions of the same question are two answers waiting to differ.
    They already did: a raw read of the setting gave 1 in the wiring and 0 here, and the span was
    installed for the shifted read point and then ran without it.
    """
    from sglang.srt.afd.checkpoint import resolved_shift

    shift = resolved_shift()
    if shift not in (0, 1):
        raise ValueError(
            f"query shift {shift} under the group cut. This cut reads the query from "
            f"between the last linear attention and the last feed-forward of a group, which is "
            f"shift 1, or from the group's output, which is shift 0. A deeper shift is a "
            f"different cut and would have to move the read point across a group boundary."
        )
    return int(shift)


def make_span_runner(model, *, device):
    """The pool's span runner, if the family asked for the span cut. None otherwise.

    Sized by `max_requests` because a recurrent state cannot be evicted and rebuilt from a prefix
    the way a KV cache can -- it is the whole history compressed -- so the slot table refuses a
    request rather than dropping one, and the refusal has to be far from the working point.
    """
    if not span_cut_wanted():
        return None
    # after the switch, never before it: a pool serving the per-layer cut has no recurrent state
    # to size and must not be refused for a limit it does not need
    max_requests = _span_slots()
    from sglang.srt.afd.layer_kinds import layer_types_of, linear_widths
    from sglang.srt.afd.linear_state import LinearStates
    from sglang.srt.afd.span import SpanRunner, group_layers

    config = model.config
    layer_types = layer_types_of(model)
    widths = linear_widths(config)
    states = LinearStates(
        slots=max_requests,
        num_v_heads=widths["v_heads"],
        head_k_dim=widths["dk"],
        head_v_dim=widths["dv"],
        device=device,
    )
    runner = SpanRunner(
        model,
        states,
        layer_types=layer_types,
        query_shift=_span_query_shift(),
    )
    spans = group_layers(layer_types)
    logger.info(
        # NOT "held here". The history is the host's and this side reaches it by callback
        # (`afd/history_service.py`); `SpanRunner` touches its slot table to release and to
        # report, never to read or advance a recurrence. The line used to say the opposite and
        # it is the reason a reader of this log concludes the pool is stateful when it is not.
        "afd pool: the group cut. %s span(s) a decode step against %s per-layer calls, %s "
        "layer(s) served here entire. The recurrent state is the HOST's, reached by callback "
        "mid-span; the slot table here is sized for up to %s request(s). "
        "The batch riding a span is fixed for its whole length: every stage in it has "
        "context-free latency, so there is nothing inside a span worth re-forming a batch for.",
        len(spans),
        len(layer_types) - 1,
        sum(len(s) for s in spans) - len([s for s in spans if s[0] >= 0]),
        max_requests,
    )
    return runner


def install_span_routing(model, client: PoolClient, *, reply_timeout_s: float = 60.0):
    """Give whole groups of layers to the pool, keeping only the attentions here.

    The cut's own overlap needs no outside schedule: the sweep sits between
    `collect_read_point` and `collect_kv`, in the gap between the two halves of a span's
    reply. The per-layer schedule hook that used to be threaded through here left with
    the arrangement it belonged to.
    """
    from sglang.srt.afd.history_service import HistoryService
    from sglang.srt.afd.layer_kinds import layer_types_of
    from sglang.srt.afd.linear_history import HistoryCache
    from sglang.srt.afd.span_routing import SpanClient, SpanRouting

    types = layer_types_of(model)
    routing = SpanRouting(
        model,
        SpanClient(client, reply_timeout_s=reply_timeout_s),
        types,
        # the same single resolution the pool reads: at shift 1 the routing attaches each span's
        # convolution windows, because the pool convolves and the ring lives on this side
        query_shift=_span_query_shift(),
    )

    # the history the pool calls back for. It lives here because it is the request's, and
    # because putting it here is what lets the pool stay out of a request's business between
    # that request's calls -- see linear_history.HistoryCache for what the round trips cost.
    from sglang.srt.afd.layer_kinds import linear_widths

    config = model.config
    widths = linear_widths(config)
    cache = HistoryCache(
        slots=_span_slots(),
        layers=len(types),
        value_heads=widths["v_heads"],
        head_k_dim=widths["dk"],
        head_v_dim=widths["dv"],
        # The ring is sized for real when this host runs the convolution. It is the last
        # per-request thing the pool held, and while it was there a request was STICKY to the pool
        # that served its previous call -- another pool would convolve against four steps of
        # someone else's history and answer fluently. 80 KiB a layer a request here, against the
        # recurrent state's 3.00 MiB.
        conv_width=_conv_width(config),
        conv_taps=widths["conv_taps"],
        device=next(model.parameters()).device,
    )
    # the pool sends rows in the order this host sent them, so the ids are the ones the routing
    # is holding for the span in flight. Read through the routing rather than captured, because a
    # captured list would be the FIRST span's rows for every span after it.
    routing.history = HistoryService(
        cache,
        rows_of=lambda frame: routing.current_rows(),
        # The convolution's own weight, which this host has to hold for the arrangement to work:
        # 80 KiB a layer, 3.75 MiB for all 48 on this model, measured from the checkpoint. The
        # layers are indexed as the pool names them, so this reads the model's own module.
        conv_weight=lambda layer: model.model.layers[
            layer
        ].linear_attn.conv1d.weight.squeeze(1),
        dims=(
            widths["k_heads"],
            widths["v_heads"],
            widths["dk"],
            widths["dv"],
        ),
    )
    client.serve = routing.history
    from sglang.srt.afd.lane import the_lane

    lane = the_lane()
    if lane is not None and _span_query_shift():
        # the composition root brought the pair up when the adopted configuration said
        # nccl; the read TRIANGLE that rides it is the early read's own protocol, so it
        # is installed only at shift 1 -- at shift 0 nothing's exchange order leaves the
        # wire and the lane idles, later and not wrong
        from sglang.srt.afd_query_shift.nccl_lane import serve_triangle

        serve_triangle(lane, routing.history)
    _install_head_shim(model, routing)
    _pull_residual_weights(model, client)
    logger.info(
        "afd host: holding %s linear layer(s) of recurrent state for up to %s request(s), %s",
        sum(1 for t in types if t != "full_attention"),
        cache.slots,
        cache.report(),
    )
    # after the routing, because what is safe to release is exactly what it routed
    from sglang.srt.afd.absent_projections import strip_routed_weights
    from sglang.srt.runtime_context import get_server_args

    # Read ONCE, here, and passed down. This function is the composition root for the cut; the
    # things below it are called by tests that have no global to read.
    remote_embedding = remote_embedding_wanted(get_server_args())
    if remote_embedding:
        from sglang.srt.afd.remote_embedding import (
            PendingIds,
            send_ids_instead,
        )

        routing.pending_ids = PendingIds()
        send_ids_instead(model, routing.pending_ids)
    routing.released = strip_routed_weights(
        model, routing, remote_embedding=remote_embedding
    )
    _refuse_absent_convolutions(model, routing)
    _refuse_a_mismatched_embedding(model)
    logger.info("afd host: the group cut is installed. %s", routing.report())
    return routing


def _refuse_absent_convolutions(model, routing) -> None:
    """With the rings here, every passenger layer's convolution weight must still be real.

    Checked at STARTUP rather than left to the first frame. A weight that stayed on the meta
    device raises `NotImplementedError: Cannot copy out of meta tensor` inside the client's
    receive thread, the host then answers nothing, and the pool reports a 30 s timeout naming the
    far end -- three hops from the cause, and the arrangement looks alive the whole time. That is
    exactly what it did the first time this flag was turned on.

    Reports the layers and how many bytes are actually held, because "the check passed" and "the
    weights are there" are the same sentence only when the second is a number.
    """
    absent, held = [], 0
    for index in sorted(routing.passengers):
        weight = model.model.layers[index].linear_attn.conv1d.weight
        if weight.is_meta:
            absent.append(index)
        else:
            held += weight.numel() * weight.element_size()
    if absent:
        raise RuntimeError(
            f"{len(absent)} passenger layer(s) have no convolution "
            f"weight left on this host: {absent[:8]}{' ...' if len(absent) > 8 else ''}. Under "
            f"the group cut the ring lives here unconditionally, and the weight that filters it "
            f"does not, so a MIX frame would raise in "
            f"the receive thread and the pool would time out blaming this end. Either the loader "
            f"built these layers on meta before `release` could keep anything, or the keep list "
            f"in `absent_projections` no longer names the convolution."
        )
    logger.info(
        "afd host: the rings are here -- %s convolution weight(s) kept, %.2f MiB",
        len(routing.passengers),
        held / 1024 / 1024,
    )


def _refuse_a_mismatched_embedding(model) -> None:
    """The embedding must be absent exactly when the pool does the lookup. Both directions.

    One direction alone is half a check. If the weight stays resident while the pool embeds, the
    server works perfectly and quietly wastes 2.368 GiB -- a failure whose only symptom is a number
    nobody reads. If the weight is gone while this host is supposed to embed, the first token dies
    on a meta tensor inside the model's own forward, with a message about devices and nothing about
    AFD.

    Neither is worth debugging twice, so both are refused at startup with the flag named. Not
    behind `if remote:` for the same reason -- the off case is the one that fails loudly and late.
    """
    from sglang.srt.runtime_context import get_server_args

    remote = remote_embedding_wanted(get_server_args())
    weight = model.model.embed_tokens.weight
    held = (
        0.0 if weight.is_meta else weight.numel() * weight.element_size() / 1024 / 1024
    )
    if remote and not weight.is_meta:
        raise RuntimeError(
            f"the remote embedding, but this host is still holding embed_tokens "
            f"({held:.0f} MiB). The pool does the lookup, so nothing here reads the weight and it "
            f"is memory the KV cache could have had. Either `strip_routed_weights` no longer "
            f"releases it, or something rebuilt it after the routing was installed."
        )
    if not remote and weight.is_meta:
        raise RuntimeError(
            "embed_tokens is on the meta device and the remote embedding is off, so this host "
            "is expected to do the lookup with a weight that has no data. The first token would "
            "die inside the model's own forward, with a message about devices that names neither "
            "this flag nor AFD."
        )
    logger.info(
        "afd host: the embedding is %s -- %s",
        "on the pool" if remote else "here",
        "released" if remote else f"{held:.0f} MiB resident",
    )


class SpanArm:
    """What the composition root needs from an arm, and nothing else.

    Four entry points now, two a side. The root holds one of these or None; it never asks which
    arm it is, and it never imports the module that answers.

        check_args       this arm's own flags, refused before a model loads. They used to be
                         checked in `arg_groups/afd_hook.py`, which had to import this package by
                         name to do it -- the one dependency direction that stops AFD shipping
                         without this arm
        transform_model  what the model runner, a FROZEN file, used to do by importing
                         `afd_query_shift.wiring` inside a try and reading three of this arm's
                         server args off the runner
        install_on_host / make_pool_runner    the two ends of the cut itself
    """

    name = "span"

    def wanted(self) -> bool:
        return span_cut_wanted()

    @staticmethod
    def absent_classes(server_args) -> tuple:
        """The linear attention, built with no storage at all, under the group cut.

        Every linear-attention layer is a PASSENGER of a span: the pool runs the whole layer and
        the host's forward is a pass-through. So `Qwen3_5GatedDeltaNet` is a class this host never
        multiplies by anything, all of it, which is exactly what the fifth entry point requires --
        the loader wraps a class rather than its instances, so a class with even one instance the
        host still uses cannot be named.

        The attention projections cannot come this way and are released after routing instead:
        they are QKVParallelLinear and RowParallelLinear, shared with the vision tower and with
        every other projection in the model. This one is not shared with anything.

        Naming it here rather than releasing it later is worth the separate mechanism because it
        moves the memory from 'taken and given back' to 'never taken', and the construction peak
        is what fails on a card smaller than the checkpoint.
        """
        if not span_cut_wanted_for(server_args):
            return ()
        try:
            from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
        except ImportError:  # a build without this family
            return ()
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        # the language-model head too: the pool computes every request's last-row
        # logits at the exit span, so this host never multiplies by it -- 2.37 GiB
        # never allocated, which on a small card is tens of thousands of KV tokens
        return (Qwen3_5GatedDeltaNet, ParallelLMHead)

    @staticmethod
    def arrangement_word(server_args) -> float:
        """The settings this arm needs BOTH ends given, folded into one number.

        The RESOLVED read point, not the requested one. The checkpoint decides it and the
        command line only warns, so two ends given the same checkpoint agree whatever either was
        asked for -- and two ends given different checkpoints disagree here rather than at the
        first token. Encoding the request instead would have let an override on one end read as
        agreement while the two served at different read points.

        Encoded rather than sent as a struct because the shared half moves one number and does not
        know what is in it. Shift is 0 or 1 under this cut (`_span_query_shift` refuses anything
        else), so two values fit in the low bits with room to spare.
        """
        from sglang.srt.afd.checkpoint import resolved_shift

        if not span_cut_wanted_for(server_args):
            return 0.0
        # +1 so that "span cut, shift 0, no rings" is distinguishable from "no arm at all"
        return float(
            1
            + 2 * int(resolved_shift(server_args))
            + 8 * int(remote_embedding_wanted(server_args))
        )

    @staticmethod
    def explain_arrangement(server_args, mine: float, theirs: float) -> str:
        """Name the setting that differs, so the message is about the configuration."""
        if not span_cut_wanted_for(server_args):
            return ""

        def unpack(word):
            n = int(word)
            if n == 0:
                return None
            rest = n - 1
            return (rest // 2) % 2, (rest // 4) % 2, (rest // 8) % 2

        here, there = unpack(mine), unpack(theirs)
        if there is None:
            return "The pool is not serving the span cut and this host expects it."
        if here is None:
            return "This host is not serving the span cut and the pool is."
        parts = []
        if here[0] != there[0]:
            parts.append(
                f"the resolved query read point is {here[0]} here and {there[0]} on the "
                f"pool. The POOL's resolution decides where the query is read; two ends "
                f"resolving differently were given different checkpoints or a derived "
                f"override on one side only."
            )
        if here[2] != there[2]:
            parts.append(
                f"the embedding lives on the pool for {'this host' if here[2] else 'the pool'} "
                f"and not for the other end. The host sends token ids only when it has no "
                f"embedding of its own; a pool expecting hidden states would embed a tensor "
                f"of ids, or read ids as floats."
            )
        if here[1] != there[1]:
            parts.append(
                "the two ends disagree about which side holds the convolution rings. The "
                "host builds the rings and the pool decides whether to send OP_STATE_MIX; "
                "a pool sending it to a host without rings is refused at the frame."
            )
        return " ".join(parts)

    @staticmethod
    def kept_parameters(server_args) -> dict:
        """The convolution, kept out of the class this arm otherwise declares absent.

        Under the group cut UNCONDITIONALLY, because on this branch the ring is unconditionally
        here: `install_span_routing` sizes it with `_conv_width(config)` with no flag in front of
        it, and `mix_host` is always installed. The pool holds nothing per request, full stop --
        that is what this branch settled, and a weight that only survived when a flag was set
        would be absent in the configuration everything actually runs in.

        Tying this to a rings flag was exactly that mistake, made while porting one
        back from the deployment branch: the flag defaults false, the ring is built anyway, and
        the weight that filters it would have been stripped. The failure is not an error either --
        it is `NotImplementedError: Cannot copy out of meta tensor` inside the receive thread,
        surfacing as a 30 s timeout blaming the far end.

        3.75 MiB for all 48 layers on this model, 0.035% of what a layer weighs.
        """
        if not span_cut_wanted_for(server_args):
            return {}
        try:
            from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
        except ImportError:  # a build without this family
            return {}
        return {Qwen3_5GatedDeltaNet: ("conv1d.weight", "conv1d.bias")}

    @staticmethod
    def transform_model(*, model, model_config, server_args):
        """Resolve the read point this process serves at. Always returns None.

        The span implements the read point itself, so nothing on the model is rewired
        here -- the per-layer conversion machinery this used to drive is gone with the
        arrangements it served. What remains is the RESOLUTION: the one place the
        checkpoint has its say, recorded before either end serves a frame, because
        resolving after the zeroing once recorded 0 on a run serving 1 and the two ends
        then disagreed at the first frame.
        """
        from sglang.srt.afd.checkpoint import (
            requested_shift,
            resolve_shift,
        )

        resolve_shift(requested_shift(server_args), model_config.hf_config)
        return None

    def install_on_host(self, model, client):
        from sglang.srt.afd.arms import arrangement_word
        from sglang.srt.runtime_context import get_server_args

        # the arrangement word goes with the capability check: both ask "is the far end the thing
        # this host was configured against", once, before the first frame
        client.require(
            PoolClient.NEEDS_SPANS,
            arrangement=arrangement_word(get_server_args()),
        )
        return install_span_routing(model, client)

    def make_pool_runner(self, model, *, device):
        return make_span_runner(model, device=device)


def _host_device(model):
    """The device this host computes on: any parameter that has real storage."""
    for p in model.parameters():
        if not p.is_meta:
            return p.device
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pull_residual_weights(model, client) -> None:
    """Every weight byte this host multiplies by, fetched from the pool once.

    The host's loader runs dummy -- it reads no weight file -- and the few tensors
    its own arithmetic touches (the convolution filters, the final norm) arrive
    here, in the fixed layer order both ends derive from the same config. Shapes
    are checked by name; a mismatch is two ends built from different checkpoints,
    refused before the first frame rather than served as fluent noise.
    """
    from sglang.srt.afd.layer_kinds import layer_types_of
    from sglang.srt.afd.protocol import OP_WEIGHTS

    import torch

    handle = client.issue_frame(0, 0, (torch.zeros(1, 1),), OP_WEIGHTS)
    got = client.collect_frame(handle, "cpu")
    types = layer_types_of(model)
    linear = [i for i, k in enumerate(types) if k != "full_attention"]
    expected = 2 * len(linear) + 1
    if len(got) != expected:
        raise RuntimeError(
            f"the pool pushed {len(got)} residual weight tensor(s) where "
            f"{expected} were expected ({len(linear)} conv filters with biases and "
            f"the final norm). The two ends were built from different checkpoints."
        )
    device = _host_device(model)

    def _land(module, name, value, index):
        target = getattr(module, name)
        if value.numel() != target.numel():
            raise RuntimeError(
                f"{index}: the pool pushed {value.numel()} value(s) and this "
                f"host built {tuple(target.shape)}. The two ends were built "
                f"from different checkpoints."
            )
        value = value.reshape(target.shape)
        if target.is_meta:
            # the module was deliberately built with no storage; the pushed
            # tensor IS its storage now
            module._parameters[name] = torch.nn.Parameter(
                value.to(device, target.dtype), requires_grad=False
            )
        else:
            with torch.no_grad():
                target.copy_(value.to(target.device, target.dtype))

    for j, index in enumerate(linear):
        conv = model.model.layers[index].linear_attn.conv1d
        w, b = got[2 * j], got[2 * j + 1]
        _land(conv, "weight", w, f"layer {index} convolution filter")
        if conv.bias is not None and b.numel():
            _land(conv, "bias", b, f"layer {index} convolution bias")
    _land(model.model.norm, "weight", got[-1], "the final norm")
    logger.info(
        "afd host: %d residual weight tensor(s) adopted from the pool -- this host "
        "read no weight file",
        expected,
    )


def _install_head_shim(model, routing) -> None:
    """The head lives with the weights: the pool's logits replace the host's GEMM.

    The routing stashes the pool-delivered PRE-softmax last-row logits for the step
    in flight, and this wraps the logits processor's one GEMM so the stashed rows
    are returned instead of multiplying a head this host does not hold. Softmax and
    every sampling knob run downstream exactly as before, on exactly the tensor they
    always ran on. The vocabulary width comes from the CONFIG, because the weight it
    used to be read from is on the meta device; a path that reaches the GEMM with
    nothing stashed and a meta head is refused by name rather than left to die
    inside a matmul about devices.
    """
    lp = getattr(model, "logits_processor", None)
    config = getattr(model, "config", None)
    text = getattr(config, "text_config", None) or config
    vocab = getattr(text, "vocab_size", None)
    if lp is None or vocab is None:
        routing._vocab = None
        return
    routing._vocab = int(vocab)
    routing._head_on_pool = True
    original = lp._compute_lm_head

    def compute(hidden_states, lm_head, embedding_bias=None):
        got = routing.take_pool_logits()
        if got is not None and got.shape[0] == hidden_states.shape[0]:
            return got
        weight = getattr(lm_head, "weight", None)
        if weight is not None and getattr(weight, "is_meta", False):
            raise RuntimeError(
                f"a logits read of {hidden_states.shape[0]} row(s) reached the head "
                f"and this host holds none -- the pool computes each request's "
                f"last-row logits, and "
                f"{'a mismatched batch arrived' if got is not None else 'nothing was stashed'}. "
                f"Paths that need logits at other positions (prompt logprobs) are "
                f"not served by a weightless host."
            )
        return original(hidden_states, lm_head, embedding_bias)

    lp._compute_lm_head = compute


register(SpanArm.name, SpanArm)
