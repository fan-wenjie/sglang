"""Installing the query-shift arm, on either end, and announcing it to AFD.

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

import torch

from sglang.srt.afd.arms import register
from sglang.srt.afd.layer_kinds import layer_types_of
from sglang.srt.afd.linear_history import HistoryCache
from sglang.srt.afd.pool_client import PoolClient

logger = logging.getLogger(__name__)


def span_cut_wanted() -> bool:
    """Whether this process was launched for the group cut.

    Read from the global server args rather than passed down, for the reason `absent_ffn` reads it
    the same way: the pool role is set up inside the scheduler process, which is spawned, and a
    module-level flag set in the parent does not cross that boundary. That failure was silent once
    already here -- the feed-forward weights were built after the flag said not to -- so the value
    is fetched where it is used.
    """
    from sglang.srt.server_args import get_global_server_args

    return bool(get_global_server_args().afd_span_cut)


def _span_slots() -> int:
    """How many requests the pool can hold recurrent state for.

    Taken from `--max-running-requests` rather than guessed. A slot table that ran out would refuse
    a request mid-generation, and a number invented here would put that cliff somewhere the
    operator never chose. If sglang has not resolved the limit yet there is nothing to derive from,
    and refusing at startup beats picking one.
    """
    from sglang.srt.server_args import get_global_server_args

    limit = get_global_server_args().max_running_requests
    if limit is None:
        raise ValueError(
            "--afd-span-cut needs --max-running-requests to size the pool's recurrent state "
            "table. A recurrent state is the whole history compressed and cannot be rebuilt from "
            "a prefix, so the table refuses rather than evicts -- and where that refusal falls "
            "has to be a number somebody chose."
        )
    return int(limit)


def _span_query_shift() -> int:
    """How far back this pool reads the next group's query. Read HERE, on the pool.

    The knob is `--afd-query-shift-layers` and it looks like a host setting, because on the
    per-layer cut it was one. Under the group cut the query projection moved to the pool with
    `W_q`, so the pool is the side that decides where the query is read from and the host's copy
    of the flag decides nothing at all.

    That was not wired, and the span read the shifted point unconditionally. The cost was not a
    wrong model -- shift 1 is the operating point either way -- it was a lost control: a run
    launched with `--afd-query-shift-layers 0` to ask what the shift costs got a shifted read
    anyway, reported no difference, and the difference had never been asked for. Both ends must be
    given the same value until the protocol carries it.
    """
    from sglang.srt.server_args import get_global_server_args

    shift = get_global_server_args().afd_query_shift_layers
    if shift not in (0, 1):
        raise ValueError(
            f"--afd-query-shift-layers={shift} under the group cut. This cut reads the query from "
            f"between the last linear attention and the last feed-forward of a group, which is "
            f"shift 1, or from the group's output, which is shift 0. A deeper shift is a "
            f"different cut and would have to move the read point across a group boundary."
        )
    return int(shift)


def make_span_runner(model, *, device):
    """The pool's span runner, if --afd-span-cut asked for one. None otherwise.

    Sized by `max_requests` because a recurrent state cannot be evicted and rebuilt from a prefix
    the way a KV cache can -- it is the whole history compressed -- so the slot table refuses a
    request rather than dropping one, and the refusal has to be far from the working point.
    """
    if not span_cut_wanted():
        return None
    # after the switch, never before it: a pool serving the per-layer cut has no recurrent state
    # to size and must not be refused for a limit it does not need
    max_requests = _span_slots()
    from sglang.srt.afd.linear_state import LinearStates
    from sglang.srt.afd.layer_kinds import layer_types_of
    from sglang.srt.afd_query_shift.span import SpanRunner, group_layers

    config = model.config
    layer_types = layer_types_of(model)
    states = LinearStates(
        slots=max_requests,
        num_v_heads=config.linear_num_value_heads,
        head_k_dim=config.linear_key_head_dim,
        head_v_dim=config.linear_value_head_dim,
        device=device,
    )
    from sglang.srt.afd_query_shift.selfcheck import watch_colocated_residual

    watch_colocated_residual(model)
    runner = SpanRunner(
        model, states, layer_types=layer_types, query_shift=_span_query_shift())
    from sglang.srt.afd_query_shift.selfcheck import watch_linear_attention

    watch_linear_attention(model, runner)
    spans = group_layers(layer_types)
    logger.info(
        "afd pool: the group cut. %s span(s) a decode step against %s per-layer calls, %s "
        "layer(s) served here entire, both recurrent states held here for up to %s request(s). "
        "The batch riding a span is fixed for its whole length: every stage in it has "
        "context-free latency, so there is nothing inside a span worth re-forming a batch for.",
        len(spans), len(layer_types) - 1,
        sum(len(s) for s in spans) - len([s for s in spans if s[0] >= 0]),
        max_requests,
    )
    return runner


def install_span_routing(model, client: PoolClient, *, sweep_ahead,
                         reply_timeout_s: float = 60.0):
    """Give whole groups of layers to the pool, keeping only the attentions here.

    `sweep_ahead` is accepted and NOT yet used, and that is deliberate rather than forgotten. The
    window under this cut sits between the two halves of a span's reply -- the pool hands over the
    query's source one feed-forward before the span's output, and the host should spend that
    feed-forward projecting the query and sweeping the cache. It does not yet; it collects both
    halves back to back. `SpanRouting.report()["sweep_window_open"]` says so, so a timing taken
    now cannot be quoted as this arrangement's without the report contradicting it.
    """
    from sglang.srt.afd.history_service import HistoryService
    from sglang.srt.afd.linear_history import HistoryCache
    from sglang.srt.afd.layer_kinds import layer_types_of
    from sglang.srt.afd_query_shift.span_routing import SpanClient, SpanRouting

    types = layer_types_of(model)
    routing = SpanRouting(
        model, SpanClient(client, reply_timeout_s=reply_timeout_s), types)

    # the history the pool calls back for. It lives here because it is the request's, and
    # because putting it here is what lets the pool stay out of a request's business between
    # that request's calls -- see linear_history.HistoryCache for what the round trips cost.
    config = model.config
    cache = HistoryCache(
        slots=_span_slots(),
        layers=len(types),
        value_heads=config.linear_num_value_heads,
        head_k_dim=config.linear_key_head_dim,
        head_v_dim=config.linear_value_head_dim,
        conv_width=1, conv_taps=1,        # the ring stays on the pool; this holds no convolution
        device=next(model.parameters()).device,
    )
    # the pool sends rows in the order this host sent them, so the ids are the ones the routing
    # is holding for the span in flight. Read through the routing rather than captured, because a
    # captured list would be the FIRST span's rows for every span after it.
    routing.history = HistoryService(cache, rows_of=lambda frame: routing.current_rows())
    client.serve = routing.history
    logger.info(
        "afd host: holding %s linear layer(s) of recurrent state for up to %s request(s), %s",
        sum(1 for t in types if t != "full_attention"), cache.slots, cache.report(),
    )
    # after the routing, because what is safe to release is exactly what it routed
    from sglang.srt.afd_query_shift.absent_projections import strip_routed_weights

    routing.released = strip_routed_weights(model, routing)
    logger.info("afd host: the group cut is installed. %s", routing.report())
    if sweep_ahead is not None:
        # it would otherwise fire from the per-layer prepare_mlp hook, on layers that no longer
        # run here at all -- a sweep launched for a feed-forward this host never issues
        sweep_ahead.routed_layers = set(routing.passengers) | set(routing.heads)
    return routing


class QueryShiftArm:
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

    name = "query-shift"

    def wanted(self) -> bool:
        return span_cut_wanted()

    @staticmethod
    def check_args(server_args) -> None:
        """This arm's flags, refused at startup rather than ignored at the first token."""
        from sglang.srt.afd_query_shift.arg_checks import check as check_arm_args

        check_arm_args(server_args)

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
        if not server_args.afd_span_cut:
            return ()
        try:
            from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
        except ImportError:                            # a build without this family
            return ()
        return (Qwen3_5GatedDeltaNet,)

    @staticmethod
    def transform_model(*, model, model_config, server_args):
        """Move the read point on the loaded model. None when this arm was not asked for.

        Returns the wiring, which carries `sweep_ahead` for the host router: the window opens
        between issuing a feed-forward and collecting it, and the schedule that fires in that gap
        is this arm's, so this arm is what hands it over.
        """
        from sglang.srt.afd_query_shift.wiring import install_early_q

        wiring = install_early_q(
            model=model,
            shift_layers=server_args.afd_query_shift_layers,
            coverage=server_args.afd_coverage,
            span_cut=server_args.afd_span_cut,
            hf_config=model_config.hf_config,
            split_attention=server_args.afd_split_attention,
            verify_split=server_args.afd_verify_split,
        )
        # the shared side asks for one named attribute rather than reaching through this arm's own
        # structure, so nothing outside knows the wiring holds a `hooks`
        wiring.sweep_ahead = wiring.hooks.sweep_ahead
        return wiring

    def install_on_host(self, model, client, *, sweep_ahead):
        client.require(PoolClient.NEEDS_SPANS)
        return install_span_routing(model, client, sweep_ahead=sweep_ahead)

    def make_pool_runner(self, model, *, device):
        return make_span_runner(model, device=device)


register(QueryShiftArm.name, QueryShiftArm)


# The ladder's rungs live beside this one and register the same way. Imported here because
# `arms.load()` looks for `<package>.installer` and nothing else -- a rung in its own module would
# announce itself to nobody, which is the failure the registry was built to make impossible.
from sglang.srt.afd_query_shift import rung2  # noqa: E402,F401 -- imported for its registration
