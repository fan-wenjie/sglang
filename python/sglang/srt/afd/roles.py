"""The two roles, as processes: a host that owns the cache and a pool that owns the weights.

    pool   loads the stack and answers `mlp(hidden)` for whichever layer a caller names
    host   loads the stack, installs whatever an arm asks for (nothing, when none is installed),
           and sends every converted layer's feed-forward to the pool instead of running it
           locally

## What this buys, and where

The feed-forward genuinely leaves the host: the pool runs it, batches across callers, and the
host's own MLP weights go unused on the converted layers. That is the arrangement's plumbing, and
it is what a second machine needs.

The OVERLAP is `sweep_ahead`, passed in here as `between`. `self.attn(q, k, v, forward_batch)`
used to be one call -- the sweep over the cache and the fold-in of this step's token inside one
kernel -- so there was nothing the host could do between issuing a feed-forward and needing its
answer.
here, between the issue and the collect.

The window exists at a layer j only when layer j+N sweeps a cache. On Qwen3.8-27B that is 16 of
64 layers, because three of every four are linear attention, whose query multiplies a recurrent
state this does not know how to partition. The other 48 feed-forwards are still issued and waited
for. That is a property of the model's layer mix, and `SweepAhead.record()` reports the count so
a speedup is read against the number of windows that actually existed.

## Why the pool loads the whole model

It uses one layer's MLP at a time and could hold only those weights. Loading the stack is wasteful
and correct, and correctness first: a pool that loaded a subset would need its own weight-mapping
code, and a mapping bug there produces plausible tokens from the wrong weights.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
from collections.abc import Callable

import torch

from sglang.srt.afd.arms import arrangement_word
from sglang.srt.afd.pool_client import PoolClient, PoolClosed
from sglang.srt.afd.pool_server import serve
from sglang.srt.afd.pushed_config import (
    adopt_from_the_pool,
    encode_config,
    pool_config,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_disagg, get_model, get_server_args

logger = logging.getLogger(__name__)

# how often the host says whether its windows are opening. Often enough to see a schedule that
# stopped overlapping mid-run, rarely enough that the line is not the workload.
REPORT_EVERY = 512


def make_pool_forward(model) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """The pool's service: one layer's feed-forward, for a batch of tokens from any caller.

    Each layer's feed-forward is bound HERE, at construction, and never looked up again. When the
    two roles share one process -- which they do in the single-machine test -- the host's router
    replaces `layer.mlp.forward`, and a pool that looked the method up per call would find the
    router, send the work back out over the socket, and wait for itself. The symptom is a hang
    with an idle GPU and no error, and it is invisible on two machines, which is exactly when it
    would be found late.
    """
    layers = model.model.layers
    bound = [layer.mlp.forward for layer in layers]

    def forward(batch: torch.Tensor, layer: int) -> torch.Tensor:
        if not 0 <= layer < len(bound):
            raise RuntimeError(
                f"a caller asked for layer {layer} of a {len(bound)}-layer stack; the two sides "
                f"are not running the same model"
            )
        with torch.no_grad():
            out = bound[layer](batch)
        return out[0] if isinstance(out, tuple) else out

    return forward


def serve_pool_in_background(
    model, port: int, min_batch: int, max_wait_ms: int, device
):
    """Answer feed-forward frames on a thread of an already-loaded server.

    The pool role reuses the ordinary launch path: the model is loaded, the scheduler starts, and
    nobody sends it a request. That is wasteful of a process and honest about what it is -- the
    alternative is a second entrypoint whose weight loading, quantisation and device placement
    would be a second implementation of the thing whose numbers must match the host's.
    """
    ready = threading.Event()
    thread = threading.Thread(
        target=run_pool,
        kwargs=dict(
            model=model,
            host="0.0.0.0",
            port=port,
            min_batch=min_batch,
            max_wait_ms=max_wait_ms,
            device=device,
            ready=ready,
        ),
        daemon=True,
        name="afd-pool",
    )
    thread.start()
    if not ready.wait(timeout=60):
        raise RuntimeError(
            f"the afd pool did not bind port {port} within 60s. A host pointed at it would fail "
            f"at its first token instead, an hour into a benchmark."
        )
    return thread


def _arm_if_wanted():
    """A derived arm, if one is registered AND the operator asked for it. None otherwise.

    None is the ordinary state. AFD ships without any arm registered, and this root does not know
    what arms exist -- it holds the object or it does not, and every branch below reads
    `if arm is not None` rather than naming an arrangement.
    """
    from sglang.srt.afd.arms import available, load, resolve

    # IN THIS PROCESS. The registry is a module-level dict, and the pool and host roles are set up
    # inside the scheduler process, which sglang SPAWNS -- so an import done by the argument check
    # in the parent registers nothing here. That is the exact boundary `span_cut_wanted` was
    # written to warn about, in this file, and the registry walked into it anyway: the arm
    # reported itself available at startup, was never installed, and the arrangement served the
    # standard path while every log line said the arm was there.
    load()
    for name in available():
        factory = resolve(name)
        arm = factory()
        if arm.wanted():
            return arm
    return None


def install_host_routing(model, pool_addr: str, connect_timeout_s: float = 30.0):
    """Point every routable layer's feed-forward at the pool.

    Returning (None, None) for an empty address rather than raising: the caller treats
    None as "no pool" and the router is simply not installed. The refusal for a host that names no pool at all belongs at
    startup, in `arg_groups/afd_hook.py`, where it can say so before a model loads.
    """
    if not pool_addr:
        logger.info(
            "afd host: no weights pool. The feed-forward stays on this machine and only the "
            "sweep travels, which is the reversed arrangement."
        )
        return None, None
    client = PoolClient(pool_addr, connect_timeout_s)
    # The pool's word arrives FIRST, before the arm is even chosen: which arm a host wants can
    # depend on settings the host no longer configures. Normally the LOADER already adopted --
    # which classes to build without storage is itself configuration -- and this is the no-op
    # replay that keeps the install path correct for a caller that skipped the loader.
    adopt_from_the_pool()
    from sglang.srt.afd.pushed_config import adopted_transfer

    arm = _arm_if_wanted()
    if arm is None:
        # ONE arrangement. The submission serves the span cut -- a stateless big card, a
        # weightless small card -- and a model no installed arm speaks for is refused by
        # name rather than served through some other line drawn through it.
        from sglang.srt.afd.arms import available

        raise ValueError(
            f"no installed arm serves {get_model().model_path!r} as an AFD host. "
            f"The arrangement is the span cut, chosen by the checkpoint's own layer "
            f"types; available arms: {available() or '(none registered)'}."
        )
    # The arm asks for itself by name at the HELLO. Two arrangements differ in what each END
    # holds, not only in what they say to each other: a host built for one of them, run
    # against a pool built for the other, would not fail -- it would run out of layers to ask
    # for. The capability bit is what turns that into a startup error.
    installed = arm.install_on_host(model, client)
    # AFTER the HELLO, never before. The HELLO is where this host CLAIMS its lane slot, and the
    # pool arms a slot when it hears the claim -- so a bring-up started first parks on a
    # rendezvous nobody has bound yet, `ready` never flips, and the arrangement stays on the TCP
    # wire with no line to say why. Slot 0 hid this: the pool pre-arms that one, so the single-host
    # deployment paired anyway and the ordering looked correct for as long as there was one host.
    _bring_up_the_lane(model, pool_addr)
    return client, installed


def _bring_up_the_lane(model, pool_addr: str) -> None:
    """This host's half of its own slot's pairing. A no-op unless the transport is the lane."""
    from sglang.srt.afd.pushed_config import adopted_transfer

    if adopted_transfer() != "nccl":
        return
    from sglang.srt.afd.installer import _host_device
    from sglang.srt.afd.lane import LANE_PORT_OFFSET, lane_port, lane_up
    from sglang.srt.runtime_context import get_disagg

    ip, _, pool_port = pool_addr.rpartition(":")
    slot = int(get_disagg().afd_host_lane or 0)
    port = lane_port(int(pool_port) + LANE_PORT_OFFSET, slot)
    logger.info("afd host: claiming lane slot %s, rendezvous at %s:%s", slot, ip, port)
    # not `next(model.parameters())`: a skeleton host's first parameter is the embedding
    # stub's meta placeholder, and a lane on meta never pairs
    lane_up("host", ip, port, _host_device(model), slot=slot)


_LANE_BASE = None


def _remember_lane_base(base: int, device) -> None:
    """What a later slot needs to arm itself, kept where the HELLO can reach it.

    The HELLO handler is inside the frame server and has neither the bootstrap port nor the
    device in hand; both are settled here, once, at pool start.
    """
    global _LANE_BASE
    _LANE_BASE = (base, device)


def arm_lane_slot(slot: int):
    """Bring up the pool's half of one slot, on demand. Returns the lane, or None.

    Called from the HELLO when a host claims a slot this pool has not armed. None whenever the
    lane is not this pool's transport at all, which is how a tcp pool ignores a claim rather
    than refusing a host that would have worked without it.
    """
    if _LANE_BASE is None:
        return None
    base, device = _LANE_BASE
    from sglang.srt.afd.lane import lane_port, lane_up

    return lane_up("pool", "0.0.0.0", lane_port(base, int(slot)), device, slot=int(slot))


def _pool_runner(model, *, device):
    """What this pool serves beyond the feed-forward, or None.

    An arm's decision alone: an arrangement that serves linear layers brings its own runner
    (and imports `pool_linear`, which is what claims OP_LAYER in the departure table). The
    `--afd-pool-linear` flag that once sat beside the arms is gone: its host half was never
    wired to answer the state callbacks on this line -- the first MIX died by name -- and
    both of its jobs have a measured better home (the span cut for serving, the
    linear-on-pool arm for the per-layer diagnostic).
    """
    arm = _arm_if_wanted()
    if arm is not None:
        return arm.make_pool_runner(model, device=device)
    return None


def run_pool(
    model,
    host: str,
    port: int,
    min_batch: int,
    max_wait_ms: int,
    device: torch.device | str,
    ready: threading.Event | None = None,
):
    """Serve until killed. Blocks."""
    logger.info(
        "afd pool: %s layers, min_batch=%s, max_wait=%sms",
        len(model.model.layers),
        min_batch,
        max_wait_ms,
    )
    if (get_disagg().afd_transfer_backend or "nccl") == "nccl":
        # rank 0 of the lane's pair; the rendezvous listens beside the bootstrap port and
        # the choice reaches the host with the rest of the pushed configuration
        from sglang.srt.afd.lane import LANE_PORT_OFFSET

        # NOT armed here. Every slot, slot 0 included, is armed when a host CLAIMS it at the
        # HELLO. Arming slot 0 at pool start was the obvious thing and it is the one that did
        # not work: the eagerly-armed slot reported `EADDRINUSE` on port 9002 six times over
        # while the lazily-armed slot 1 paired first try, on the same process and the same
        # card. Whatever else startup is doing with that port, a slot armed after the frame
        # server is serving does not race it.
        _remember_lane_base(port + LANE_PORT_OFFSET, device)

    def after_built(departure):
        # `serve` never returns -- its accept loop is the pool's life -- so everything
        # that needs the built departure is wired here. The riders path used to be
        # assigned after the call and was dead code, silently, which is how this hook
        # earned its test.
        departure.riders_path = envs.SGLANG_DEBUG_AFD_RIDERS_PATH.get()

    departure = serve(
        forward=make_pool_forward(model),
        runner=_pool_runner(model, device=device),
        host=host,
        port=port,
        min_batch=min_batch,
        max_wait_s=max_wait_ms / 1000.0,
        device=device,
        ready=ready,
        # what this pool will tell a caller about its own configuration. Computed
        # by the arms, opaque here -- see arms.arrangement_word.
        arrangement=arrangement_word(get_server_args()),
        # and what it will PUSH: the settings a host adopts instead of configuring itself,
        # stamped with the schema and code versions. See pushed_config.
        pushed=encode_config(pool_config(get_server_args(), model=model)),
        after_built=after_built,
    )
    # An operator asks for the riders histogram by naming a path. Without one the pool keeps the
    # record in memory and the health check reports "unknown" rather than guessing at a number
    # that decides whether this arrangement is paying for itself.
    return departure


class PoolSide:
    """What a pool process keeps alive for as long as it serves.

    Nothing reads this attribute. It exists because the departure thread outlives the
    call that built it and nothing else holds a reference: dropping it would let the
    thread be collected while the socket it answers on is still accepting.
    """

    def __init__(self, thread):
        self.thread = thread


class HostSide:
    """The host's end of the arrangement, held for the same reason `PoolSide` is."""

    def __init__(self, pool_client, routing):
        self.pool_client = pool_client
        self.routing = routing


def install_pool_side(model, *, max_context: int, device):
    """Take the pool side, or None when --afd-mode does not name this process one.

    The composition lives here rather than in the caller because the caller is the frozen model
    runner, which may construct and delegate but not decide: which of the two sides a process is,
    and what each side needs built, is this module's question.
    """
    disagg = get_disagg()
    if disagg.afd_mode != "pool":
        return None
    thread = serve_pool_in_background(
        model,
        port=disagg.afd_bootstrap_port,
        min_batch=disagg.afd_min_batch,
        max_wait_ms=disagg.afd_max_wait_ms,
        device=device,
    )
    return PoolSide(thread)


def install_host_side(model, *, transform):
    """Take the host side, or None when --afd-mode does not name this process one.

    `transform` is whatever arm this build carries, already installed. The sweep schedule is asked
    of it once and handed to both the router and the cache pool: the window is the gap between a
    feed-forward's issue and its collect, and both ends of it have to be the same schedule object
    or the sweep fires outside the gap it was meant to fill.
    """
    disagg = get_disagg()
    if disagg.afd_mode != "host":
        return None
    client, routing = install_host_routing(model, pool_addr=disagg.afd_pool_addr)
    return HostSide(client, routing)
