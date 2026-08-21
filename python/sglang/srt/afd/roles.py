"""The two roles, as processes: a host that owns the cache and a pool that owns the weights.

    pool   loads the stack and answers `mlp(hidden)` for whichever layer a caller names
    host   loads the stack, installs the Early-Q read point, and sends every converted layer's
           feed-forward to the pool instead of running it locally

## What this buys, and where

The feed-forward genuinely leaves the host: the pool runs it, batches across callers, and the
host's own MLP weights go unused on the converted layers. That is the arrangement's plumbing, and
it is what a second machine needs.

The OVERLAP is `sweep_ahead`, passed in here as `between`. `self.attn(q, k, v, forward_batch)`
used to be one call -- the sweep over the cache and the fold-in of this step's token inside one
kernel -- so there was nothing the host could do between issuing a feed-forward and needing its
answer. `split_attention` partitions that call, and the half that needs only the query is launched
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
from sglang.srt.afd.pool_client import PoolClient, PoolClosed
from sglang.srt.afd.pool_server import serve

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


def routable_layers(model) -> tuple[int, ...]:
    """The layers whose feed-forward this router can speak for.

    A sparse block takes a forward batch the wire does not carry, so it stays on the host. Chosen
    here rather than refused at the first token, because a stack that turns out to be half
    routable is a different arrangement from the one a benchmark was launched to measure.
    """
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    return tuple(
        i for i, layer in enumerate(model.model.layers)
        if not isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)
    )


def make_cache_pool(model, *, enabled, max_context: int, device):
    """The cache half of a two-pool split: a history, a sweep, an append. No weights."""
    if not enabled:
        return None
    from sglang.srt.afd.pool_attention import CachePool, KVHolder

    logger.info(
        "afd cache pool: histories only, up to %s positions a request. It holds no weight and "
        "answers no feed-forward: the sweep shares nothing between callers, so it is kept off "
        "the service whose whole economy is one weight read serving many.",
        max_context,
    )
    return CachePool(KVHolder(device, max_context), model.model.layers)


def attach_cache_pool(sweep_ahead, *, addr, connect_timeout_s: float = 30.0):
    """Point the Early-Q window at a cache pool, so its sweep goes out over the wire.

    This is what makes the two pools concurrent. The window already opens between the issue and
    the collect of the feed-forward -- that is the whole point of the moved read point -- so a
    sweep issued there is in flight to one machine while the feed-forward is in flight to another.
    """
    if sweep_ahead is None or not addr:
        return None
    from sglang.srt.afd.rendezvous import AppendLedger

    client = PoolClient(addr, connect_timeout_s)
    client.require(PoolClient.NEEDS_CACHE)
    sweep_ahead.cache_client = client
    sweep_ahead.ledger = AppendLedger()
    logger.info("afd host: sweeps go to the cache pool at %s, issued inside the window", addr)
    return client


def make_sweep_service(model, *, enabled, max_context: int, device):
    """The pool's cache and sweep, if --afd-kv-on-pool asked for them. None otherwise."""
    if not enabled:
        return None
    from sglang.srt.afd.pool_attention import KVHolder, SweepService
    from sglang.srt.afd.read_point import layer_types_of

    logger.info(
        "afd pool: holding the KV cache and the key/value projections, up to %s positions a "
        "request. The sweep is answered on the connection thread rather than queued for a "
        "departure: it reads the caller's own cache, so there is no shared weight read for a "
        "departure to amortise.",
        max_context,
    )
    return SweepService(model, KVHolder(device, max_context), layer_types_of(model))


def install_kv_on_pool(model, *, client, mode):
    """Which of the two lines to draw through attention, or neither.

        "cache"       the pool holds the cache and sweeps it; the host joins this step's token
        "projection"  the pool holds W_k and W_v; the host keeps the cache and the whole attention

    A null client is the reversed arrangement, where there is no weights pool for the key and
    value to move to. Checked before `mode`, because a run that asked for one of these lines and
    has nowhere to draw it should say so rather than quietly serving the unconverted stack.
    """
    if client is None:
        if mode:
            raise ValueError(
                f"--afd-pool-attention={mode!r} moves part of attention to a weights pool and "
                f"--afd-pool-addr names none. Either name one or drop the setting; installing "
                f"nothing here would serve the ordinary attention under this flag's name."
            )
        return None
    if not mode:
        return None
    from sglang.srt.afd.read_point import layer_types_of
    from sglang.srt.afd.remote_attention import (
        install_kv_projection,
        install_remote_attention,
    )

    from sglang.srt.afd.pool_client import PoolClient

    types = layer_types_of(model)
    if mode == "cache":
        client.require(PoolClient.NEEDS_CACHE | PoolClient.NEEDS_KV_PROJECTION)
        return install_remote_attention(model, client, types)
    if mode == "projection":
        client.require(PoolClient.NEEDS_KV_PROJECTION)
        return install_kv_projection(model, client, types)
    raise ValueError(f'--afd-pool-attention is "cache" or "projection", got {mode!r}')


def serve_pool_in_background(model, port: int, min_batch: int, max_wait_ms: int, device,
                             attention=None):
    """Answer feed-forward frames on a thread of an already-loaded server.

    The pool role reuses the ordinary launch path: the model is loaded, the scheduler starts, and
    nobody sends it a request. That is wasteful of a process and honest about what it is -- the
    alternative is a second entrypoint whose weight loading, quantisation and device placement
    would be a second implementation of the thing whose numbers must match the host's.
    """
    ready = threading.Event()
    thread = threading.Thread(
        target=run_pool,
        kwargs=dict(model=model, host="0.0.0.0", port=port, min_batch=min_batch,
                    max_wait_ms=max_wait_ms, device=device, ready=ready, attention=attention),
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


def install_host_routing(model, pool_addr: str, sweep_ahead, connect_timeout_s: float = 30.0):
    """Point every routable layer's feed-forward at the pool, and open the sweep window.

    A host with no weights pool is a real arrangement, not a misconfiguration: the reversed one,
    where the feed-forward stays put and only the sweep travels, to a cache pool named by
    --afd-cache-addr. It was unreachable until now because this function was the only door into
    host mode and it required an address, so the arrangement the budget table ranks first at long
    context had never once been run.

    Returning (None, None) rather than raising is what makes it reachable, and the two callers
    that use the result already treat None as "no pool" -- `install_kv_on_pool` refuses a null
    client, and the router is simply not installed.
    """
    if not pool_addr:
        logger.info(
            "afd host: no weights pool. The feed-forward stays on this machine and only the "
            "sweep travels, which is the reversed arrangement."
        )
        return None, None
    client = PoolClient(pool_addr, connect_timeout_s)
    client.require(PoolClient.NEEDS_FEED_FORWARD)
    return client, install_pool_routing(
        model, client, routable_layers(model), sweep_ahead=sweep_ahead
    )


def run_pool(model, host: str, port: int, min_batch: int, max_wait_ms: int,
             device: torch.device | str, ready: threading.Event | None = None, attention=None):
    """Serve until killed. Blocks."""
    logger.info("afd pool: %s layers, min_batch=%s, max_wait=%sms",
                len(model.model.layers), min_batch, max_wait_ms)
    import os

    departure = serve(
        forward=make_pool_forward(model),
        host=host,
        port=port,
        min_batch=min_batch,
        max_wait_s=max_wait_ms / 1000.0,
        device=device,
        ready=ready,
        attention=attention,
    )
    # An operator asks for the riders histogram by naming a path. Without one the pool keeps the
    # record in memory and the health check reports "unknown" rather than guessing at a number
    # that decides whether this arrangement is paying for itself.
    departure.riders_path = os.environ.get("AFD_RIDERS_PATH")
    return departure


class PoolRouting:
    """Replaces the feed-forward of named layers with a call to the pool."""

    def __init__(self, model, client: PoolClient, layers: tuple[int, ...], request_id: int,
                 between: Callable[[int], None] | None = None):
        self.model = model
        self.client = client
        self.layers = layers
        # Frames are keyed by (request_id, layer), so an id reused while a frame is outstanding
        # crosses two answers. A fixed id is safe only while one caller talks to the pool; the
        # counter makes every call of this host unique, and the random base keeps two hosts on
        # one pool from starting at the same number.
        self.request_id = request_id
        base = request_id if request_id else int.from_bytes(os.urandom(4), "big") << 24
        self._ids = itertools.count(base + 1)
        # what runs while the pool works. Called with the layer whose feed-forward is in flight,
        # AFTER the issue: `issue` copies the hidden states to the host and so synchronises the
        # stream, and work launched before it would be waited on by the send.
        self.between = between
        self._undo: list[Callable] = []
        # Skill check 4: whether anything overlapped. Reported from here because the host runs
        # inside the scheduler process, where no caller can reach the client to ask -- and a
        # schedule whose windows all closed looks exactly like one whose windows all opened,
        # except in these two numbers.
        self._calls = 0
        self._local_fallbacks = 0
        self._install()

    def _install(self) -> None:
        for layer_id in self.layers:
            layer = self.model.model.layers[layer_id]
            original = layer.mlp.forward
            layer.mlp.forward = self._route(layer_id, original)
            self._undo.append(
                lambda ly=layer, o=original: setattr(ly.mlp, "forward", o)
            )

    def _route(self, layer_id: int, original: Callable) -> Callable:
        def routed(hidden_states, *args, **kwargs):
            if args or kwargs:
                # a MoE block takes a forward_batch and the pool has no way to carry it
                raise RuntimeError(
                    f"layer {layer_id}'s feed-forward takes more than the hidden states "
                    f"({len(args)} positional, {sorted(kwargs)}); this router only speaks for a "
                    f"dense MLP. Route it locally or teach the wire its other arguments."
                )
            device, dtype = hidden_states.device, hidden_states.dtype
            try:
                handle = self.client.issue(next(self._ids), layer_id, hidden_states)
            except (OSError, PoolClosed) as e:
                return self._without_the_pool(layer_id, original, hidden_states, e)
            if self.between is not None:
                # the window. The next converted layer's query was projected from THIS layer's
                # h_l, so its cache sweep can be launched now and will run on the GPU while the
                # host waits on the socket below.
                self.between(layer_id)
            try:
                out = self.client.collect(handle, device)
            except (OSError, PoolClosed) as e:
                # THIS is where a pool dying lands, and where the first version did not look. A
                # call is outstanding when the process goes away, so the failure arrives at the
                # collect, not the issue -- and an exception inside a forward pass kills sglang's
                # scheduler, so a pool restart took the whole server with it.
                return self._without_the_pool(layer_id, original, hidden_states, e)
            self._calls += 1
            if self._calls % REPORT_EVERY == 0:
                # only the calls since the last line: a cumulative mean carries the pool's JIT
                # warm-up forever, and a schedule that stopped overlapping halfway through would
                # be averaged back into looking fine
                report = self.client.overlap_report(last=REPORT_EVERY)
                logger.info(
                    "afd host: %s pool call(s); last %s: build %.2f ms, send %.2f ms, "
                    "outstanding %.2f ms, blocked %.2f ms, hidden %.1f%%",
                    self._calls,
                    REPORT_EVERY,
                    1e3 * report.get("mean_build_s", 0.0),
                    1e3 * report.get("mean_send_s", 0.0),
                    1e3 * report["mean_outstanding_s"],
                    1e3 * report["mean_blocked_s"],
                    100.0 * (1.0 - report["mean_blocked_s"] / report["mean_outstanding_s"])
                    if report["mean_outstanding_s"] > 0 else 0.0,
                )
            return out.to(dtype)

        return routed

    def _without_the_pool(self, layer_id: int, original, hidden_states, error):
        """Serve this layer locally because the pool is gone, and say so.

        The alternative is what the first version did: re-raise, and let an exception inside a
        forward pass kill the scheduler. A pool restart then stops a server that was carrying
        hundreds of requests, which is a worse outcome than serving them from weights that were
        supposed to be elsewhere.

        So the host computes it, and the degradation is COUNTED and logged rather than silent. The
        host has these weights -- it loads the whole stack -- so the answer is right; what is lost
        is the arrangement, and a run that fell back for half its layers must not be able to
        report the arrangement's throughput under the arrangement's name.
        """
        self._local_fallbacks += 1
        if self._local_fallbacks == 1 or self._local_fallbacks % 512 == 0:
            logger.warning(
                "afd host: the pool at %s is unreachable (%s). Running layer %s locally; %s "
                "layer(s) so far have been served without it. The answers are right and the "
                "arrangement is not in effect -- any timing taken from here is a colocated timing.",
                self.client.address, type(error).__name__, layer_id, self._local_fallbacks,
            )
        self.client.reconnect()
        out = original(hidden_states)
        return out[0] if isinstance(out, tuple) else out

    def remove(self) -> None:
        for fn in self._undo:
            fn()
        self._undo.clear()


def install_pool_routing(model, client: PoolClient, layers: tuple[int, ...],
                         request_id: int = 0, sweep_ahead=None) -> PoolRouting:
    if not layers:
        raise ValueError(
            "no layer routed to the pool. A host that runs every feed-forward itself is the "
            "colocated arrangement with a socket open beside it."
        )
    between = None
    if sweep_ahead is not None:
        # the schedule takes over the trigger for the layers it routes; without this handover the
        # sweep would fire from the prepare_mlp wrapper, BEFORE the issue, and the send would wait
        # on it -- a correct model with the window closed, and a benchmark that reads as "the
        # overlap does not help".
        sweep_ahead.routed_layers = set(layers)
        between = sweep_ahead.sweep_after_issue
    logger.info(
        "afd host: routing %s layer(s) to the pool at %s, %s of them opening a sweep window",
        len(layers), client.address,
        len(set(layers) & set(sweep_ahead.sweep_at)) if sweep_ahead is not None else 0,
    )
    return PoolRouting(model, client, tuple(layers), request_id, between=between)
