"""The two roles, as processes: a host that owns the cache and a pool that owns the weights.

    pool   loads the stack and answers `mlp(hidden)` for whichever layer a caller names
    host   loads the stack, installs the Early-Q read point, and sends every converted layer's
           feed-forward to the pool instead of running it locally

## What this buys today, and what it does not

The feed-forward genuinely leaves the host: the pool runs it, batches across callers, and the
host's own MLP weights go unused on the converted layers. That is the arrangement's plumbing, and
it is what a second machine needs.

The OVERLAP is not here yet, and the reason is worth stating rather than discovering in a
benchmark. In sglang's layer, `self.attn(q, k, v, forward_batch)` is one call: the sweep over the
cache and the fold-in of the current position happen inside one kernel. The protocol's whole point
is that the sweep needs only the query -- so the sweep can run while the pool works, and only the
fold-in needs `x_{l+1}`. Until the backend returns `(o, lse)` for the cached positions separately,
the host has nothing to do between issuing the call and needing its answer, and
`overlap_report()["mean_hidden_s"]` will read near zero. It reads near zero honestly: that is the
state of this port, not a property of the arrangement.

The triton backend already carries an `attn_lse` in its metadata, so the split is reachable. It is
the next step, not this one.

## Why the pool loads the whole model

It uses one layer's MLP at a time and could hold only those weights. Loading the stack is wasteful
and correct, and correctness first: a pool that loaded a subset would need its own weight-mapping
code, and a mapping bug there produces plausible tokens from the wrong weights.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import torch
from sglang.srt.afd.pool_client import PoolClient
from sglang.srt.afd.pool_server import serve

logger = logging.getLogger(__name__)


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


def run_pool(model, host: str, port: int, min_batch: int, max_wait_ms: int,
             device: torch.device | str, ready: threading.Event | None = None):
    """Serve until killed. Blocks."""
    logger.info("afd pool: %s layers, min_batch=%s, max_wait=%sms",
                len(model.model.layers), min_batch, max_wait_ms)
    return serve(
        forward=make_pool_forward(model),
        host=host,
        port=port,
        min_batch=min_batch,
        max_wait_s=max_wait_ms / 1000.0,
        device=device,
        ready=ready,
    )


class PoolRouting:
    """Replaces the feed-forward of named layers with a call to the pool."""

    def __init__(self, model, client: PoolClient, layers: tuple[int, ...], request_id: int):
        self.model = model
        self.client = client
        self.layers = layers
        self.request_id = request_id
        self._undo: list[Callable] = []
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
            handle = self.client.issue(self.request_id, layer_id, hidden_states)
            # Nothing between issue and collect YET -- see this module's docstring. The two calls
            # are kept apart so the day the sweep moves in between, only the middle changes.
            out = self.client.collect(handle, device)
            return out.to(dtype)

        return routed

    def remove(self) -> None:
        for fn in self._undo:
            fn()
        self._undo.clear()


def install_pool_routing(model, client: PoolClient, layers: tuple[int, ...],
                         request_id: int = 0) -> PoolRouting:
    if not layers:
        raise ValueError(
            "no layer routed to the pool. A host that runs every feed-forward itself is the "
            "colocated arrangement with a socket open beside it."
        )
    logger.info("afd host: routing %s layer(s) to the pool at %s", len(layers), client.address)
    return PoolRouting(model, client, tuple(layers), request_id)
