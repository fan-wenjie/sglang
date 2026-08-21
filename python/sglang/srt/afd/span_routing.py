"""The host under the group cut: attention, a KV cache, and three layers out of every four gone.

`roles.PoolRouting` replaces one layer's feed-forward with a call. This replaces a whole group of
layers with one call, which is a different install even though it is the same socket underneath:
the layers between two full attentions do not run here at all, and the host does not hold their
weights.

## What runs where

    host        embedding, the query and key/value projections, the softmax attention, the KV
                cache, and nothing else. Three of every four layers are not merely unused here --
                under `absent_ffn` they were never built
    pool        every feed-forward, every linear-attention layer, both of their recurrent states,
                and the residual that threads them

## The tour, and why the cut is where it is

A group of requests fills one bus. It visits cities together where the reading is SHARED -- a
feed-forward reads 2760 MiB of weights and reads them once for everybody aboard, so a rider who
wandered off would make somebody do that read twice. It disperses where the reading is PRIVATE:
a softmax attention reads the caller's own KV cache and there is nothing to amortise, so a 1k
request and a 128k request have no reason to wait for each other.

That boundary is the cut. It is not "every four layers" -- four is what this model's layout makes
it -- and `span.group_layers` reads it off the model rather than assuming the interval.

## Two buses, and the reason to want them

One bus in flight leaves both machines idle half the time: the pool waits out the host's attention,
the host waits out the pool's span. Staggering two groups fills both --

    one group    17 x (span 2046us + round trip 628us + attention 260us)  = 46.9 ms a step
    two groups   17 x span, with everything else in the shadow            = 34.8 ms a step

-- and the pool becomes the only thing on the critical path, which is the arrangement working as
intended rather than a problem to fix.

## How big a bus

Measured, not chosen (`benchmark/afd/span_cost.py`): a span is a weight read, so it costs the same
for one rider as for sixteen.

    riders     span     each
         4   2046 us  511.6 us
        16   2249 us  140.6 us     10% more bus for 4x the passengers
        32   2849 us   89.0 us     the arithmetic starts to be charged per head

Sixteen is where the flat part ends on this hardware. Above it the span stops being a weight read
and starts being a matrix multiply, and a bus that big also holds its riders longer than a rider
who missed it would wait.
"""

from __future__ import annotations

import itertools
import logging
import os

import torch
from sglang.srt.afd.pool_client import PoolClient, PoolClosed
from sglang.srt.afd.protocol import OP_SPAN, OP_SPAN_ENTER, OP_SPAN_EXIT, OP_SPAN_Q
from sglang.srt.afd.span import group_layers

logger = logging.getLogger(__name__)


class SpanClient:
    """One socket to the pool, speaking spans rather than feed-forwards.

    Wraps `PoolClient` rather than replacing it: the framing, the reconnect and the overlap
    accounting are the same, and only the shape of a call differs.
    """

    def __init__(self, client: PoolClient, *, reply_timeout_s: float) -> None:
        self.client = client
        self.reply_timeout_s = reply_timeout_s
        base = int.from_bytes(os.urandom(4), "big") << 24
        self._ids = itertools.count(base + 1)
        self.calls = 0

    def issue(self, group: int, o: torch.Tensor, row_ids: torch.Tensor, op: int = OP_SPAN):
        """Put a span on the wire and return without waiting for either half of its answer.

        `row_ids` travels because the pool holds a recurrent state per REQUEST and a decode batch
        carries one token from each of several. A pool that had to guess whose row was whose would
        advance one request's memory with another's token, and the output would stay fluent.
        """
        if row_ids.shape[0] != o.shape[0]:
            raise ValueError(
                f"{row_ids.shape[0]} row id(s) for {o.shape[0]} row(s). Every row has to say whose "
                f"recurrent state it advances."
            )
        handle = self.client.issue_frame(
            next(self._ids), group, (o, row_ids.reshape(-1, 1).to(torch.int64)), op
        )
        self.calls += 1
        return handle

    def collect_read_point(self, handle, device) -> torch.Tensor:
        """The first half: `h_(l+3)`, the query's source, which the pool sends early.

        Collected under its OWN opcode. The client's reply table is keyed by (request, layer, op)
        precisely so two answers to one call cannot be confused, and a span is the first caller to
        use that on purpose: both halves share a request and a group and differ only in which they
        are. Taking them under one key would hand the caller whichever landed first, and the query
        source and the span output are the same shape -- so the swap would attend with a hidden
        state from the wrong end of the span and say nothing about it.
        """
        early = handle._replace(op=OP_SPAN_Q)
        return self.client.collect_frame(early, device)[0]

    def collect_output(self, handle, device) -> torch.Tensor:
        return self.client.collect_frame(handle, device)[0]

    def collect(self, handle, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Both halves, in order. Present for callers with nothing to put between them -- which is
        not the arrangement: the point of splitting the reply is to project the query and sweep the
        cache between these two lines."""
        return (self.collect_read_point(handle, device),
                self.collect_output(handle, device))


class SpanRouting:
    """Replaces whole groups of layers with a call to the pool.

    Installed over sglang's decoder layers: the full-attention layer that heads a span keeps its
    attention and gives up its output projection and everything after it; the linear layers in the
    span do not run at all.
    """

    def __init__(self, model, client: SpanClient, layer_types: list[str]):
        self.model = model
        self.client = client
        self.spans = group_layers(layer_types)
        self.heads = {s[0] for s in self.spans if s[0] >= 0}
        self.passengers = {i for s in self.spans for i in s[1:]}
        self._undo: list = []
        self._install()

    def _install(self) -> None:
        layers = self.model.model.layers
        for span in self.spans:
            for layer_id in span[1:]:
                self._make_pass_through(layers[layer_id], layer_id)
        logger.info(
            "afd host: %s span(s), %s layer(s) served entirely by the pool, %s attention(s) kept "
            "here. Round trips a step: %s, against %s under the per-layer cut.",
            len(self.spans), len(self.passengers), len(self.heads),
            len(self.spans), len(layers) - 1,
        )

    def _make_pass_through(self, layer, layer_id: int):
        """A layer the pool runs. Its forward returns its input untouched.

        Returning the input rather than deleting the layer is what keeps this an install rather
        than a fork: sglang's model iterates its own layer list, and a list with holes in it would
        be a change to the model file. The weights are absent either way -- `absent_ffn` builds
        them on the meta device -- so this is not a layer running for free, it is a layer that
        cannot run.
        """
        original = layer.forward

        # matched to `Qwen3HybridLinearDecoderLayer.forward(hidden_states, residual, ...)` rather
        # than written as `*args`: a pass-through that swallowed a signature it did not understand
        # would keep working against a model whose layers take their arguments in another order,
        # and return the wrong one of them as the hidden state
        def pass_through(hidden_states, residual=None, *args, **kwargs):
            return hidden_states, residual

        layer.forward = pass_through
        self._undo.append(lambda ly=layer, o=original: setattr(ly, "forward", o))

    def remove(self) -> None:
        for fn in self._undo:
            fn()
        self._undo.clear()

    def report(self) -> dict:
        return {"spans": len(self.spans), "layers_on_the_pool": len(self.passengers),
                "attentions_here": len(self.heads), "calls": self.client.calls}


def bus_size_note(riders: int) -> str:
    """What a bus of this size costs a rider, from the measurement rather than from a guess."""
    measured = {1: 2070, 4: 2046, 8: 2132, 16: 2249, 32: 2849, 64: 3927}
    nearest = min(measured, key=lambda k: abs(k - riders))
    return (f"a bus of {riders} rides a span measured at about {measured[nearest]} us "
            f"({measured[nearest] / max(riders, 1):.0f} us a rider); the flat part of that curve "
            f"ends at 16 on this hardware")
