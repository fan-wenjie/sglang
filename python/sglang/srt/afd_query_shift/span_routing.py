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

A group of requests fills one bus. It stays in formation wherever the stage it is passing through
has FIXED latency, and disperses only where latency VARIES.

    feed-forward       fixed     a weight read, the same for any context
    linear attention   fixed     7.0 us at 1k, 32k and 256k alike
    softmax attention  VARIES    19 us at 1k, 2397 us at 128k

Holding a batch through a fixed-latency stage is free -- everybody finishes together. Holding it
through a variable one makes every short-context rider wait for the longest, which is the only
thing worth paying a round trip to avoid. So the cut goes where the latency stops being fixed.

That it lands on "weights to the pool, KV cache to the host" is a fact about this model, where the
variable-latency stage happens to be the one with the cache. The rule is the latency. It is also
not "every four layers" -- four is what this layout makes it -- and `span.group_layers` reads the
boundary off the model rather than assuming the interval.

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
from sglang.srt.afd.protocol import (
    OP_SPAN,
    OP_SPAN_ENTER,
    OP_SPAN_EXIT,
    OP_SPAN_Q,
    pack_positions,
)
from sglang.srt.afd_query_shift.span import group_layers

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

    def issue(self, group: int, o: torch.Tensor, row_ids: torch.Tensor, positions,
              op: int = OP_SPAN):
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
        # packed, not flattened. mrope carries a temporal, a height and a width row, so a
        # multimodal stack's positions are (3, tokens) and reshaping them to a column would hand
        # the rotation three times as many positions as there are tokens -- which is exactly how
        # this failed, at 366 against 122.
        handle = self.client.issue_frame(
            next(self._ids), group,
            (o, row_ids.reshape(-1, 1).to(torch.int64),
             pack_positions(positions.to(torch.int64))),
            op,
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

    def collect_output(self, handle, device):
        return self.client.collect_frame(handle, device)

    def collect_kv(self, handle, device):
        """The second half: this step's key and value, for the cache.

        They arrive after the query because they read `x_l`, which is not complete until the
        span's last feed-forward runs. That is not a delay this side pays for -- the sweep is
        already running against the query by the time they land.
        """
        got = self.client.collect_frame(handle, device)
        if len(got) != 2:
            raise RuntimeError(
                f"a span reply carried {len(got)} tensor(s) where the key and value were "
                f"expected. The two sides are running different versions of the protocol, and "
                f"unpacking anyway would append something that is not a key to the cache."
            )
        return got

    def collect(self, handle, device):
        """Both halves, in order. Present for callers with nothing to put between them -- which is
        not the arrangement: the point of splitting the reply is to sweep the cache between these
        two lines."""
        return (self.collect_read_point(handle, device),
                *self.collect_kv(handle, device))


_TRACED = {}


def _trace(layer_id: int, hidden_states, *, q, k, v, attn, out) -> None:
    """One line a head, the first few calls, when SGLANG_AFD_TRACE is set. Off otherwise.

    The arrangement's fault is that its output is approximately its input, and every check that
    could see WHERE that happens is either on the pool -- which the in-process probe has now
    cleared end to end -- or absent. This prints the magnitude of each tensor crossing the host's
    half, which is enough to say whether the attention output is empty, whether the query and the
    key and value arrived, and whether the span's reply resembles what was sent into it.

    Norms rather than hashes: the question is "is this tensor there at all", and two tensors that
    differ are not interesting here while a tensor that is zero is the whole answer. It is capped
    per layer because a decode step calls this 16 times and a generation calls it 16 times again.
    """
    import os

    if not os.environ.get("SGLANG_AFD_TRACE"):
        return
    seen = _TRACED.get(layer_id, 0)
    if seen >= int(os.environ.get("SGLANG_AFD_TRACE", "2")):
        return
    _TRACED[layer_id] = seen + 1

    def size(t):
        if t is None:
            return "-"
        f = t.float()
        return f"{tuple(t.shape)} |{f.norm():.4g}| max {f.abs().max():.4g}"

    logger.info(
        "afd trace layer %s call %s: in %s | q %s | k %s | v %s | attn %s | out %s",
        layer_id, seen, size(hidden_states), size(q), size(k), size(v), size(attn), size(out),
    )


class SpanRouting:
    """Replaces whole groups of layers with a call to the pool.

    Installed over sglang's decoder layers: the full-attention layer that heads a span keeps its
    attention and gives up everything else -- its projections included, so it does not even
    compute the query it attends with. The linear layers in the span do not run at all.

    The pieces it does use are named off the DECODER LAYER (`layer.attn`), not off a submodule.
    `self_attention` is a method on that layer, not an object, and reaching through it for `.attn`
    is an attribute error at the first token rather than at install.
    """

    def __init__(self, model, client: SpanClient, layer_types: list[str]):
        self.model = model
        self.client = client
        self.spans = group_layers(layer_types)
        self.heads = {s[0] for s in self.spans if s[0] >= 0}
        self.passengers = {i for s in self.spans for i in s[1:]}
        self._undo: list = []
        # the span issued by the previous head and not yet collected. One at a time per pass:
        # the host has nothing to do between issuing at layer l and collecting at layer l+4, so
        # depth here would buy nothing until several requests are in flight at once
        self._outstanding = None
        # what the previous head returned, so the next one can check nothing ran in between
        self._returned = None
        # the row ids of the span in flight. The pool calls back for the history mid-span, on the
        # reader thread, and it needs to know whose rows it is asking about -- and a list captured
        # at install would be the first span's rows for every span after it.
        self._rows = None
        # the history this host holds and the pool calls back for; set by install_span_routing
        self.history = None
        # the cache partition, built once a forward pass and shared by every sweep in it
        from sglang.srt.afd.split_attention import PerPassIndex

        self._index = PerPassIndex()
        self.sweeps = 0
        # prefill calls, which have no window to open. Kept apart from `sweeps` so a report can
        # say whether a run's windows were absent or merely shut.
        self.fused = 0
        self._install()

    def _install(self) -> None:
        layers = self.model.model.layers
        ordered = [s[0] for s in self.spans if s[0] >= 0]
        for span in self.spans:
            for layer_id in span[1:]:
                self._make_pass_through(layers[layer_id], layer_id)
        for position, layer_id in enumerate(ordered):
            self._make_head(
                layers[layer_id], layer_id,
                # the first head has no span behind it, so it opens one with the embedding it was
                # handed; the last has no span in front, so it closes the stack instead
                opens=(position == 0), closes=(position == len(ordered) - 1),
            )
        logger.info(
            "afd host: %s span(s), %s layer(s) served entirely by the pool, %s attention(s) kept "
            "here. Round trips a step: %s, against %s under the per-layer cut.",
            len(self.spans), len(self.passengers), len(self.heads),
            len(self.spans), len(layers) - 1,
        )

    def _make_head(self, layer, layer_id: int, *, opens: bool, closes: bool) -> None:
        """The one layer of a span that stays here: its attention, and nothing else.

        The residual stream does NOT arrive through sglang's layer loop. It arrives from the pool,
        which is the only end that has it -- the three layers in between did not run here. So this
        forward ignores the `hidden_states` and `residual` it is passed, EXCEPT at the first head,
        where what is passed is the embedding and there is no span behind it yet.

        Ignoring an argument is the kind of thing that works until an install half fails, so it is
        checked rather than assumed: each head records the tensor it returned and the next head
        refuses anything else. A pass-through that did not take -- one layer of forty-eight missed
        by the install, which is a mistake this tree has made -- would otherwise show up as fluent
        output and a throughput number about a different arrangement.
        """
        original = layer.forward

        def head(positions, hidden_states, residual=None, forward_batch=None, **kwargs):
            attn = layer.attn
            device = hidden_states.device
            rows = self._row_ids(forward_batch)
            self._rows = [int(r) for r in rows]
            if opens:
                handle = self.client.issue(
                    layer_id, hidden_states, rows, positions, OP_SPAN_ENTER)
            else:
                self._check_untouched(layer_id, hidden_states)
                handle = self._outstanding

            # the query arrives ALREADY PROJECTED, one feed-forward before the key and value.
            # Everything between this line and the next runs while the pool is still working.
            q = self.client.collect_read_point(handle, device)
            state = self._sweep(attn, forward_batch, q)

            k, v = self.client.collect_kv(handle, device)
            attn_output = self._join(attn, forward_batch, k, v, state, q)

            if closes:
                last = self.client.issue(layer_id, attn_output, rows, positions, OP_SPAN_EXIT)
                out = self.client.collect_output(last, device)[0]
                _trace(layer_id, hidden_states, q=q, k=k, v=v, attn=attn_output, out=out)
                return out, None
            _trace(layer_id, hidden_states, q=q, k=k, v=v, attn=attn_output, out=None)
            self._outstanding = self.client.issue(
                layer_id, attn_output, rows, positions, OP_SPAN)
            self._returned = attn_output
            return attn_output, None

        layer.forward = head
        self._undo.append(lambda ly=layer, o=original: setattr(ly, "forward", o))

    def _sweep(self, attn, forward_batch, q):
        """Attend the cache with the query alone. THIS is the window.

        It runs between the two halves of the pool's reply, so the pool's last feed-forward --
        298 us -- is spent while this reads the cache. The sweep is the variable-latency stage,
        up to 2397 us at 128k, and starting it a feed-forward early is the only lever this side
        has on it.

        A backend that cannot be partitioned is refused rather than quietly fused: a fused call
        here would be correct and would close the window, and the only symptom would be a
        throughput number that reads as this arrangement's.
        """
        from sglang.srt.model_executor.forward_context import get_attn_backend

        from sglang.srt.afd.split_attention import split_refusal, sweep

        backend = get_attn_backend()
        refusal = split_refusal(backend, attn, forward_batch)
        if refusal is not None:
            if forward_batch.forward_mode.is_decode():
                # at DECODE the window is the schedule. Running fused here would be a correct
                # model with the window shut and nothing downstream could tell, so it is fatal.
                raise RuntimeError(
                    f"the attention backend cannot be split at decode, so the sweep cannot be "
                    f"started before the key and value arrive: {refusal}."
                )
            # at PREFILL there is no window to open: the key and the value arrive with the query,
            # so there is nothing for a head start to be ahead OF. The fused path is the right one
            # and skipping the split costs nothing -- but it is COUNTED, because a run whose
            # window never opened must not be indistinguishable from one whose window is shut.
            self.fused += 1
            return None
        self.sweeps += 1
        return sweep(backend, attn, forward_batch, q=q, index=self._index)

    def _join(self, attn, forward_batch, k, v, state, q):
        """Fold this step's token into the swept cache, and write the cache."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        from sglang.srt.afd.split_attention import join

        if state is None:
            # no prefix to sweep: the first token of a request has nothing behind it. The fused
            # path is the right one here and the window had nothing to hide anyway.
            out = attn(q, k, v, forward_batch)
        else:
            out = join(get_attn_backend(), attn, forward_batch, k=k, v=v, state=state,
                       index=self._index)
        # ONE shape on the wire. The two paths do not agree on their own: the fused call returns
        # (rows, heads, head dim) and `join` returns (rows, heads x head dim), so a prefill and a
        # decode would put different shapes on the same socket. The pool multiplies this by a gate
        # of the flat shape, and the mismatch surfaced there rather than here -- two files away
        # from the divergence.
        return out.reshape(out.shape[0], -1)

    def _check_untouched(self, layer_id: int, hidden_states) -> None:
        if self._returned is None or hidden_states is not self._returned:
            raise RuntimeError(
                f"layer {layer_id} was handed a hidden state the previous head did not return, so "
                f"a layer between them ran. Under the group cut those layers belong to the pool "
                f"and their weights are on the meta device here; one of them running locally "
                f"means the pass-through install missed it. The output would stay fluent."
            )

    def current_rows(self):
        """Whose rows the span in flight carries, for the pool's callback to key its reads by.

        Raises rather than returning an empty list: a state read that could not say whose history
        it wanted would either fail loudly here or read slot zero for everybody, and only one of
        those is visible in the output.
        """
        if self._rows is None:
            raise RuntimeError(
                "the pool asked for a history reading with no span in flight on this host. The "
                "two ends disagree about which call is outstanding."
            )
        return self._rows

    @staticmethod
    def _row_ids(forward_batch) -> torch.Tensor:
        """Whose recurrent state each ROW advances -- one id a row, not one a request.

        `req_pool_indices` is sglang's per-REQUEST handle, and in decode the two coincide because
        a decode batch carries one token from each request. In prefill they do not: 122 tokens
        from one request is 122 rows and one index, and sending that pair is how this failed at
        its first token, twice.

        So the index is repeated by each request's extend length. A row sent under the wrong id
        advances somebody else's memory with this token, and neither end has any way to notice.
        """
        ids = forward_batch.req_pool_indices
        lengths = forward_batch.extend_seq_lens
        if lengths is None:
            return ids                                   # decode: one row a request already
        if lengths.shape[0] != ids.shape[0]:
            raise RuntimeError(
                f"{lengths.shape[0]} extend length(s) for {ids.shape[0]} request(s); the batch "
                f"disagrees with itself about how many requests it holds, and expanding it "
                f"anyway would attribute one request's tokens to another."
            )
        return ids.repeat_interleave(lengths.to(ids.device))

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
                "attentions_here": len(self.heads), "calls": self.client.calls,
                # the cache sweep runs between the two halves of the pool's reply, so the pool's
                # last feed-forward is spent while this side reads its cache. Counted, not
                # asserted: a window that stopped opening -- a backend that started refusing the
                # split, a request with no prefix -- looks exactly like one that never shut, and
                # the difference is the whole schedule.
                "sweep_window_open": True, "sweeps": self.sweeps, "fused_prefills": self.fused}


def bus_size_note(riders: int) -> str:
    """What a bus of this size costs a rider, from the measurement rather than from a guess."""
    measured = {1: 2070, 4: 2046, 8: 2132, 16: 2249, 32: 2849, 64: 3927}
    nearest = min(measured, key=lambda k: abs(k - riders))
    return (f"a bus of {riders} rides a span measured at about {measured[nearest]} us "
            f"({measured[nearest] / max(riders, 1):.0f} us a rider); the flat part of that curve "
            f"ends at 16 on this hardware")
