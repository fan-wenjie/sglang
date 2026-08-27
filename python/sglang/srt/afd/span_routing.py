"""The host under the group cut: attention, a KV cache, and three layers out of every four gone.

The per-layer cut replaced one layer's feed-forward with a call. This replaces a whole group of
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
import time

import torch

from sglang.srt.afd.pool_client import PoolClient
from sglang.srt.afd.protocol import (
    OP_SPAN,
    OP_SPAN_LANE,
    OP_SPAN_ENTER,
    OP_SPAN_EXIT,
    OP_SPAN_Q,
    pack_positions,
)
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

    def issue(
        self,
        group: int,
        o: torch.Tensor,
        row_ids: torch.Tensor,
        positions,
        op: int = OP_SPAN,
        windows=None,
    ):
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
        tensors = (
            o,
            row_ids.reshape(-1, 1).to(torch.int64),
            pack_positions(positions.to(torch.int64)),
        )
        if windows is not None:
            # the span's convolution windows, one per linear layer, because the ring lives here
            # and the convolution is about to run over there. See `SpanRouting._windows_for`.
            tensors = tensors + (windows,)
        handle = self.client.issue_frame(next(self._ids), group, tensors, op)
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

    def kv_already_arrived(self, handle) -> bool:
        """Whether the second half of the reply is already waiting, without taking it."""
        return self.client.reply_waiting(handle)

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
        return (
            self.collect_read_point(handle, device),
            *self.collect_kv(handle, device),
        )


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
        layer_id,
        seen,
        size(hidden_states),
        size(q),
        size(k),
        size(v),
        size(attn),
        size(out),
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

    def __init__(
        self, model, client: SpanClient, layer_types: list[str], *, query_shift: int = 0
    ):
        self.query_shift = int(query_shift)
        self.model = model
        self.client = client
        self.types = list(layer_types)
        self.spans = group_layers(layer_types)
        self._span_by_head = {s[0]: s for s in self.spans}
        self.heads = {s[0] for s in self.spans if s[0] >= 0}
        # the earliest head, which is where a forward pass first reaches this routing
        self._first_head = min(self.heads) if self.heads else None
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
        # set by the installer when the embedding runs on the pool; None keeps the hidden state
        # travelling, which is what every arrangement before this did
        self.pending_ids = None
        self.sweeps = 0
        # Which arm of the schedule's max(host, pool) actually won each window. The design
        # requires the POOL arm: the host is deliberately the small card, and any change that
        # moves work here until the host arm wins has broken the schedule's premise, however
        # correct its answers. Counted at the sweep's end and warned on, not assumed.
        self.pool_was_the_max = 0
        self.host_was_the_max = 0
        # prefill calls, which have no window to open. Kept apart from `sweeps` so a report can
        # say whether a run's windows were absent or merely shut.
        self.fused = 0
        # decode layers whose backend could not be split, so their window was shut
        self.fused_decodes = 0
        # head layers run, and how often the counters below are published. The report is the
        # only place a run says whether its mechanism EXECUTED -- a comparison between two arms
        # that never varied the thing under test reads as a clean null, and this arrangement has
        # produced three of those. Printed at install it is all zeros, which is the state it is
        # least worth reading in.
        self.head_calls = 0
        self.report_every = 200
        # The window's WIDTH, summed and counted rather than averaged here, so a report
        # can divide and a caller can see how many calls the average rests on. The count
        # is kept because an average without its denominator is how a mixed population
        # once got read as a measurement -- see `PoolClient._count_inbound`.
        self.window_s = 0.0
        self.windows = 0
        self._install()

    def _install(self) -> None:
        layers = self.model.model.layers
        ordered = [s[0] for s in self.spans if s[0] >= 0]
        for span in self.spans:
            for layer_id in span[1:]:
                self._make_pass_through(layers[layer_id], layer_id)
        for position, layer_id in enumerate(ordered):
            self._make_head(
                layers[layer_id],
                layer_id,
                # the first head has no span behind it, so it opens one with the embedding it was
                # handed; the last has no span in front, so it closes the stack instead
                opens=(position == 0),
                closes=(position == len(ordered) - 1),
            )
        logger.info(
            "afd host: %s span(s), %s layer(s) served entirely by the pool, %s attention(s) kept "
            "here. Round trips a step: %s, against %s under the per-layer cut.",
            len(self.spans),
            len(self.passengers),
            len(self.heads),
            len(self.spans),
            len(layers) - 1,
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
            if opens and layer_id == self._first_head:
                # once a forward pass, before any span of it runs. See `slot_reset`: both slot
                # tables are keyed by a row id sglang reuses, and neither was ever cleared.
                from sglang.srt.afd.slot_reset import (
                    forget_starting_requests,
                )

                forget_starting_requests(
                    forward_batch, history=self.history, client=self.client.client
                )
            rows = self._row_ids(forward_batch)
            self._rows = [int(r) for r in rows]
            if opens:
                # Token ids when the embedding is on the pool, the hidden state when it is here.
                # `hidden_states` is the NaN placeholder `remote_embedding` returned -- carrying it
                # would send 10 KiB a row of nothing, and the pool could not tell it from a real
                # hidden state.
                payload = (
                    self.pending_ids.take(int(rows.shape[0]))
                    if self.pending_ids is not None
                    else hidden_states
                )
                handle = self.client.issue(
                    layer_id,
                    payload,
                    rows,
                    positions,
                    OP_SPAN_ENTER,
                    # the ENTER runs the prologue span, whose head the grouping names -1
                    windows=self._decode_windows(-1, rows, forward_batch),
                )
            else:
                self._check_untouched(layer_id, hidden_states)
                handle = self._outstanding

            # the query arrives ALREADY PROJECTED, one feed-forward before the key and value.
            # Everything between this line and the next runs while the pool is still working.
            q = self.client.collect_read_point(handle, device)
            read_point_at = time.perf_counter()
            state = self._sweep(attn, forward_batch, q)

            if self.client.kv_already_arrived(handle):
                # the pool finished its feed-forward AND the wire before this side finished its
                # sweep: the host arm was the longer one this window
                self.host_was_the_max += 1
            else:
                self.pool_was_the_max += 1
            self._refuse_a_host_shaped_max()
            k, v = self.client.collect_kv(handle, device)
            # The window's WIDTH, which no count can give. `sweeps` says the two-piece reply
            # happened, and it happens under EITHER shift -- it read 91% on both arms of a
            # comparison where the shift was not in fact being varied, and separated nothing.
            # What separates the arms is how much pool work sits between the halves: a whole
            # feed-forward at shift 1, only the k/v projection at shift 0. That is this number.
            self.window_s += time.perf_counter() - read_point_at
            self.windows += 1
            self.head_calls += 1
            if self.head_calls % self.report_every == 0:
                logger.info("afd host: %s", self.report())
            attn_output = self._join(attn, forward_batch, k, v, state, q)

            if closes:
                last = self.client.issue(
                    layer_id, attn_output, rows, positions, OP_SPAN_EXIT
                )
                out = self.client.collect_output(last, device)[0]
                _trace(
                    layer_id, hidden_states, q=q, k=k, v=v, attn=attn_output, out=out
                )
                return out, None
            _trace(layer_id, hidden_states, q=q, k=k, v=v, attn=attn_output, out=None)
            span_windows = self._decode_windows(layer_id, rows, forward_batch)
            # THE HOST DECIDES, at issue, one frame at a time: the triangle rides the NCCL
            # lane exactly when the lane is up, the batch is one decode row, and the
            # windows are attached (windows non-None is already that predicate). The op on
            # the frame is what carries the decision, so readiness cannot race: the
            # expectations below are announced for exactly the spans that say span_lane.
            span_op = OP_SPAN
            if span_windows is not None and int(rows.shape[0]) == 1:
                span_op = self._lane_op_for(layer_id, rows)
            self._outstanding = self.client.issue(
                layer_id,
                attn_output,
                rows,
                positions,
                span_op,
                windows=span_windows,
            )
            self._returned = attn_output
            return attn_output, None

        layer.forward = head
        self._undo.append(lambda ly=layer, o=original: setattr(ly, "forward", o))

    def _lane_op_for(self, head: int, rows) -> int:
        """OP_SPAN_LANE with the triples announced, or OP_SPAN when there is no lane.

        A middle span at rung 2 cooks EVERY linear layer it holds, in span order (`run`
        sends the first's early before the head's mlp and `_linear_run` the rest), so the
        host can announce the whole sequence at issue -- which (layer, row) each lane
        triple is for -- and nothing about it needs to cross the wire.
        """
        from sglang.srt.afd.lane import the_lane
        from sglang.srt.afd_query_shift.nccl_lane import the_triangle

        lane = the_lane()
        triangle = the_triangle()
        if lane is None or not lane.ready or triangle is None:
            return OP_SPAN
        span = self._span_by_head.get(head)
        if span is None or span[0] < 0:
            return OP_SPAN
        linear = [i for i in span[1:] if self.types[i] == "linear_attention"]
        if not linear:
            return OP_SPAN
        row = int(rows.reshape(-1)[0])
        cache = self.history.cache
        ring_w = cache.conv.shape[-2]
        kh, vh, dk, dv = self._dims_for_lane()
        q_w = vh * dk
        slab_w = vh * dk + vh * dv + 2 * vh + ring_w
        triangle.expect([(lid, row, q_w, slab_w) for lid in linear])
        return OP_SPAN_LANE

    def _dims_for_lane(self):
        svc = self.history
        return svc.dims

    def _decode_windows(self, head: int, rows, forward_batch):
        """`_windows_for`, at decode only. A prefill chunk's rows are one request's own tokens;
        they convolve sequentially inside the scan on this side, and a window per token would be
        both wrong and enormous."""
        if self.query_shift == 0 or self.history is None:
            return None
        if forward_batch is None or not forward_batch.forward_mode.is_decode():
            return None
        return self._windows_for(head, rows)

    def _windows_for(self, head: int, rows) -> torch.Tensor | None:
        """The span's convolution PARTIALS: one weighted column a layer, serving both cooks.

        The early convolution and the mix convolution read the same three history columns and
        the same first three taps of the weight -- they differ only in the newest tap -- so the
        history's whole contribution is one weighted sum:

            partial[c] = sum over i of w[c, i] * ring[c, 1 + i]

        and each cook finishes it as `silu(partial + w_last * x)` with its own newest tap. One
        20 KiB column a layer replaces a 60 KiB window, and the pool's kernel loses the tap
        loop. Reassociation-level difference only, the same regime as the fused cook.

        `None` whenever the pool will not cook: shift 0, the lower rungs of the ladder, a
        prefill pass (its rows convolve as chunks on this side, inside the scan), or a history
        this routing was not given. Order matches `SpanRunner._hold_windows`: ascending layer
        id, flat, because the wire carries (rows, columns) and nothing else.
        """
        if self.query_shift == 0 or self.history is None:
            return None
        span = self._span_by_head.get(head)
        if span is None:
            return None
        linear = [i for i in span if self.types[i] == "linear_attention"]
        if not linear:
            return None
        cache = self.history.cache
        slots = torch.tensor(
            [cache.slot_of(int(r)) for r in rows],
            device=cache.conv.device,
            dtype=torch.int64,
        )
        partials = []
        for lid in linear:
            window = cache.conv[lid].index_select(0, slots)[..., 1:]
            weight = self.history.conv_weight(lid).to(window.device)[:, :-1]
            partials.append((window * weight).sum(-1))
        stacked = torch.stack(partials, dim=1)
        return stacked.reshape(stacked.shape[0], -1)

    def _refuse_a_host_shaped_max(self) -> None:
        """Warn, loudly and repeatedly, when the host has become the schedule's max.

        The whole arrangement prices itself on e = max(host arm, pool arm) resolving to the
        pool: the host's early work is supposed to FIT inside the pool's feed-forward, and the
        host is deliberately the slower, cheaper card. An occasional pre-arrived frame is wire
        jitter; a majority of them means some change has moved work onto the host until its arm
        is the longer one -- at which point the shift is serialising work onto the bottleneck
        and every layer pays the overflow. That has happened here once already, silently, and
        cost 10.7 percentage points before a ladder found it. Hence a standing tripwire rather
        than a review note: the warning re-fires every period the condition persists, because a
        broken premise that was logged once at startup is a premise nobody is looking at.
        """
        decided = self.pool_was_the_max + self.host_was_the_max
        # Period and majority are arbitrary; chosen to stay quiet under jitter, not tuned.
        if decided == 0 or decided % 512:
            return
        if self.host_was_the_max * 2 <= decided:
            return
        logger.warning(
            "afd query shift: the HOST is the schedule's max in %d of %d windows. The design "
            "requires the pool's feed-forward to be the longer arm; something now on the host "
            "-- a sweep grown past the window, or work an edit moved here -- exceeds it, and "
            "every layer is paying the overflow on the bottleneck card.",
            self.host_was_the_max,
            decided,
        )

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
        from sglang.srt.afd.split_attention import split_refusal, sweep
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        refusal = split_refusal(backend, attn, forward_batch)
        if refusal is not None:
            if forward_batch.forward_mode.is_decode():
                # At DECODE the window is the schedule, and running fused here closes it. That
                # used to be fatal, on the argument that a fused call would be correct and
                # nothing downstream could tell -- which is the right objection and the wrong
                # remedy. The remedy is that it CAN be told: every such layer is counted and the
                # count rides in the same report as the windows, so a run whose window never
                # opened cannot be quoted as one whose window is open.
                #
                # Fatal costs more than it buys on a family whose full-attention layers cannot
                # be split at all. `kimi_linear`'s seven are MLA, whose decode takes its own
                # path through `forward_decode`; refusing them refuses the whole checkpoint,
                # including its twenty recurrent layers, which have no such trouble and are
                # where this arrangement's case is made.
                if not self.fused_decodes:
                    logger.warning(
                        "afd host: this backend cannot split an attention at decode (%s), so "
                        "the sweep runs fused and the window is SHUT for those layers. The "
                        "recurrent layers are unaffected. `fused_decodes` in the report counts "
                        "them; a latency compared against an arrangement with the window open "
                        "is not comparing the same thing.",
                        refusal,
                    )
                self.fused_decodes += 1
                return None
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
        from sglang.srt.afd.split_attention import join
        from sglang.srt.model_executor.forward_context import get_attn_backend

        if state is None:
            # no prefix to sweep: the first token of a request has nothing behind it. The fused
            # path is the right one here and the window had nothing to hide anyway.
            out = attn(q, k, v, forward_batch)
        else:
            out = join(
                get_attn_backend(),
                attn,
                forward_batch,
                k=k,
                v=v,
                state=state,
                index=self._index,
            )
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
            return ids  # decode: one row a request already
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
        return {
            "spans": len(self.spans),
            "layers_on_the_pool": len(self.passengers),
            "attentions_here": len(self.heads),
            "calls": self.client.calls,
            # the cache sweep runs between the two halves of the pool's reply, so the pool's
            # last feed-forward is spent while this side reads its cache. Counted, not
            # asserted: a window that stopped opening -- a backend that started refusing the
            # split, a request with no prefix -- looks exactly like one that never shut, and
            # the difference is the whole schedule.
            # COUNTED, not asserted -- which is what the comment above asks for and what
            # neither of this field's two previous values did. It was a hardcoded True; then
            # it was "corrected" to a hardcoded False on the strength of a docstring in
            # `installer.py` saying the window was not open, without reading `head`, where
            # the sweep sits between `collect_read_point` and `collect_kv` and always has.
            # A literal cannot tell a window that stopped opening from one that never shut,
            # in either direction.
            "sweep_window_open": self.sweeps > 0,
            "pool_was_the_max": self.pool_was_the_max,
            "host_was_the_max": self.host_was_the_max,
            "sweeps": self.sweeps,
            "fused_prefills": self.fused,
            "fused_decodes": self.fused_decodes,
            # milliseconds of POOL work between the two halves of the reply -- the window's
            # width. Shift 1 should show a whole feed-forward here and shift 0 only the k/v
            # projection; if the two arms agree, the shift is not reaching the pool.
            "window_ms": (
                round(1000 * self.window_s / self.windows, 3) if self.windows else 0.0
            ),
            "windows": self.windows,
        }


def bus_size_note(riders: int) -> str:
    """What a bus of this size costs a rider, from the measurement rather than from a guess."""
    measured = {1: 2070, 4: 2046, 8: 2132, 16: 2249, 32: 2849, 64: 3927}
    nearest = min(measured, key=lambda k: abs(k - riders))
    return (
        f"a bus of {riders} rides a span measured at about {measured[nearest]} us "
        f"({measured[nearest] / max(riders, 1):.0f} us a rider); the flat part of that curve "
        f"ends at 16 on this hardware"
    )
