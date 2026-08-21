"""One group of four layers as a single call: the pool's whole job between two attentions.

The per-layer arrangement sends a frame for every feed-forward. On this model that is 63 round
trips a decode step, and the measurements say the fixed cost of a round trip -- 130 us of latency
plus about 148 us of protocol -- is most of what a feed-forward call costs at batch 4. The cut here
is chosen to make that number small by making the unit large.

## Where the cut is, and why there

Qwen3.8-27B's layers run `[linear, linear, linear, full] x 16`. Read from one full attention to the
next, a group is:

    gate | W_o | FFN | lin | FFN | lin | FFN | lin | FFN | W_q -> q,  W_kv -> k, v
    ^                                                                             ^
    the host's attention output                     the next attention's query, key and value

Everything between those two arrows is weights and per-request recurrent state, and none of it is
the KV cache. So the whole of it goes to the pool as ONE call, and **the host keeps only the KV
cache and the sweep over it**. It runs no weight matrix at all.

    round trips a step     16   (against 63 for the per-layer cut)
    span, measured       2046 us at batch 4 (benchmark/afd/span_cost.py)

## What the two ends cost, which is the only comparison worth making

They run in parallel, so the arrangement's step time is the MAX of the two, not the sum. Summing
them was wrong twice in this arrangement's history and in the same direction both times.

    context      pool     host      max     host busy
      1,024   37.3 ms   0.3 ms  37.3 ms           1%
     28,449   37.3 ms   8.3 ms  37.3 ms          22%
    131,072   37.3 ms  38.4 ms  38.4 ms         100%

    colocated, bfloat16, measured               ~38 ms

**The ceiling is parity, and it is reached.** 30.0 of the pool's 37.3 ms is reading the model's
50 GiB of weights once, and a colocated server reads the same 50 GiB -- no arrangement of two
machines makes that read smaller. What disaggregation buys is that the 32 GiB card does not have
to hold them, and that one pool's read can serve a busload of riders from several hosts.

Every earlier "N times faster" figure here was against the per-layer cut's 126.9 ms, which is a
comparison against a bad implementation rather than against a baseline.

The two ends balance near 131k context. Below it the pool is the bottleneck and the host idles;
above it the sweep is, and more pool does not help.

## The rule that decides where to cut: fixed latency against variable latency

A batch only has to be re-formed where latency VARIES. That is the whole principle, and every
other property of this arrangement follows from it.

    stage                     does its cost depend on context?   measured
    feed-forward              no                                 a weight read; context-free
    linear attention          no                                 7.0 us at 1k, 32k and 256k alike
    softmax attention         YES                                19 us at 1k, 2397 us at 128k

Where latency is fixed, everybody in a batch finishes together and holding the batch costs
nothing. Where it varies, one long-context rider makes every short-context rider wait -- so that
is the only place worth paying to break formation.

On this model the variable-latency stage is the softmax attention, and the softmax attentions are
exactly where the KV cache is, which is why the cut looks like "weights on the pool". That is a
coincidence of this model, not the rule. The rule is the latency.

This is a different rule from the per-layer arrangement's, not a tuning of it. That one cut by
OWNERSHIP -- a recurrent state belongs to the request, so it stays with the request -- which is
why forty-eight linear-attention layers stayed on the host there and come here now. Their state is
per-request and their latency is fixed, and under this rule the second fact is the one that counts.

## The bus, the station, and the waiting room

A group's batch is formed once, at the aggregation point before `W_o`, and cannot change until the
span ends -- the four feed-forwards are one uninterrupted read of 2760 MiB of weights, and a rider
who joined halfway would need that read done again. So:

  * riders board at `W_o`, together
  * the bus runs the span, 2046 us, with a fixed passenger list
  * at the far end everybody gets off into the waiting room, and the batch DISSOLVES
  * each rider's host computes its own softmax attention, taking as long as its own context takes
  * whoever finishes first boards the next bus

The last two points are what make this schedule tolerate mixed context lengths. A 128k request's
sweep is 2397 us and a 1k request's is 19 us; under a fixed batch the short one would wait for the
long one every layer. Here it does not wait at all -- it takes the next departure, and the long one
takes the one after. **Context length decides which bus a request catches, not how fast a bus
runs.** That is the property the per-layer cut did not have.

## Why the reply is two messages

The read point is shift 1: the next full attention's query projects from `h_{l+3}`, the residual
after the last linear layer's attention and before its feed-forward. That value exists one
feed-forward before the span's output does -- so the query is projected there, HERE, and sent
immediately. The host sweeps with it while this side spends the last feed-forward.

The key and value follow, after that feed-forward, because they read `x_l` and `x_l` is not
complete until it runs. They are not on the critical path: this step's join uses key and value the
caller already has, and the cache only has to hold them by the NEXT step.

Putting the query projection on this side rather than the host's is a TRADE, and it is worth
writing down which way it runs, because under a sum it looks free and under a max it does not:

    on the host   costs host time, of which there is 29 ms spare at 28k        free
                  costs 3.12 GiB of a 32 GiB card                             scarce
    on the pool   costs 1.87 ms a step on the side that IS the bottleneck      +5%
                  gives the 3.12 GiB back: 70,689 tokens a request to 83,489   +18% context

Taken, deliberately: on a consumer card the context length is the binding constraint, and the host
that results holds no weights at all -- it can be a cheap large-memory card rather than a second
copy of the pool.

What that tail is worth, measured rather than assumed:

    4 x feed-forward     2040 MiB   74% of the span   361 us each   measured
    3 x linear attention  660 MiB   24%               200 us each   measured
    W_o                    60 MiB    2%                35 us
    one round trip                                    628 us

The per-layer figures are MEASURED (`benchmark/afd/span_parts.py`); dividing the weight bytes gave
298 and 129, which is 22% and 55% low. The linear attention is much further from its weight-read
floor than the feed-forward is, because it is three small matrices and a recurrence rather than one
large read.

That measurement is also the sharpest statement of why the cut is where it is:

    one feed-forward                                361 us
    one linear attention plus one round trip        828 us     2.29x the feed-forward
    three linear attentions plus one round trip    1227 us     0.85x of four feed-forwards

**At LAYER granularity the wire costs more than twice the work it enables. At SPAN granularity it
fits inside the feed-forward chain with 15% to spare.** Same wire, same layers, opposite sign --
which is the whole of what changed between the two cuts.

The early message is covered by one feed-forward: 361 us against the ~105 us the query itself
spends on the wire, so the query is at the host and being swept with well before the span ends.
The pool still waits for the host's answer afterwards -- that wait is the "host busy 22%" line in
section 19, and it is what a second tour group fills.

Shift 2 would move the read point to `h_{l+2}`, which exists before the last linear attention runs,
and the cover would become 129 + 298 = 427 us. That is the one place where the linear attentions
could be made to pay for the wire, and it is not taken here: the arrangement's standing constraint
is that only the query moves and the shift stays at 1. Recorded as a knob with a known gain and a
known cost, not as an oversight.

The residual never travels. A group's input residual is the previous group's output, which the pool
computed and can keep; sending it back and forth would be 5120 columns a direction for a value
neither end has changed.

## What the pool has to hold, that it did not before

Both of a linear layer's recurrent states, because the host is no longer in the loop to run any of
it: the gated delta rule's state (1.5 MiB a layer a request) and the short convolution's (about
60 KiB). `linear_state.py` notes that the convolution stays on the host -- that was true of the
per-layer cut, where moving it bought a fortieth of the memory for a second round trip. Under this
cut there is no second round trip to spend, and no host-side call to run it in.
"""

from __future__ import annotations

import logging
import threading

import torch

logger = logging.getLogger(__name__)


def group_layers(layer_types: list[str]) -> list[tuple[int, ...]]:
    """The spans, as layer indices, read off the model's own layer list.

    A span runs from a full-attention layer's output projection to the next full-attention layer's
    hidden input, so it is named by the full layer that starts it and contains every layer up to
    but not including the next full one.

    The layers BEFORE the first full attention are their own span with no `W_o` at its head -- on
    this model that is layers 0, 1, 2, fed by the embedding. It is returned with a leading -1 so a
    caller can tell the headless one from the rest without re-deriving the layout.
    """
    full = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
    if not full:
        raise ValueError(
            "no full-attention layer in this stack, so there is no place to cut a span. The group "
            "cut is defined by where the KV cache is, and a model with no KV cache has no host "
            "side to keep."
        )
    spans = []
    if full[0] > 0:
        spans.append((-1, *range(0, full[0])))
    for start, nxt in zip(full, full[1:]):
        spans.append(tuple(range(start, nxt)))
    # the last full attention runs to the end of the stack; its span is the tail, which ends at
    # the norm and the language-model head rather than at another attention
    spans.append(tuple(range(full[-1], len(layer_types))))
    return spans


def _runs(request_ids) -> list[tuple[int, int, int]]:
    """The batch as (request, first row, row count), in order.

    A decode bus has one row a request and every run is length one. A prefill bus has one row a
    TOKEN, so a request's rows are a contiguous run that has to be walked in order. A bus carrying
    both has runs of both lengths, which is why this is not a `forward_mode` question: the mode
    describes the batch and the runs describe the rows.
    """
    runs, start = [], 0
    for i, r in enumerate(request_ids):
        if i + 1 == len(request_ids) or int(request_ids[i + 1]) != int(r):
            runs.append((int(r), start, i + 1 - start))
            start = i + 1
    return runs


def _add_and_norm(norm, hidden: torch.Tensor, residual: torch.Tensor | None):
    """sglang's fused add-and-normalise, and the residual it returns.

    Split out because the residual it produces is the thing the read point reads. Called with
    `residual=None` for the first layer of a pass, where there is nothing to add.
    """
    if residual is None:
        return norm(hidden), hidden
    out = norm(hidden, residual)
    if not isinstance(out, tuple):
        raise RuntimeError(
            f"{type(norm).__name__} did not return a (hidden, residual) pair when given a "
            f"residual. The span reads the residual as the shifted read point's source, and a "
            f"norm that folds it in silently would leave the query projecting from the wrong "
            f"tensor -- a correct-looking model with the read point undone."
        )
    return out


def _runs_of(request_ids) -> list[tuple[int, int, int]]:
    """(request id, first row, row count) for each consecutive run of the same request.

    A decode batch is one row a request, so every run has length one and this is the identity. A
    PREFILL chunk is many rows of ONE request -- its own consecutive tokens -- and that is the case
    the tables below were written without.
    """
    runs = []
    for index, r in enumerate(request_ids):
        rid = int(r)
        if runs and runs[-1][0] == rid:
            runs[-1][2] += 1
        else:
            runs.append([rid, index, 1])
    return [(rid, start, count) for rid, start, count in runs]


def _trace_step(tag, layer_id, value) -> None:
    """One tensor on the way through a span, when SGLANG_AFD_SELFCHECK is set.

    Used to bisect the prologue. Every other span's boundary now agrees with the model's own
    per-layer residual to about one percent; `run_prologue` is 11% high, and it is the only span
    with its own code path -- it starts from the embedding and has no W_o, no output gate and no
    attention output at its head.

    BOTH sides of the feed-forward are logged, because the model's own layer returns the stream
    BEFORE its feed-forward is folded in: `postprocess_layer` hands back (mlp output, x + attn) and
    the NEXT layer's `prepare_attn` does the addition. So a forward hook on the model's layer gives
    x + attn, and `residual + hidden` here gives x + attn + mlp. Comparing those two was the first
    reading of this bisect -- 38.5 against 28.8, called +34% -- and they are different quantities.
    The pre-mlp line is the one that lines up; the post-mlp line is kept so the difference between
    them is visible rather than assumed.

    This is the third time in this search that two correct measurements of different quantities
    were compared. The other two: a trace that walked rows against a reference that walked layers,
    and a control that reversed a filter on both sides at once.
    """
    import os

    if not os.environ.get("SGLANG_AFD_SELFCHECK"):
        return
    seen = _STEP_SEEN.get(tag, 0)
    if seen >= int(os.environ.get("SGLANG_AFD_SELFCHECK", "3")):
        return
    _STEP_SEEN[tag] = seen + 1
    row = value[0].float()
    logger.info("afd step: %s layer %s -- rows %s |%.5g| rms %.5g max %.5g",
                tag, layer_id, value.shape[0], float(row.norm()),
                float(row.pow(2).mean().sqrt()), float(row.abs().max()))


def _trace_mix(layer_id, *, hidden, alpha, beta, q_tilde, s, reading, core, gated, out) -> None:
    """One linear layer's stages, row 0, when SGLANG_AFD_SELFCHECK is set.

    Layer 0's output agrees with the model to five figures and layer 1's is 5% out, and on a first
    prefill the recurrent state and the convolution ring are zero for both -- so nothing this side
    CARRIES differs between them and the only difference is the input. These are the stages
    between that input and the output, so whichever one first stops looking like its neighbour on
    the other layer is where to look.
    """
    import os

    if not os.environ.get("SGLANG_AFD_SELFCHECK"):
        return
    key = f"mix-L{layer_id}"
    if _STEP_SEEN.get(key, 0) >= 1:
        return
    _STEP_SEEN[key] = 1

    rows = hidden.shape[0]

    def one(t):
        """Row 0, whatever axis layout the tensor happens to be in.

        `core` is (rows, heads*dim) and `gated` is (rows*heads, dim) -- the z-gated norm reshapes
        for its own kernel -- so `t[0]` is a whole row of one and a single HEAD of the other. The
        first version of this line used `t[0]` for both and printed 0.0107 beside 1.4261 as though
        they were comparable. Fourth time in this search. Reshaped to (rows, -1) first so the axis
        cannot vary between the things on one line.
        """
        row = t.reshape(rows, -1)[0].float()
        return f"|{float(row.norm()):.5g}|"

    a, b = alpha[0].float(), beta[0].float()
    logger.info(
        "afd mix: layer %s -- hidden %s | alpha mean %.5f min %.5f max %.5f | beta mean %.5f | "
        "q~ %s | reading %s | s %s | core %s | gated %s | out %s",
        layer_id, one(hidden), float(a.mean()), float(a.min()), float(a.max()), float(b.mean()),
        one(q_tilde), one(reading), one(s), one(core), one(gated), one(out),
    )


_STEP_SEEN = {}


_HEAD_SEEN = {}


def _trace_head(group, request_ids, residual, *, attn_output, gated, projected) -> None:
    """The three things only a MIDDLE span does, against the stream they join.

    The residual leaving the first middle span is 34.7 where the model's own is 112.0 -- it fell by
    two thirds across a group, and adding vectors cannot halve a norm unless what is added points
    against what it is added to. So the number that matters is not any tensor's magnitude, it is
    the COSINE between what this span adds and the residual it adds to. A contribution roughly
    orthogonal to the stream is ordinary; one that is negative is the answer.

    Three candidates, and this separates them: the output gate applied here to an attention output
    computed on the host, `W_o` on that gated output, and the residual taken from this side's own
    table rather than carried in the call. Printed in that order so the first line where the sign
    turns names the stage.
    """
    import os

    if not os.environ.get("SGLANG_AFD_SELFCHECK"):
        return
    seen = _HEAD_SEEN.get(group, 0)
    if seen >= 2:
        return
    _HEAD_SEEN[group] = seen + 1

    stream = residual[0].float()

    def against(t):
        row = t[0].float()
        if row.numel() != stream.numel():
            return f"|{float(row.norm()):.5g}| (width {row.numel()}, not the stream's)"
        cos = torch.nn.functional.cosine_similarity(row, stream, dim=0)
        return f"|{float(row.norm()):.5g}| cos {float(cos):+.4f}"

    # the request id, so this line can be matched against `_trace_head`'s. Without it the
    # residual taken here (18.8 at group 3) and the residual kept after the prologue (76.8) could
    # not be told apart as "the table handed back the wrong value" from "these are two different
    # requests", and that difference is the whole question.
    logger.info(
        "afd head: request %s group %s call %s -- residual |%.5g| rows %s | attn %s | gated %s "
        "| projected %s",
        int(request_ids[0]), group, seen, float(stream.norm()), residual.shape[0],
        against(attn_output), against(gated), against(projected),
    )


_RESIDUAL_SEEN = {}


def _trace_residual(runner, request_ids, residual) -> None:
    """The residual stream leaving each span, when SGLANG_AFD_SELFCHECK is set. Off otherwise.

    This is the pool's half of a localisation the arrangement still needs. Its final hidden state
    sits at cosine 0.451 to the colocated model's on the same single-token prompt, with almost the
    same norm -- so it is not empty and not rescaled, it has turned. A turn that large arrives
    somewhere, and the residual at each span boundary is the sequence that says where.

    A span boundary is a LAYER boundary: the value kept here entering group g is the model's own
    hidden state at the group's first layer, so this sequence is directly comparable to a
    colocated forward's per-layer hidden states. Norms first because a stage that drops or doubles
    a contribution shows up in the magnitude before it shows up anywhere else.
    """
    import os

    if not os.environ.get("SGLANG_AFD_SELFCHECK"):
        return
    # ROW 0 ONLY, and one line a CALL. The first version incremented per row, so a 122-token
    # prefill logged 122 lines from a single span and they were read as 122 span boundaries --
    # a sequence across TOKEN POSITIONS compared against a reference across LAYERS. Both were
    # correct measurements of different quantities, and the comparison was of neither.
    rid = int(request_ids[0])
    step = _RESIDUAL_SEEN.get(rid, 0)
    if step >= int(os.environ.get("SGLANG_AFD_SELFCHECK", "20")):
        return
    _RESIDUAL_SEEN[rid] = step + 1
    row = residual[0].float()
    logger.info(
        "afd residual: request %s span %s -- rows %s wide %s |%.5g| rms %.5g max %.5g",
        rid, step, residual.shape[0], row.numel(),
        float(row.norm()), float(row.pow(2).mean().sqrt()), float(row.abs().max()),
    )


class SpanRunner:
    """Runs one group's span on the pool's own weights.

    Stateless per call except for what a request remembers: the residual entering the group, and
    the two recurrent states of each linear layer. Those are keyed by request and live here for
    the length of a generation.
    """

    def __init__(self, model, states, *, layer_types: list[str], query_shift: int) -> None:
        self.model = model
        self.states = states
        self.query_shift = query_shift
        ordered = group_layers(layer_types)
        self.spans = {s[0]: s for s in ordered}
        # which attention each span feeds. The query projection runs on this side, so the span has
        # to know whose weights to project with -- and getting it from the NEXT span's head rather
        # than from `layer_id + 1` is what keeps this right on a model whose attentions are not
        # evenly spaced.
        self.next_attention = {
            span[-1]: nxt[0] for span, nxt in zip(ordered, ordered[1:])
        }
        self._residual: dict[int, torch.Tensor] = {}
        # the output gate for the query already sent. It is applied to the attention's output
        # before W_o, and W_o is this side's first act on the next call, so it never travels.
        self._gate: dict[int, torch.Tensor] = {}
        self._lock = threading.Lock()
        # the callbacks a span uses to reach the history, per THREAD: several groups can be in
        # flight at once and an instance attribute would cross one span's callback into another's
        self._local = threading.local()
        self.served = 0

    # -- what a request remembers between spans ---------------------------------------------

    def _take_residual(self, request_ids, rows: int, like: torch.Tensor) -> torch.Tensor:
        """The residual entering this group, ONE ROW A ROW, in the caller's row order.

        A request making its first call has none, which is the headless span at the bottom of the
        stack; anything else means a group ran out of order and the answer would be a correct
        forward pass over the wrong history.

        Per ROW rather than per request, which is what this was and what it cost. `request_ids` has
        one entry a row: a decode batch is one row a request, so keeping a single row under the
        request id was right there and only there. A 122-token prefill is 122 rows of ONE request,
        each row overwrote the last, and what survived was the FINAL token's residual -- handed
        back to all 122 positions on the next call. Measured at the boundary: 76.76 written
        leaving the prologue, 18.833 read entering the next group, same request, same boundary.
        Every position in the prompt then ran the rest of the stack on the last token's state.
        """
        with self._lock:
            rows_out = []
            for rid, _start, count in _runs_of(request_ids):
                held = self._residual.get(rid)
                if held is None:
                    raise RuntimeError(
                        f"request {rid} has no residual on the pool, so this is not its first span "
                        f"and the group before it never ran here. The forward would be correct "
                        f"arithmetic over a history this request does not have."
                    )
                if held.shape[0] != count:
                    raise RuntimeError(
                        f"request {rid} kept {held.shape[0]} row(s) of residual and is asking for "
                        f"{count} back. A chunk's rows are its own consecutive tokens and each has "
                        f"its own residual; a mismatch here would broadcast one token's state over "
                        f"the others and stay fluent."
                    )
                rows_out.append(held)
        return torch.cat(rows_out, dim=0).to(like.dtype)

    def _keep_residual(self, request_ids, residual: torch.Tensor) -> None:
        """Keep every row, not one a request. See `_take_residual` for what the one row cost."""
        with self._lock:
            for rid, start, count in _runs_of(request_ids):
                self._residual[rid] = residual[start : start + count].clone()
        _trace_residual(self, request_ids, residual)

    def _gated(self, request_ids, attn_output: torch.Tensor) -> torch.Tensor:
        """Apply the output gate this side computed with the query the host swept with.

        The model multiplies the attention's output by `sigmoid(gate)` before the output
        projection. The gate comes out of the same projection as the query, so this side has it,
        and the output projection is this side's first act on the next call -- sending it would be
        sending a value back to the machine that produced it and then receiving it again.
        """
        gate = self._take_gate(request_ids, attn_output)
        flat = attn_output.reshape(attn_output.shape[0], -1)
        if flat.shape != gate.shape:
            raise RuntimeError(
                f"an attention output of {tuple(attn_output.shape)} against a gate of "
                f"{tuple(gate.shape)}. The caller's two attention paths -- fused for a prefill, "
                f"split for a decode -- have to put ONE shape on the wire; they do not agree on "
                f"their own, and the mismatch arrives here rather than where it is made."
            )
        return flat * torch.sigmoid(gate)

    def _keep_gate(self, request_ids, gate: torch.Tensor) -> None:
        """Kept FLAT and per ROW.

        `forward_prepare_native` hands the gate back head-shaped on this model, and the attention
        output it multiplies arrives flat, so the reshape happens once here rather than at every
        use. Per row for the reason in `_take_residual`.
        """
        flat = gate.reshape(gate.shape[0], -1)
        with self._lock:
            for rid, start, count in _runs_of(request_ids):
                self._gate[rid] = flat[start : start + count].clone()

    def _take_gate(self, request_ids, like: torch.Tensor) -> torch.Tensor:
        """The gate for the query already sent, one row A ROW, in the caller's row order.

        Same shape of table as the residual and the same correction: a prefill chunk's rows are one
        request's own tokens, and a gate kept per request would apply the last token's gate to all
        of them.
        """
        with self._lock:
            rows = []
            for rid, _start, count in _runs_of(request_ids):
                held = self._gate.get(rid)
                if held is None:
                    raise RuntimeError(
                        f"request {rid} sent back an attention output for a query this pool never "
                        f"projected. The gate is applied to that output before W_o, so there is no "
                        f"way to finish the layer -- and skipping it would be a correct-looking "
                        f"model with one nonlinearity missing."
                    )
                if held.shape[0] != count:
                    raise RuntimeError(
                        f"request {rid} kept {held.shape[0]} gate row(s) and is asking for {count}."
                    )
                rows.append(held)
        return torch.cat(rows, dim=0).to(like.dtype)

    def release(self, request_id: int) -> int:
        """Forget one request. Returns how many recurrent slots were cleared."""
        with self._lock:
            self._residual.pop(int(request_id), None)
            self._gate.pop(int(request_id), None)
        return self.states.release(int(request_id))

    # -- the span ---------------------------------------------------------------------------

    def _span_of(self, group: int) -> tuple[int, ...]:
        span = self.spans.get(group)
        if span is None:
            raise KeyError(
                f"layer {group} does not start a span. The spans on this model start at "
                f"{sorted(self.spans)}; a frame naming any other layer was built against a "
                f"different layout than the pool loaded."
            )
        return span

    def run(self, request_ids, group: int, attn_output: torch.Tensor, positions,
            on_query=None):
        """From the host's attention output to the next attention's query, key and value.

        `on_query` is called with `q` the moment it exists, which is one feed-forward before the
        key and value do. That is the head start: the host sweeps its cache with a query it did
        not have to project, while this side spends the last feed-forward.

        This is the middle span, and there are fifteen of them on this model. The two at the ends
        of the stack are shaped differently and have their own methods -- an `if` here for each of
        them would put three arrangements in one function and hide which one a measurement ran.
        """
        layers = self.model.model.layers
        span = self._span_of(group)
        head, rest = span[0], span[1:]
        if head < 0 or not rest:
            raise ValueError(
                f"the span at {group} is an end of the stack, not a middle span: use "
                f"`run_prologue` for the layers below the first attention and `run_epilogue` for "
                f"the tail. They return different things and conflating them would report one "
                f"arrangement's cost under another's name."
            )
        residual = self._take_residual(request_ids, attn_output.shape[0], attn_output)

        # the head layer's tail: the output gate, the output projection, the feed-forward. The
        # attention itself ran on the host, which is the whole point of the cut -- and the gate
        # was computed here on the previous call, so what comes back over the wire is the bare
        # attention output and the nonlinearity is applied on this side
        gated = self._gated(request_ids, attn_output)
        attn_out, _ = layers[head].o_proj(gated)
        _trace_head(group, request_ids, residual,
                    attn_output=attn_output, gated=gated, projected=attn_out)
        hidden, residual = _add_and_norm(
            layers[head].post_attention_layernorm, attn_out, residual)
        hidden = layers[head].mlp(hidden)

        hidden, residual = self._linear_run(request_ids, rest[:-1], hidden, residual)
        return self._finish(request_ids, rest[-1], hidden, residual, on_query, positions)

    def run_prologue(self, request_ids, embedded: torch.Tensor, positions, on_query=None):
        """The layers below the first attention, fed by the embedding rather than by a `W_o`.

        On this model that is layers 0, 1 and 2, and it is where a request's residual starts.

        It projects a query like any other span. The query belongs to layer 3, and layer 3 is not
        the shift's exempt layer -- three layers sit beneath it, so under shift 1 it reads `h_2`
        and there is a residual for it to read. Only layer 0 is exempt, and layer 0 is inside this
        span rather than at its end.
        """
        span = self._span_of(-1)[1:]
        _trace_step("embedding", -1, embedded)
        hidden, residual = self._linear_run(request_ids, span[:-1], embedded, None)
        return self._finish(request_ids, span[-1], hidden, residual, on_query, positions)

    def _finish(self, request_ids, layer_id: int, hidden, residual, on_query, positions):
        """The span's last linear layer, and the next attention's projections.

        Where the query projection sits is not a detail -- it is the only place it can go. The
        query's source is `h_(l+3)`, which this layer produces, and the projection has to happen on
        the side that produces it or the source crosses the wire before it can be used. Run HERE,
        between the last linear attention and the last feed-forward, three things follow:

            the host receives a query it can sweep with immediately, rather than a hidden state it
            must project first -- and a projection on the host is a GEMV on a machine that is
            weight-read bound at decode, which is the finding this arrangement already confirmed

            the send is covered by the last feed-forward, 298 us, so the host's sweep starts one
            feed-forward earlier in the group. The sweep is the variable-latency stage, up to
            2397 us at 128k, and starting it early is the only lever this side has on it

            the host holds no projection weights at all, which is 3.12 GiB of a 32 GiB card given
            back to the KV cache -- 70,689 tokens a request to 83,489, across a break-even the
            arrangement otherwise sits exactly on

        The key and value are projected after the feed-forward, because they read `x_l` and `x_l`
        is not complete until it runs. That is not a problem: they are off the critical path. This
        step's join uses key and value the caller already has, and the cache only has to hold them
        by the NEXT step.

        The output gate stays here. It is applied to the attention's output before `W_o`, and `W_o`
        is the pool's first act on the next call, so sending the gate would be sending a value back
        to the machine that computed it.
        """
        layers = self.model.model.layers
        last = layers[layer_id]
        hidden, residual = _add_and_norm(last.input_layernorm, hidden, residual)
        hidden = self._linear_attention(last.linear_attn, request_ids, layer_id, hidden)
        hidden, read_point = _add_and_norm(last.post_attention_layernorm, hidden, residual)

        # the projections live on the DECODER LAYER, not on a submodule: `self_attention` is a
        # method that runs them. Naming a submodule that does not exist fails at the first token,
        # not at install, which is where this cost a deployment round.
        nxt = layers[self.next_attention[layer_id]]
        if self.query_shift == 1:
            # the read point, one feed-forward before the group's output. The send is covered by
            # that feed-forward, which is the window.
            q, _, _, gate = nxt.forward_prepare_native(
                positions, nxt.input_layernorm(read_point))
            if on_query is not None:
                on_query(q)

        hidden = last.mlp(hidden)
        # the next group's input residual is this one's output, and the pool is the one that has
        # it. It stays here rather than travelling both ways for a value neither end changed.
        x = read_point + hidden
        normed = nxt.input_layernorm(x)
        if self.query_shift == 0:
            # STANDARD WIRING, and the control the shifted read has to be measured against. The
            # query comes from the same tensor as the key and value, so there is nothing to send
            # early and the window is shut -- which is the point: a run at 0 is asking what the
            # shift buys, and it can only answer if the read actually moved.
            q, _, _, gate = nxt.forward_prepare_native(positions, normed)
            if on_query is not None:
                on_query(q)
        self._keep_residual(request_ids, x)
        self._keep_gate(request_ids, gate)
        _, k, v, _ = nxt.forward_prepare_native(positions, normed)
        self.served += 1
        return q, k, v

    def run_epilogue(self, request_ids, group: int, o: torch.Tensor):
        """The last attention's tail: its output projection, its feed-forward, and the final norm.

        Returns the normalised hidden state rather than logits, so that whether the language-model
        head runs here or on the host stays a separate decision -- it is a 5120 x vocab read, which
        is the largest single weight in the model and the one most worth measuring on its own.
        """
        layers = self.model.model.layers
        head = self._span_of(group)[0]
        residual = self._take_residual(request_ids, o.shape[0], o)
        attn_out, _ = layers[head].o_proj(self._gated(request_ids, o))
        hidden, residual = _add_and_norm(
            layers[head].post_attention_layernorm, attn_out, residual)
        hidden = layers[head].mlp(hidden)
        # NOT normalised here. sglang's own forward applies `self.norm` after its layer loop, and
        # the closing head hands it residual=None, so it takes the `self.norm(hidden_states)`
        # branch on whatever this returns. Normalising here as well applied the final RMSNorm
        # TWICE: with a learned weight w that is w squared elementwise, and this checkpoint's
        # `model.language_model.norm.weight` runs from -0.285 to 1.711, so squaring flips the sign
        # of every negative channel and rescales the rest between 0.08x and 2.93x. What this
        # returns is the residual stream itself, which is what the model's own last layer returns.
        self.served += 1
        return residual + hidden

    def _linear_run(self, request_ids, layer_ids, hidden, residual):
        """Whole linear-attention layers, back to back, with nothing between them to interrupt."""
        layers = self.model.model.layers
        for layer_id in layer_ids:
            layer = layers[layer_id]
            hidden, residual = _add_and_norm(layer.input_layernorm, hidden, residual)
            hidden = self._linear_attention(layer.linear_attn, request_ids, layer_id, hidden)
            hidden, residual = _add_and_norm(layer.post_attention_layernorm, hidden, residual)
            _trace_step("pre-mlp", layer_id, residual)
            hidden = layer.mlp(hidden)
            # the feed-forward's own output, never compared until now. Layer 0's x + attn is exact
            # and layer 1's is 6.3% high, and layer 1's INPUT is layer 0's residual plus THIS. An
            # attention that is perfectly right computes a wrong answer from a wrong input, so the
            # feed-forward between them has to be ruled in or out before the attention is blamed.
            _trace_step("mlp-out", layer_id, hidden)
            _trace_step("post-mlp", layer_id, residual + hidden)
        return hidden, residual

    def _linear_attention(self, attn, request_ids, layer_id: int, hidden: torch.Tensor):
        """One linear-attention layer: everything but the history, which is the host's.

        The weights are here and the recurrent state is not, so the layer is split at the one
        place the recurrence allows. `linear_history` proves the identity and
        `benchmark/afd/gdn_split.py` measures it against the fused kernel; what happens here is
        the arrangement of it across two machines:

            here        input projection, gates, normalisation, the convolution -- and the QUERY
                        COEFFICIENT q~ = q - beta (k.q) k, which folds the key's correction into
                        the query so the far end contracts the state ONCE
            over there  r = S q~, one contraction of a state this end does not hold
            here        core = alpha r + beta (k.q) v, and the value never crossed the wire
            over there  the key, the value and the gates, deferred, to advance the state

        The decay is applied HERE, to what comes back. Sending it would be sending a per-head
        scalar across so it could be multiplied and sent back, and the reading is the same shape
        either way.
        """
        from sglang.srt.afd.linear_history import (
            expand_to_value_heads,
            gates,
            normalise,
            query_coefficient,
        )

        ask = getattr(self._local, "ask_host", None)
        if ask is None:
            raise RuntimeError(
                f"layer {layer_id} has no way to reach the history. The recurrent state lives on "
                f"the caller's side under this cut, so a span with no callback would have to "
                f"either invent a state or hold one here -- the first is wrong and the second is "
                f"the arrangement this cut replaced."
            )
        # sglang's own names, not transformers'. The two libraries split this projection
        # differently -- one fused `in_proj_qkvz` here against four separate ones there -- and
        # writing the other library's names produced an AttributeError at the first token, which
        # is the same shape of mistake as the state layout being transposed between them.
        qkvz, _ = attn.in_proj_qkvz(hidden)
        ba, _ = attn.in_proj_ba(hidden)
        query, key, value, z, b, a = attn.fix_query_key_value_ordering(qkvz, ba)
        rows = hidden.shape[0]
        flat = [t.reshape(rows, -1) for t in (query, key, value)]
        mixed = self._convolve(attn, torch.cat(flat, dim=-1), request_ids, layer_id)

        width = flat[0].shape[-1]
        q = mixed[:, :width].reshape(rows, attn.num_k_heads // attn.attn_tp_size,
                                     attn.head_k_dim)
        k = mixed[:, width : 2 * width].reshape(rows, attn.num_k_heads // attn.attn_tp_size,
                                                attn.head_k_dim)
        v = mixed[:, 2 * width :].reshape(rows, attn.num_v_heads // attn.attn_tp_size,
                                          attn.head_v_dim)
        alpha, beta = gates(a, b, attn.A_log, attn.dt_bias)
        q, k = normalise(q, k, scale=attn.head_k_dim ** -0.5)
        heads = attn.num_v_heads // attn.attn_tp_size
        q, k = expand_to_value_heads(q, heads), expand_to_value_heads(k, heads)
        q_tilde, s = query_coefficient(q, k, beta)

        # One call for the whole bus when every run is one row -- that is decode, and the rows
        # are independent. A run longer than one row is a request's own tokens in order, and the
        # far end has to advance its state between them, so those go one call a run.
        #
        # Which means a mixed bus pays for its prefill riders in ROUND TRIPS rather than in span
        # time: the feed-forwards and the projections, which are 74% of the cost, still run once
        # for everybody. That is the whole reason the leftover tokens are worth carrying.
        # A run longer than one row is a request's own tokens in order, and the far end has to
        # advance its state BETWEEN them -- so those carry the key, the value and the gates in the
        # same message. A deferred update cannot serve them: token n reads what token n-1 wrote,
        # and the deferral has not arrived. Single-row runs keep the split, which is the one
        # contraction the query coefficient bought.
        step = (k, v, alpha, beta)
        reading = ask(layer_id, request_ids, q_tilde, step=step)
        # The FIRST token of a request's FIRST chunk reads a state nothing has written to, so this
        # is exactly zero or the slot carries someone else's history. One number, and it decides
        # between "the recurrence is wrong" and "the recurrence is right and started wrong".
        #
        # Gated on both conditions. The first version logged row 0 of any single-row call, and a
        # single-row call is usually a DECODE step, whose state is legitimately non-zero -- it read
        # 6.21 and decided nothing. Keyed per layer as well: one key for all of them spent the
        # whole budget on layer 0.
        if len(request_ids) > 1 and int(request_ids[0]) not in self._residual:
            _trace_step(f"first-read-L{layer_id}", layer_id, reading[0:1].reshape(1, -1))
        core = alpha.unsqueeze(-1) * reading + s.unsqueeze(-1) * v.float()

        # the state's own copy of the key, deferred: it only has to be applied before the NEXT
        # step, which is the argument OP_APPEND already makes for a KV cache
        # single-row riders only; a multi-row one advanced inside its scan and applying the same
        # tokens again would fold each of them into the history twice
        defer = getattr(self._local, "defer_update", None)
        if defer is not None:
            defer(layer_id, request_ids, k, v, alpha, beta)

        core = core.reshape(rows, -1).to(hidden.dtype)
        gated_core = attn.norm(core.reshape(-1, attn.head_v_dim),
                               z.reshape(-1, attn.head_v_dim))
        out, _ = attn.out_proj(gated_core.reshape(rows, -1))
        # the stages of one layer, row 0 of a first prefill, so layer 0 and layer 1 can be read
        # side by side. Layer 0 agrees with the model exactly and layer 1 is 5% out, and on a
        # first prefill the state and the ring are zero for BOTH -- so nothing this side carries
        # differs between them and only the input does.
        if len(request_ids) > 1 and int(request_ids[0]) not in self._residual:
            _trace_step(f"attn-in-L{layer_id}", layer_id, hidden)
            _trace_mix(layer_id, hidden=hidden, alpha=alpha, beta=beta, q_tilde=q_tilde, s=s,
                       reading=reading, core=core, gated=gated_core, out=out)
        return out

    def _convolve(self, attn, qkv: torch.Tensor, request_ids, layer_id: int) -> torch.Tensor:
        """The short convolution, against a ring this side keeps.

        The kernel is depthwise and causal, K = 4, each channel filtered on its own:

            out[c] = silu( bias[c] + sum over t of w[c,t] * x[c, n-K+1+t] )

        so the ring holds K entries, not K-1: the window IS `[ring[1:], x_new]`, which is K wide
        against a K-wide weight. An earlier version here allocated K-1 and built a window of three
        against four -- caught by writing the formula down rather than by a test, because the span
        path has none yet.

        The ring is this side's own last K projections -- values the pool computed itself -- so
        keeping it here costs no round trip and no history. It is 80 KiB a layer a request against
        the recurrent state's 3.00 MiB, 2% of what a request remembers, and it is the only
        per-request thing this side holds.
        """
        from sglang.srt.afd.linear_state import LinearStates  # noqa: F401 -- slot table only

        # sglang holds this as (channels, 1, taps) -- the model's own code squeezes it the same
        # way before handing it to the backend. Squeezed ONCE here so neither path below can
        # broadcast against the extra axis, which is what it does rather than raising.
        weight = attn.conv1d.weight.squeeze(1)
        ring = self.states.conv_buffer(
            layer_id, width=qkv.shape[-1], taps=weight.shape[-1], dtype=qkv.dtype)
        runs = _runs(request_ids)
        slots = [self.states.slot_of(r) for r, _, _ in runs]
        self.states.note_touched(slots, ("conv", layer_id))
        if all(n == 1 for _, _, n in runs):
            # decode: one row a request, so the ring update is a scatter and every row is
            # independent of every other
            index = torch.tensor(slots, device=qkv.device, dtype=torch.long)
            held = ring.index_select(0, index)
            window = torch.cat([held[..., 1:], qkv.unsqueeze(-1)], dim=-1)
            ring.index_copy_(0, index, window)
            out = (window * weight).sum(-1)
            return torch.nn.functional.silu(out)

        # prefill, or a bus carrying both: a request's rows are its own tokens IN ORDER, and each
        # is convolved against its predecessors rather than against the pre-chunk ring. Scattering
        # them would write the same slot several times, which is undefined, and would filter every
        # token with the same history.
        from sglang.srt.afd.linear_history import prefill_convolve

        pieces = []
        for slot, (_, start, count) in zip(slots, runs):
            got, tail = prefill_convolve(ring[slot], qkv[start : start + count], weight)
            ring[slot] = tail
            pieces.append(got)
        return torch.cat(pieces, dim=0)


    def seed(self, request_id: int, residual: torch.Tensor) -> None:
        """Give a request its first residual: the embedding, for the headless span.

        Stored with a ROW axis even for one row, because that is what the table holds now -- a
        prefill chunk keeps one row a token. A bare vector seeded here would be read back as a
        chunk of `hidden_size` rows, and the refusal it earns names a row count rather than a
        shape, so it is normalised at the door.
        """
        if residual.dim() == 1:
            residual = residual.unsqueeze(0)
        with self._lock:
            self._residual[int(request_id)] = residual.clone()

    def report(self) -> dict:
        with self._lock:
            held = len(self._residual)
        return {"spans_served": self.served, "residuals_held": held,
                "groups": sorted(self.spans), **self.states.report()}
