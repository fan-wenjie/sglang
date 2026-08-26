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
import os
import threading
import time

import torch

from sglang.srt.afd.linear_runner import LinearRunner

logger = logging.getLogger(__name__)


_LAYERS: dict = {}  # slot -> [count, attention+callback seconds, mlp seconds]


def _layer_parts(slot: int, attn_s: float, mlp_s: float) -> None:
    """Per POSITION in the span, not per layer id: the question is whether the cost is spread
    across the three or sits in one, and a span always visits the same three positions.
    """
    row = _LAYERS.setdefault(slot, [0, 0.0, 0.0])
    row[0] += 1
    row[1] += attn_s
    row[2] += mlp_s
    if slot or row[0] % 200:
        return
    parts = " | ".join(
        f"#{k} attn+cb {1e3 * v[1] / v[0]:.3f} mlp {1e3 * v[2] / v[0]:.3f}"
        for k, v in sorted(_LAYERS.items())
    )
    logger.info("afd span layers: %s visits -- %s (ms each)", row[0], parts)


_PARTS = [0, 0.0, 0.0, 0.0]  # count, head mlp, three linear layers, finish


# What shift 1 serves is the full push arrangement, priced on both axes before it became
# the only implementation: wall clock -21% against the blocking baseline's real
# distribution over 24 order-rotated flights (the 13-second bimodal tail collapsed to a
# +-1 s band), approximation +0.37% bits per byte under the own-norm serving arm against a
# pre-registered kill of +2.68%. A measurement ladder (SGLANG_AFD_QS_RUNG) once selected
# the mechanisms one at a time; its verdicts live in the design record and the scaffolding
# is gone -- the query_shift flag, resolved through the checkpoint, is the one switch.

_LINEAR = [0.0, 0.0, 0.0, 0]


def _linear_parts(project_s: float, callback_s: float, finish_s: float) -> None:
    """One linear layer's three parts on the POOL, every 2000.

    `project` is the fused projection, the ordering split, the pack and the gates. `callback` is
    the whole of the mix -- the send, the far end's work, and the wait. `finish` is the z-gated
    norm and the output projection.

    `linear_runner` has a counter of the same shape and the group cut never reaches it: the span
    runs its own `_linear_attention`, so under this arrangement that one has always read zero.
    """
    for i, v in enumerate((project_s, callback_s, finish_s)):
        _LINEAR[i] += v
    _LINEAR[3] += 1
    if _LINEAR[3] % 2000:
        return
    n = _LINEAR[3]
    logger.info(
        "afd pool linear: %d layers -- project %.3f, callback %.3f, finish %.3f ms",
        n,
        *(x * 1e3 / n for x in _LINEAR[:3]),
    )


def _span_parts(mlp_s: float, linear_s: float, finish_s: float) -> None:
    """Every 200 spans, the three pieces of a span's own time.

    `linear` includes the three callbacks, so subtracting the pool's callback total from it says
    whether the linear layers are slow or whether they are waiting.
    """
    _PARTS[0] += 1
    _PARTS[1] += mlp_s
    _PARTS[2] += linear_s
    _PARTS[3] += finish_s
    if _PARTS[0] % 200:
        return
    n = _PARTS[0]
    logger.info(
        "afd span parts: %s spans -- head mlp %.3f ms, three linear layers %.3f ms "
        "(callbacks included), finish %.3f ms",
        n,
        1e3 * _PARTS[1] / n,
        1e3 * _PARTS[2] / n,
        1e3 * _PARTS[3] / n,
    )


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


def _project_prefix(linear, x: torch.Tensor, width: int, what: str) -> torch.Tensor:
    """The first `width` outputs of a fused projection, without computing the rest.

    A fused weight's outputs are its ROWS, so a leading group of them is a prefix of the weight
    and taking it is a smaller matmul rather than a different one. A view, not a copy:
    `weight[:n]` and `weight[:n].contiguous()` measured the same, so the copy would be a byte
    cost for nothing.

    Refused rather than approximated when the layer is quantised. A quantised linear is not
    `weight @ x`, and slicing its rows would silently compute something else -- which on this path
    means a query coefficient that is subtly wrong and a model that still produces text.
    """
    method = getattr(linear, "quant_method", None)
    if method is not None and type(method).__name__ != "UnquantizedLinearMethod":
        raise NotImplementedError(
            f"the early view slices this projection's weight to skip {what}, and the layer is "
            f"served by {type(method).__name__}. A quantised projection is not a matrix multiply "
            f"against `.weight`, so the slice would compute something else and say nothing. Run "
            f"the arrangement unquantised, or teach this function the method's own slicing."
        )
    return torch.nn.functional.linear(x, linear.weight[:width])


def _project_suffix(linear, x: torch.Tensor, skip: int, what: str) -> torch.Tensor:
    """Everything but the first `skip` outputs of a fused projection.

    The mirror of `_project_prefix`, with the same quantisation refusal for the same reason:
    a quantised linear is not `weight @ x`, and slicing its rows would compute something else.
    """
    method = getattr(linear, "quant_method", None)
    if method is not None and type(method).__name__ != "UnquantizedLinearMethod":
        raise NotImplementedError(
            f"the mix slices this projection's weight to skip {what}, and the layer is "
            f"served by {type(method).__name__}. Run the arrangement unquantised, or teach "
            f"this function the method's own slicing."
        )
    return torch.nn.functional.linear(x, linear.weight[skip:])


def _project_qk(attn, normed: torch.Tensor, key_width: int) -> torch.Tensor:
    """`[q | k]` from the fused `[q | k | v | z]` projection, without computing v or z.

    The fused weight is laid out `[q | k | v | z]` at widths
    `[key_dim, key_dim, value_dim, value_dim]`, so q and k are a PREFIX of its rows and taking
    them is a smaller matmul rather than a different one. Measured on this model's shapes, one
    row: 112 us fused against 16 us sliced, because the output is 25% of the width and this shape
    is bound by writing it.

    A view, not a copy: `weight[:n]` and `weight[:n].contiguous()` measured the same, so the copy
    would be a byte cost for nothing.

    Refused rather than approximated when the layer is quantised. A quantised linear is not
    `weight @ x` and slicing its rows would silently compute something else -- which on this path
    would be a query coefficient that is subtly wrong and a model that still produces text.
    """
    return _project_prefix(attn.in_proj_qkvz, normed, 2 * key_width, "v and z")


class SpanRunner:
    """Runs one group's span on the pool's own weights.

    Stateless per call except for what a request remembers: the residual entering the group, and
    the two recurrent states of each linear layer. Those are keyed by request and live here for
    the length of a generation.
    """

    # What this runner adds to the pool's HELLO word. The pool reads it off whatever runner it was
    # given rather than knowing the number, so bit 8 -- "serves whole spans, four layers a call" --
    # is stated by the thing that serves them. A host asks for the bit before its first token, so
    # a runner without one turns into "this pool does not serve whole spans" at connect time,
    # which is where this constant went missing once.
    capability_bit = 8

    def __init__(
        self,
        model,
        states,
        *,
        layer_types: list[str],
        query_shift: int,
    ) -> None:
        self.model = model
        self.states = states
        # The linear layer's own arithmetic, which is standard AFD's rather than this arm's. Held
        # as a collaborator instead of inherited: a span is four of its calls in a row plus the
        # feed-forwards between them, and a pool serving one layer a call runs exactly the same
        # object.
        # When the caller holds the convolution ring, this pool writes nothing per request and a
        # request can be answered by any pool. It is the last piece of per-request state on this
        # side; the recurrent states themselves already live on the caller.
        self.linear = LinearRunner(model, states)
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
        # The SAME thread-local the collaborator reads. The pool sets `ask_host` and
        # `defer_update` on the runner it was given, and the recurrence reads them from inside the
        # linear layer; two objects would mean the callbacks were installed on one and looked for
        # on the other, which raises nothing and simply never calls back.
        self._local = self.linear._local
        # The early projection reads the previous layer's post-attention norm OUTPUT, then
        # re-scales it into THIS layer's own reading: both norms see the same residual, so they
        # share the same rms and differ only in the learned per-channel weight -- one ratio a
        # layer, computed here, applies the own norm at the price of a multiply. Measured
        # offline: the reused reading alone cost +1.46 points of bits per byte (1.83 against
        # 0.37), because a mis-normed raw q column compounds through the convolution history.
        # A checkpoint whose weights trip the near-zero guard is refused at startup, loudly,
        # rather than served with silently different arithmetic.
        layers = model.model.layers

        def scale_of(norm) -> torch.Tensor:
            # the norm's EFFECTIVE per-channel scale, not its stored weight. These layers are
            # GemmaRMSNorm -- `x/rms(x) * (1 + weight)` -- and the stored weights sit near -1
            # while the scales sit near zero. The first wiring passed the weights themselves;
            # the ratio came out the wrong sign and the deployment spoke garbage from its
            # first token, undetected because bits-per-byte only exercises prefill and the
            # latency flights compared arm against arm. In fp32 because `1 + w` near w = -1
            # is exactly where bfloat16 runs out of digits; cast back where it multiplies.
            return 1.0 + norm.weight.float()

        self._norm_ratio = {}
        fell_back = 0
        if query_shift:
            try:
                from sglang.srt.afd_query_shift.pool_cook import norm_ratio
            except ImportError as e:
                raise RuntimeError(
                    f"query shift {query_shift} needs the early-read package "
                    f"(sglang.srt.afd_query_shift), which this tree does not carry. "
                    f"Shift 0 -- the span cut, serially -- is the arrangement this "
                    f"tree serves; the early read lives on the derived branch."
                ) from e

            for index in range(1, len(layers)):
                attn_mod = getattr(layers[index], "linear_attn", None)
                if attn_mod is None:
                    continue
                try:
                    self._norm_ratio[index] = norm_ratio(
                        scale_of(layers[index].input_layernorm),
                        scale_of(layers[index - 1].post_attention_layernorm),
                    ).to(layers[index].input_layernorm.weight.dtype)
                except ValueError:
                    # a checkpoint whose scales DO have near-zero channels earns the guard's
                    # refusal, and such a layer runs its own full norm on the raw residual
                    self._norm_ratio[index] = None
                    fell_back += 1
        # the same identity for the ATTENTION's early query: `_finish` normalises the read
        # point with the next head's own input norm, and the last linear layer's post-attention
        # norm already read the same residual -- one ratio replaces one full RMSNorm a span
        self._attn_norm_ratio = {}
        if query_shift:
            from sglang.srt.afd_query_shift.pool_cook import norm_ratio

            for last_id, head_id in self.next_attention.items():
                try:
                    self._attn_norm_ratio[int(last_id)] = norm_ratio(
                        scale_of(layers[head_id].input_layernorm),
                        scale_of(layers[last_id].post_attention_layernorm),
                    ).to(layers[head_id].input_layernorm.weight.dtype)
                except ValueError:
                    self._attn_norm_ratio[int(last_id)] = None
                    fell_back += 1
            if fell_back:
                logger.info(
                    "afd: %d layer(s) run their own full norm -- near-zero channels in the "
                    "reused scale make the ratio shortcut unsafe there",
                    fell_back,
                )
        self.served = 0

    # -- what a request remembers between spans ---------------------------------------------

    def _take_residual(
        self, request_ids, rows: int, like: torch.Tensor
    ) -> torch.Tensor:
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

    def _hold_windows(self, linear_layers, windows) -> None:
        """Keep this call's convolution windows where its layers can find them.

        `windows` is `(rows, len(linear_layers), channels)` -- each layer's convolution
        PARTIAL, the history columns already weighted and summed by the caller, because the
        ring is the caller's. Held on the DEPARTURE's thread-local and overwritten every call:
        a partial describes one token's history, and a stale one finishing the next token's
        convolution is wrong with no shape to say so.
        """
        if windows is None:
            self._local.span_windows = None
            return
        rows = windows.shape[0]
        if windows.dim() != 2 or windows.shape[1] % len(linear_layers):
            raise RuntimeError(
                f"a span call attached partials shaped {tuple(windows.shape)} for "
                f"{len(linear_layers)} linear layers. Matching them up by position would "
                f"convolve some layer against another's history."
            )
        windows = windows.reshape(rows, len(linear_layers), -1)
        self._local.span_windows = {
            int(lid): windows[:, i] for i, lid in enumerate(linear_layers)
        }

    def run(
        self,
        request_ids,
        group: int,
        attn_output: torch.Tensor,
        positions,
        on_query=None,
        windows=None,
    ):
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
        self._hold_windows(rest, windows)
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
        hidden, residual = _add_and_norm(
            layers[head].post_attention_layernorm, attn_out, residual
        )
        # Where a span's non-callback time goes. The header budgets 2079 us for this work,
        # measured per layer locally; the deployment spends 4.6x that. Timed rather than reasoned
        # about, because the first guess -- the blocking `.to("cpu")` on the send path -- came
        # back at 12-19 us and is not it.
        # The span's FIRST linear layer gets its window here, from the head layer's own
        # post-attention residual. The head's feed-forward is a feed-forward like any other and
        # there is no reason for it to be the one with nothing beside it -- with this, every
        # linear layer in the group has a read on the wire while a feed-forward runs.
        #
        # BEFORE the mlp below, which is the whole point: the read goes out, then the pool spends
        # a feed-forward, and the host contracts in that time.
        _t0 = time.perf_counter()
        self._send_early(
            layers[rest[0]],
            layers[rest[0]].linear_attn,
            request_ids,
            rest[0],
            hidden,
            raw_residual=residual,
        )
        hidden = layers[head].mlp(hidden)
        _t1 = time.perf_counter()
        hidden, residual = self._linear_run(
            request_ids,
            rest[:-1],
            hidden,
            residual,
            ahead_of=rest[-1],
            # the head layer's post-attention residual, which is the read point for the span's
            # first linear layer under shift 1 -- the same tensor `_issue_read` was just given
            prev_residual=residual,
        )
        _t2 = time.perf_counter()
        out = self._finish(request_ids, rest[-1], hidden, residual, on_query, positions)
        _span_parts(_t1 - _t0, _t2 - _t1, time.perf_counter() - _t2)
        return out

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look the caller's token ids up in the embedding this pool holds.

        The caller sends ids when it does not hold `embed_tokens` at all -- 2.368 GiB it then does
        not pay for, and 10 KiB a row it does not send either, since an id is 8 bytes.
        """
        flat = token_ids.reshape(-1).to(self.model.model.embed_tokens.weight.device)
        return self.model.model.embed_tokens(flat)

    def run_prologue(
        self,
        request_ids,
        embedded: torch.Tensor,
        positions,
        on_query=None,
        windows=None,
    ):
        """The layers below the first attention, fed by the embedding rather than by a `W_o`.

        On this model that is layers 0, 1 and 2, and it is where a request's residual starts.

        It projects a query like any other span. The query belongs to layer 3, and layer 3 is not
        the shift's exempt layer -- three layers sit beneath it, so under shift 1 it reads `h_2`
        and there is a residual for it to read. Only layer 0 is exempt, and layer 0 is inside this
        span rather than at its end.
        """
        span = self._span_of(-1)[1:]
        self._hold_windows(span, windows)
        hidden, residual = self._linear_run(
            request_ids, span[:-1], embedded, None, ahead_of=span[-1]
        )
        return self._finish(
            request_ids, span[-1], hidden, residual, on_query, positions
        )

    def _finish(
        self,
        request_ids,
        layer_id: int,
        hidden,
        residual,
        on_query,
        positions,
    ):
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
        # the pass's last linear layer. `_linear_run` sent its early projection during the
        # feed-forward before this one, or did not -- a prefill chunk, or a run with no layer to
        # open the window in -- and either way this layer's own call is the same.
        hidden = self._linear_attention(
            last,
            last.linear_attn,
            request_ids,
            layer_id,
            hidden,
            # the residual handed in IS the previous layer's post-attention one -- what
            # `_linear_run` returns -- so this layer's decode rows take the early view whether or
            # not a window was opened for them
            prev_attn_residual=residual,
        )
        hidden, read_point = _add_and_norm(
            last.post_attention_layernorm, hidden, residual
        )

        # the projections live on the DECODER LAYER, not on a submodule: `self_attention` is a
        # method that runs them. Naming a submodule that does not exist fails at the first token,
        # not at install, which is where this cost a deployment round.
        nxt = layers[self.next_attention[layer_id]]
        if self.query_shift == 1:
            # the read point, one feed-forward before the group's output. The send is covered by
            # that feed-forward, which is the window.
            # `hidden` is the same residual under the last layer's norm weights; the ratio
            # applies the head's own weights without a second reduction (pool_cook.norm_ratio,
            # held to the identity by test)
            attn_ratio = self._attn_norm_ratio[int(layer_id)]
            early_in = (
                hidden * attn_ratio
                if attn_ratio is not None
                else nxt.input_layernorm(read_point)
            )
            q, _, _, gate = nxt.forward_prepare_native(positions, early_in)
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
            layers[head].post_attention_layernorm, attn_out, residual
        )
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

    def run_epilogue_with_logits(self, request_ids, group: int, o: torch.Tensor):
        """The epilogue, plus the language-model head for each request's LAST row.

        A chunk's rows are one request's own tokens in order, so the row a sampler
        needs is the last of each id-run -- computed from the ids the frame already
        carries, batched into one [requests, vocab] GEMM. PRE-softmax on purpose:
        softmax and every sampling knob stay on the host with the sampler. What this
        buys the host is its whole 2.37 GiB head: never allocated, its weight read
        never on its critical path, its gibibytes gone to KV.
        """
        from sglang.srt.afd.slots import _runs

        hidden = self.run_epilogue(request_ids, group, o)
        last = [first + n - 1 for _, first, n in _runs(request_ids)]
        root = self.model
        normed = root.model.norm(hidden[last])
        if isinstance(normed, tuple):
            normed = normed[0]
        logits = torch.nn.functional.linear(
            normed.to(root.lm_head.weight.dtype), root.lm_head.weight
        )
        return hidden, logits

    def _send_early(
        self, layer, attn, request_ids, layer_id, prev_attn_normed, raw_residual=None
    ):
        """Cook the NEXT layer's read and send it, so the host contracts while a feed-forward runs.

        Everything preparatory happens here, on this side, synchronously: the projection, the
        convolution against the window the host attached to this span, the normalisation and the
        coefficient. The host receives finished materials and does one contraction. The first
        build sent raw materials and let the host prepare them -- 1.207 ms of preparation against
        the 0.461 ms window a feed-forward opens, so the schedule's max moved to the host and a
        ladder priced it at 10.69 percentage points. `pool_cook` is the half that moved back.

        Synchronous, deliberately, where the raw build had a worker thread and a CUDA event. The
        cook is a few small kernels on the card that is idle 83% of a span, and the worker bought
        their overlap at 0.5 ms of interpreter contention -- the measured pool-side cost nearly
        doubled. The send itself is queued on the outbox (the copy is non-blocking behind an
        event), so this thread issues the feed-forward next, not a stream synchronisation.

        Sends nothing, and the layer reads at its own step, whenever the window cannot be opened:

            shift 0               the arm that is standard AFD. There is no early view to read
                                  from and nothing to overlap, by construction
            rung below 2          the ladder's lower rungs measure the span mechanism alone
            any run > 1 row       a prefill chunk. Its read is an OP_STATE_SCAN carrying k, v and
                                  the gates from THIS layer's own input, which does not exist yet
            no previous residual  the first layer of a pass, or an isolated one
            no window             the host attached none -- a prefill pass, or a host that does
                                  not hold this span's rings
        """
        if self.query_shift == 0:
            return None
        lane = None
        if getattr(self._local, "use_lane", False):
            from sglang.srt.afd.lane import the_lane

            lane = the_lane()
        issue = getattr(self._local, "issue_host", None)
        if issue is None and lane is None:
            # Not a fallback. At shift 1 the early read is the arrangement, so a departure that
            # did not install the split calls is misconfigured, and running on without them is
            # how three separate comparisons came to measure a mechanism that never executed.
            raise RuntimeError(
                f"layer {layer_id} is running at query shift 1 and nothing installed "
                f"`issue_host`. `afd.pool_linear` installs it on every departure; a runner "
                f"without it can only serve the arrangement shift 1 exists to replace, and do "
                f"it silently."
            )
        if any(n != 1 for _, _, n in _runs_of(request_ids)):
            return None
        if prev_attn_normed is None or layer is None:
            return None
        windows = getattr(self._local, "span_windows", None)
        window = None if windows is None else windows.get(int(layer_id))
        if window is None:
            return None
        from sglang.srt.afd_query_shift.cook_kernel import cook_early_fused

        raw = self._early_materials(
            layer, attn, request_ids, layer_id, prev_attn_normed, raw_residual
        )
        if raw is None:
            return None
        early_qk, beta = raw
        key_width = early_qk.shape[-1] // 2
        weight = attn.conv1d.weight.squeeze(1)
        # Fused: one launch where the eager sequence was about twenty. The eager path stays the
        # reference (`pool_cook.cook_early`, held to the fused one by `test_afd_cook_kernel`);
        # the clocks priced the eager cook at 0.573 ms live for 0.148 of arithmetic, and the
        # difference is interpreter time this thread shares with the reader and the sender.
        q_tilde, q = cook_early_fused(
            early_qk,
            beta,
            window[:, : 2 * key_width].to(early_qk.device),
            weight[: 2 * key_width],
            key_heads=attn.num_k_heads // attn.attn_tp_size,
            value_heads=attn.num_v_heads // attn.attn_tp_size,
            head_k_dim=attn.head_k_dim,
        )
        pending = getattr(self._local, "early_handles", None)
        if pending is None:
            pending = self._local.early_handles = {}
        if lane is not None:
            from sglang.srt.afd_query_shift import nccl_lane

            reading_w = (attn.num_v_heads // attn.attn_tp_size) * attn.head_v_dim
            pending[int(layer_id)] = (
                "lane",
                nccl_lane.send_early(lane, q_tilde.reshape(1, -1), reading_w),
            )
        else:
            pending[int(layer_id)] = issue(layer_id, request_ids, q_tilde)
        kept = getattr(self._local, "early_kept", None)
        if kept is None:
            kept = self._local.early_kept = {}
        # the query stays HOME: `s = beta (k . q)` is assembled here now, and the ring's q
        # channels advance with the raw column -- the operator has one query and this is it,
        # so the current projection computes no q at all
        kept[int(layer_id)] = (q, early_qk[:, :key_width])
        cooked = getattr(self._local, "early_cooked", None)
        if cooked is None:
            cooked = self._local.early_cooked = set()
        cooked.add(int(layer_id))

    def _early_materials(
        self, layer, attn, request_ids, layer_id, prev_attn_normed, raw_residual=None
    ):
        """This layer's query and key, projected from the PREVIOUS layer's post-attention residual.

        Returns `(pre-convolution [q | k], beta)` for the far end to finish, or None when there is
        no previous layer -- an isolated layer, or the last of a pass -- which then falls back to
        its own inputs rather than inventing one.

        Unconvolved, deliberately. The convolution against the ring, the normalisation, the query
        coefficient and the contraction all happen on the side that holds the history; this side
        stops at the projection because that is the last step it can take without the ring.
        """
        if prev_attn_normed is None or layer is None:
            return None
        # imported here rather than at module scope for the same reason the normal path does it:
        # `linear_history` reaches back into this package and a top-level import closes the loop.
        # astcheck does not catch a name imported inside one function and used in another, which
        # is how the first version of this reached the deployment as a NameError on the first span.
        from sglang.srt.afd.linear_history import write_strength

        # SLICED. `in_proj_qkvz` is fused and lays out `[q | k | v | z]` at widths
        # `[key_dim, key_dim, value_dim, value_dim]` -- 2048, 2048, 6144, 6144 on this model. The
        # early view needs q and k, which is 25% of that output, and the fused call spends 112 us
        # against 16 us for the slice (measured, this device, one row). Time tracks output width
        # almost exactly here, so the usual "the fused one is faster anyway" does not hold: this
        # shape is bound by writing the output.
        #
        # v and z are not computed at all. v is not wanted -- it feeds `beta (k.q) v`, which is
        # this step's own contribution and is projected AFTER the feed-forward from `x_l`; z is
        # consumed by the norm on the current path. Both used to be produced here and discarded.
        # THIS layer's own reading, at the price of one multiply. The previous layer's
        # post-attention norm produced `prev_attn_normed` from the same residual, so the two
        # norms share the same rms and differ only in the learned scale -- the install-time
        # ratio applies the difference. Reusing the neighbour's reading unscaled was measured
        # at +1.46 points of bits per byte (1.83 against 0.37 for the whole arrangement): the
        # mis-normed raw q column compounds through the convolution history, so the reuse and
        # the early-q ring were poisonous TOGETHER while each alone is nearly free.
        # The reused variants were deleted with their figures banked in the design record.
        ratio = self._norm_ratio[int(layer_id)]
        if ratio is not None:
            normed = prev_attn_normed * ratio
        else:
            if raw_residual is None:
                raise RuntimeError(
                    f"layer {layer_id} needs its own full norm (the ratio shortcut is unsafe "
                    f"there) and the caller did not pass the raw residual it would read."
                )
            normed = layer.input_layernorm(raw_residual)
        rows = normed.shape[0]
        key_width = attn.key_dim // attn.attn_tp_size
        qk = _project_qk(attn, normed, key_width)
        # SLICED for the same reason `in_proj_qkvz` is: `in_proj_ba` lays out `[b | a]` and this
        # path wants b alone, because `a` only feeds the decay and the decay is the far end's,
        # taken from the current projection rather than this one.
        heads = attn.num_v_heads // attn.attn_tp_size
        b = _project_prefix(attn.in_proj_ba, normed, heads, "the decay's input")
        query, key = qk.split([key_width, key_width], dim=-1)
        # The convolution slices too: it runs over `[q | k | v]` and the early pass needs the q
        # and k channels only. Depthwise means each channel is filtered on its own, so taking a
        # prefix of the channels is exactly the same arithmetic on those channels.
        # `write_strength`, not `gates`. The decay is five kernels and this path discards it --
        # the far end takes alpha from the current projection, not this one.
        beta = write_strength(b)
        # The RAW projection, unconvolved. The convolution needs the ring, the ring is the
        # caller's, and 80 KiB of ring a layer a request will not cross this link -- MEASURED at
        # 30 MiB a token at eight callers against a 670 MB/s ceiling. So the materials go instead:
        # 8 KiB a row, amortised over a departure like every other callback.
        return torch.cat([query, key], dim=-1), beta

    def _linear_run(
        self,
        request_ids,
        layer_ids,
        hidden,
        residual,
        ahead_of=None,
        prev_residual=None,
    ):
        """Whole linear-attention layers, back to back, each one's read beside its neighbour's
        feed-forward.

        The order is the schedule, and the schedule is the only thing here worth reading twice:

            layer i    finish layer i against the reading the far end already has, then the
                       attention, and take the post-attention residual
            layer i    ISSUE layer i+1's read from that residual -- before the feed-forward
            layer i    run the feed-forward, while the host contracts layer i+1's state
            layer i+1  collect

        Written the other way round -- feed-forward first, issue after -- it is the arrangement
        this replaced, and it produces exactly the same tokens. Only the pool's `hidden` counter
        can tell them apart, which is why that counter exists.

        `ahead_of` is the layer AFTER this run -- `_finish`'s -- and it is the difference between
        one window a span and two. A span's three linear layers are split two here and one there,
        so without it the last feed-forward in this loop would have nothing beside it and the
        layer that follows would ask and wait as before. Returns the handle for it alongside the
        stream, because whoever runs that layer has to be the one to collect.
        """
        layers = self.model.model.layers
        for position, layer_id in enumerate(layer_ids):
            layer = layers[layer_id]
            hidden, residual = _add_and_norm(layer.input_layernorm, hidden, residual)
            hidden = self._linear_attention(
                layer,
                layer.linear_attn,
                request_ids,
                layer_id,
                hidden,
                prev_attn_residual=prev_residual,
            )
            hidden, residual = _add_and_norm(
                layer.post_attention_layernorm, hidden, residual
            )
            prev_residual = residual
            # the window. It has to open HERE -- after this layer's residual exists and before
            # this layer's feed-forward -- because those two events are what it spans.
            nxt = layer_ids[position + 1] if position + 1 < len(layer_ids) else ahead_of
            if nxt is not None:
                self._send_early(
                    layers[nxt],
                    layers[nxt].linear_attn,
                    request_ids,
                    nxt,
                    hidden,
                    raw_residual=residual,
                )
            # the feed-forward's INPUT, the one step between layer 0 (exact) and layer 1 (6.3%
            # high) that has never been compared. Its output has: 0.6% at layer 0.
            hidden = layer.mlp(hidden)
            # the feed-forward's own output, never compared until now. Layer 0's x + attn is exact
            # and layer 1's is 6.3% high, and layer 1's INPUT is layer 0's residual plus THIS. An
            # attention that is perfectly right computes a wrong answer from a wrong input, so the
            # feed-forward between them has to be ruled in or out before the attention is blamed.
        return hidden, residual

    def _finish_linear(self, attn, core, z, rows: int, hidden):
        """The z-gated norm and the output projection, which stay on the pool under both cuts.

        Shared by the READ path and the MIX path deliberately: they differ in WHERE `core` was
        computed and in nothing after it, and a second copy of these three lines is where the two
        arrangements would quietly drift apart.

        Returns `gated_core` and the cast `core` as well as the output, because the READ path
        traces them and the trace reads locals this function does not have. An earlier version of
        this split moved the trace in here with the arithmetic and left it referring to
        `request_ids`, `alpha`, `q_tilde` and `reading` -- four unbound names that only raise once
        a first prefill with more than one row reaches the branch.
        """
        core = core.reshape(rows, -1).to(hidden.dtype)
        gated_core = attn.norm(
            core.reshape(-1, attn.head_v_dim), z.reshape(-1, attn.head_v_dim)
        )
        out, _ = attn.out_proj(gated_core.reshape(rows, -1))
        return out, gated_core, core

    def _linear_attention(
        self,
        layer,
        attn,
        request_ids,
        layer_id: int,
        hidden: torch.Tensor,
        prev_attn_residual: torch.Tensor | None = None,
    ):
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

        `_send_early` may have sent this layer's shifted projection during the PREVIOUS layer's
        feed-forward, in which case the far end contracted the state while that ran and this call
        collects the result inside its own reply. When it did not -- a prefill chunk, an isolated
        layer, the first layer of a pass -- the far end contracts at this step instead, and the
        layer is exactly what it was before the window existed. Nothing here distinguishes them.
        """
        from sglang.srt.afd.linear_history import (
            gates,
        )

        ask = getattr(self._local, "ask_host", None)
        # The stateless arrangement (#75). With the ring on the CALLER, this side holds nothing per
        # request: the pre-convolution projection goes down with the gates and `core` comes back,
        # and the convolution, the state read and the state advance all happen over there. It does
        # not save a crossing -- the callback already happened, this changes what rides in it --
        # and it is not meant to: what it buys is that a request stops being sticky to the pool
        # that served its last call, which is what multi-pool routing and mid-request pool
        # replacement rest on.
        mix = getattr(self._local, "mix_host", None)
        if mix is None and ask is None:
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
        _p0 = time.perf_counter()
        rows = hidden.shape[0]
        cooked = getattr(self._local, "early_cooked", None)
        cooked_here = cooked is not None and int(layer_id) in cooked
        key_width = attn.key_dim // attn.attn_tp_size
        value_width = attn.value_dim // attn.attn_tp_size
        if cooked_here:
            # The operator has ONE query and it went early, so the current projection computes
            # no q at all: the fused weight's q rows are sliced away, an eighth of a weight
            # read the serial arrangement pays and this one does not. The same slice-not-copy
            # argument as `_project_prefix`, from the other end of the rows.
            kvz = _project_suffix(
                attn.in_proj_qkvz, hidden, key_width, "the early query's rows"
            )
            key, value, z = kvz.split([key_width, value_width, value_width], dim=-1)
            value = value.reshape(rows, -1, attn.head_v_dim)
            z = z.reshape(rows, -1, attn.head_v_dim)
            ba, _ = attn.in_proj_ba(hidden)
            nv = attn.num_v_heads // attn.attn_tp_size
            b, a = ba.split([nv, nv], dim=-1)
            packed = torch.cat([key, value.reshape(rows, -1)], dim=-1)
        else:
            qkvz, _ = attn.in_proj_qkvz(hidden)
            ba, _ = attn.in_proj_ba(hidden)
            query, key, value, z, b, a = attn.fix_query_key_value_ordering(qkvz, ba)
            flat = [t.reshape(rows, -1) for t in (query, key, value)]
            packed = torch.cat(flat, dim=-1)
        # BEFORE the convolution. The model's `mixed_qkv`, which is what reaches
        # `self.attn(forward_batch, mixed_qkv=...)`, is pre-convolution too -- the convolution runs
        # inside the backend. Publishing the post-convolution tensor here compared one side's
        # filtered channels against the other's unfiltered ones and read 100% apart on every key
        # head while the layer's output agreed to 2%, which is impossible and is how it was caught.
        self._local.last_packed = packed.detach()
        # Both shifts take this path, and it is the only one. What shift 1 adds happened before
        # this line: `_issue_read` sent the shifted projection to the far end, which contracted
        # the state with it while the feed-forward above ran and keeps the result for this call.
        # Nothing here has to know that; the reply is the same shape either way.
        _p1 = time.perf_counter()
        windows = getattr(self._local, "span_windows", None)
        if cooked_here:
            cooked.discard(int(layer_id))
            # The cooked path. The early frame for this layer went out a feed-forward ago with
            # the coefficient already formed; what remains to cross is this step's own half,
            # finished here for the same reason: the far end contracts and assembles, and any
            # preparation over there is the schedule's max moving back to the host.
            from sglang.srt.afd_query_shift.cook_kernel import assemble_and_pack
            from sglang.srt.afd_query_shift.pool_cook import cook_mix

            lane = None
            if getattr(self._local, "use_lane", False):
                from sglang.srt.afd.lane import the_lane

                lane = the_lane()
            collect = getattr(self._local, "collect_early", None)
            apply_step = getattr(self._local, "apply_host", None)
            window = None if windows is None else windows.get(int(layer_id))
            if (
                lane is None and (collect is None or apply_step is None)
            ) or window is None:
                raise RuntimeError(
                    f"layer {layer_id}'s early frame was cooked and sent, but the assembly "
                    f"is missing "
                    f"{'its collect/apply calls' if window is not None else 'its window'}. "
                    f"The far end has answered a contraction nothing will claim, and serving "
                    f"the raw-materials op instead would read the state twice for one token."
                )
            kept = getattr(self._local, "early_kept", {}).pop(int(layer_id), None)
            handles = getattr(self._local, "early_handles", {}).pop(int(layer_id), None)
            if kept is None or handles is None:
                raise RuntimeError(
                    f"layer {layer_id} was cooked but its query or its reading's handles were "
                    f"not kept. Assembling from the current projection instead would compute a "
                    f"different model with nothing raising."
                )
            q_kept, raw_q = kept
            k, v = cook_mix(
                packed,
                window[:, key_width:].to(packed.device),
                attn.conv1d.weight.squeeze(1)[key_width:],
                key_heads=attn.num_k_heads // attn.attn_tp_size,
                value_heads=attn.num_v_heads // attn.attn_tp_size,
                head_k_dim=attn.head_k_dim,
                head_v_dim=attn.head_v_dim,
            )
            # The reading was pushed back while the feed-forward above ran; this wait is a
            # table lookup when the schedule holds, and the tripwire counts when it does not.
            if isinstance(handles, tuple) and handles and handles[0] == "lane":
                from sglang.srt.afd_query_shift import nccl_lane

                reading = (
                    nccl_lane.collect_reading(handles[1])
                    .float()
                    .reshape(
                        rows, attn.num_v_heads // attn.attn_tp_size, attn.head_v_dim
                    )
                )
            else:
                reading = collect(handles, rows).reshape(
                    rows, attn.num_v_heads // attn.attn_tp_size, attn.head_v_dim
                )
            # gates + s + core in ONE launch: a dozen eager kernels' interpreter time was the
            # cost, not their arithmetic (test_afd_assemble_kernel holds the agreement)
            # the APPLY frame is packed INSIDE the assembly launch: the naive packing --
            # five casts and a cat on this serial path -- gave back on the pool what it
            # saved on the host (flown, mixed-to-negative). The kernel already holds k and
            # v for the core, so the slab costs it stores, not launches, and the old
            # column cat disappears into the slab's last two segments.
            core, slab = assemble_and_pack(
                a, b, attn.A_log, attn.dt_bias, k, q_kept, reading, v, raw_q, packed
            )
            if lane is not None:
                nccl_lane.send_apply(lane, slab)
            else:
                apply_step(layer_id, request_ids, slab)
            _p2 = time.perf_counter()
            out = self._finish_linear(attn, core, z, rows, hidden)[0]
            _linear_parts(_p1 - _p0, _p2 - _p1, time.perf_counter() - _p2)
            return out
        if mix is not None:
            # Everything from the convolution to `core` happens on the far side. `z` stays here
            # and is consumed by the norm below, so it never crosses -- the traffic table in
            # AFD_STATELESS_POOL.md was pessimistic by that much.
            alpha, beta = gates(a, b, attn.A_log, attn.dt_bias)
            core = mix(layer_id, request_ids, packed, alpha, beta)
            _p2 = time.perf_counter()
            out = self._finish_linear(attn, core, z, rows, hidden)[0]
            _linear_parts(_p1 - _p0, _p2 - _p1, time.perf_counter() - _p2)
            return out

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

    def forget_namespace(self, namespace: int) -> int:
        """Drop a departed host's residuals and gates, so nothing of it lingers here.

        The shared half settles a departed host once, on the connection's way out
        (`Departure.host_departed`); this is the span runner's share of that settling,
        reached through the same duck-typed hook the recurrent slots use.
        """
        mask = 0xFFFF << 40
        with self._lock:
            mine = [r for r in list(self._residual) if (r & mask) == namespace]
            for r in mine:
                self._residual.pop(r, None)
            gates_mine = [r for r in list(self._gate) if (r & mask) == namespace]
            for r in gates_mine:
                self._gate.pop(r, None)
        return len(mine) + len(gates_mine)

    def report(self) -> dict:
        with self._lock:
            held = len(self._residual)
        return {
            "spans_served": self.served,
            "residuals_held": held,
            "groups": sorted(self.spans),
            **self.states.report(),
        }
