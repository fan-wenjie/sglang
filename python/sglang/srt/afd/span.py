"""One group of four layers as a single call: the pool's whole job between two attentions.

The per-layer arrangement sends a frame for every feed-forward. On this model that is 63 round
trips a decode step, and the measurements say the fixed cost of a round trip -- 130 us of latency
plus about 148 us of protocol -- is most of what a feed-forward call costs at batch 4. The cut here
is chosen to make that number small by making the unit large.

## Where the cut is, and why there

Qwen3.8-27B's layers run `[linear, linear, linear, full] x 16`. Read from one full attention to the
next, a group is:

    W_o | FFN | linear attn | FFN | linear attn | FFN | linear attn | FFN | -> x for the next attn
    ^                                                                     ^
    the host's attention output                        the next attention's hidden input

Everything between those two arrows is weights and per-request recurrent state, and none of it is
the KV cache. So the whole of it goes to the pool as ONE call, and the host keeps the query and
key/value projections, the softmax attention, and the KV cache.

    round trips a step     16   (against 63 for the per-layer cut)
    span, measured       2046 us at batch 4 (benchmark/afd/span_cost.py)

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
feed-forward before the span's output does. So the pool sends it as soon as it has it, the host
starts its query projection and its cache sweep against it, and the pool runs the last feed-forward
while that happens. The span's own tail pays for the host's head start.

What that tail is worth, measured rather than assumed:

    4 x feed-forward     2040 MiB   74% of the span   298 us each
    3 x linear attention  660 MiB   24%               129 us each
    W_o                    60 MiB    2%                35 us
    one round trip                                    628 us

So the early message is covered by ONE feed-forward, 298 us of the 628 -- not by the span. The
whole span covers the round trip 3.3 times over, but the early half is sent near the end and has
only what follows it to hide behind.

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


class SpanRunner:
    """Runs one group's span on the pool's own weights.

    Stateless per call except for what a request remembers: the residual entering the group, and
    the two recurrent states of each linear layer. Those are keyed by request and live here for
    the length of a generation.
    """

    def __init__(self, model, states, *, layer_types: list[str]) -> None:
        self.model = model
        self.states = states
        self.spans = {s[0]: s for s in group_layers(layer_types)}
        self._residual: dict[int, torch.Tensor] = {}
        self._lock = threading.Lock()
        self.served = 0

    # -- what a request remembers between spans ---------------------------------------------

    def _take_residual(self, request_ids, rows: int, like: torch.Tensor) -> torch.Tensor:
        """The residual entering this group, one row a rider, in the rider order.

        A request making its first call has none, which is the headless span at the bottom of the
        stack; anything else means a group ran out of order and the answer would be a correct
        forward pass over the wrong history.
        """
        with self._lock:
            rows_out = []
            for r in request_ids:
                held = self._residual.get(int(r))
                if held is None:
                    raise RuntimeError(
                        f"request {r} has no residual on the pool, so this is not its first span "
                        f"and the group before it never ran here. The forward would be correct "
                        f"arithmetic over a history this request does not have."
                    )
                rows_out.append(held)
        return torch.stack(rows_out, dim=0).to(like.dtype)

    def _keep_residual(self, request_ids, residual: torch.Tensor) -> None:
        with self._lock:
            for i, r in enumerate(request_ids):
                self._residual[int(r)] = residual[i].clone()

    def release(self, request_id: int) -> int:
        """Forget one request. Returns how many recurrent slots were cleared."""
        with self._lock:
            self._residual.pop(int(request_id), None)
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

    def run(self, request_ids, group: int, o: torch.Tensor, on_read_point=None):
        """From the group's `W_o` input to the next attention's hidden input.

        `on_read_point` is called with `h_{l+3}` the moment it exists, which is one feed-forward
        before the return value does. It is how the host's query projection and cache sweep get
        their head start; a caller that passes nothing simply gets both tensors at the end.

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
        residual = self._take_residual(request_ids, o.shape[0], o)

        # the head layer's tail: its output projection and its feed-forward. The attention itself
        # ran on the host, which is the whole point of the cut
        attn_out, _ = layers[head].self_attn.o_proj(o)
        hidden, residual = _add_and_norm(
            layers[head].post_attention_layernorm, attn_out, residual)
        hidden = layers[head].mlp(hidden)

        hidden, residual = self._linear_run(request_ids, rest[:-1], hidden, residual)
        return self._finish(request_ids, rest[-1], hidden, residual, on_read_point)

    def run_prologue(self, request_ids, embedded: torch.Tensor, on_read_point=None):
        """The layers below the first attention, fed by the embedding rather than by a `W_o`.

        On this model that is layers 0, 1 and 2, and it is where a request's residual starts.

        It hands over a read point like any other span. The query it feeds belongs to layer 3, and
        layer 3 is not the shift's exempt layer -- three layers sit beneath it, so under shift 1 it
        reads `h_2` and there is a residual for it to read. Only layer 0 is exempt, and layer 0 is
        inside this span rather than at its end.
        """
        span = self._span_of(-1)[1:]
        hidden, residual = self._linear_run(request_ids, span[:-1], embedded, None)
        return self._finish(request_ids, span[-1], hidden, residual, on_read_point)

    def _finish(self, request_ids, layer_id: int, hidden, residual, on_read_point):
        """The span's last linear layer, whose residual is the next attention's query source.

        The read point exists BEFORE this layer's feed-forward, and handing it over there rather
        than at the end is the whole of the overlap: the host projects its query and sweeps its
        cache while the pool spends the last feed-forward of the span.
        """
        last = self.model.model.layers[layer_id]
        hidden, residual = _add_and_norm(last.input_layernorm, hidden, residual)
        hidden = self._linear_attention(last.linear_attn, request_ids, layer_id, hidden)
        hidden, read_point = _add_and_norm(last.post_attention_layernorm, hidden, residual)
        if on_read_point is not None:
            on_read_point(read_point)
        hidden = last.mlp(hidden)
        # the next group's input residual is this one's output, and the pool is the one that has
        # it. It stays here rather than travelling both ways for a value neither end changed.
        self._keep_residual(request_ids, read_point + hidden)
        self.served += 1
        return read_point, hidden

    def run_epilogue(self, request_ids, group: int, o: torch.Tensor):
        """The last attention's tail: its output projection, its feed-forward, and the final norm.

        Returns the normalised hidden state rather than logits, so that whether the language-model
        head runs here or on the host stays a separate decision -- it is a 5120 x vocab read, which
        is the largest single weight in the model and the one most worth measuring on its own.
        """
        layers = self.model.model.layers
        head = self._span_of(group)[0]
        residual = self._take_residual(request_ids, o.shape[0], o)
        attn_out, _ = layers[head].self_attn.o_proj(o)
        hidden, residual = _add_and_norm(
            layers[head].post_attention_layernorm, attn_out, residual)
        hidden = layers[head].mlp(hidden)
        hidden, _ = _add_and_norm(self.model.model.norm, hidden, residual)
        self.served += 1
        return hidden

    def _linear_run(self, request_ids, layer_ids, hidden, residual):
        """Whole linear-attention layers, back to back, with nothing between them to interrupt."""
        layers = self.model.model.layers
        for layer_id in layer_ids:
            layer = layers[layer_id]
            hidden, residual = _add_and_norm(layer.input_layernorm, hidden, residual)
            hidden = self._linear_attention(layer.linear_attn, request_ids, layer_id, hidden)
            hidden, residual = _add_and_norm(layer.post_attention_layernorm, hidden, residual)
            hidden = layer.mlp(hidden)
        return hidden, residual

    def _linear_attention(self, attn, request_ids, layer_id: int, hidden: torch.Tensor):
        """One linear-attention layer, whole, against state this pool holds.

        sglang's own path for this reads its state out of a `ForwardBatch`'s cache. There is no
        forward batch here -- the pool serves frames, not requests -- so the projection and the
        gating are the model's own modules and the two stateful steps are called against slot
        buffers indexed the same way sglang indexes its own.
        """
        from sglang.srt.layers.attention.linear.gdn_backend import causal_conv1d_update

        from sglang.srt.afd.linear_state import LinearStateService

        qkvz, _ = attn.in_proj_qkvz(hidden)
        ba, _ = attn.in_proj_ba(hidden)
        query, key, value, z, b, a = attn.fix_query_key_value_ordering(qkvz, ba)
        query, key, value = (t.reshape(t.shape[0], -1) for t in (query, key, value))
        mixed_qkv = torch.cat((query, key, value), dim=-1)

        slots = [self.states.slot_of(int(r)) for r in request_ids]
        indices = torch.tensor(slots, device=hidden.device, dtype=torch.int32)
        conv = self.states.conv_buffer(
            layer_id, width=mixed_qkv.shape[-1],
            taps=attn.conv_weights.shape[-1] - 1, dtype=mixed_qkv.dtype)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv, conv, attn.conv_weights, attn.bias, attn.activation,
            conv_state_indices=indices,
        )
        self.states.note_touched(slots, ("conv", layer_id))

        service = LinearStateService(self.states, scale=attn.head_k_dim ** -0.5)
        core = service.step(request_ids, layer_id, mixed_qkv, a, b, attn.A_log, attn.dt_bias)

        core = attn.norm(core.reshape(-1, attn.head_v_dim), z.reshape(-1, z.shape[-1]))
        core = core.reshape(hidden.shape[0], -1)
        out, _ = attn.out_proj(core)
        return out

    def seed(self, request_id: int, residual: torch.Tensor) -> None:
        """Give a request its first residual: the embedding, for the headless span."""
        with self._lock:
            self._residual[int(request_id)] = residual.clone()

    def report(self) -> dict:
        with self._lock:
            held = len(self._residual)
        return {"spans_served": self.served, "residuals_held": held,
                "groups": sorted(self.spans), **self.states.report()}
