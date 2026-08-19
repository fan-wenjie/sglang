---
name: afd-early-q
description: Add attention/feed-forward disaggregation with an Early-Q read point to SGLang, running the two sides asynchronously. Use when wiring a host that owns the KV cache to a stateless feed-forward pool, when adding the q-shift-layers knob, or when a schedule that should overlap does not.
---

# AFD with an Early-Q read point

Two sides. The **host** owns the KV cache and runs attention. The **pool** owns the static weights
and runs the feed-forward, statelessly, for whoever calls. The point of the exercise is that under
one wiring the host can call the pool and keep working, and under the other it cannot.

This skill is the procedure. It is not a description of a finished thing: at the time of writing
upstream SGLang has no AFD of its own -- 1374 branches and the remote were searched, and the `afd`
matches are hex substrings in commit SHAs -- so the argument names below follow SGLang's **P/D**
disaggregation conventions rather than an existing AFD, and are chosen so that an upstream AFD
could adopt them without a rename.

## The one edge that moves, in SGLang's own terms

A `Qwen3_5AttentionDecoderLayer.forward` walks the residual stream through three named stages:

```
hidden, residual = layer_communicator.prepare_attn(...)    # hidden = LN1(x),  residual = x
hidden           = self.self_attention(...)                # the attention output
hidden, residual = layer_communicator.prepare_mlp(...)     # hidden = LN2(h),  residual = h  <=== h
hidden           = self.mlp(hidden)                        # the feed-forward
hidden, residual = layer_communicator.postprocess_layer(...)
```

`residual` after `prepare_mlp` **is** `h_l`: the stream after this layer's attention and before its
feed-forward. That single tensor is the whole intervention. Early-Q projects layer `l+1`'s query
from it; the key and value still come from `x_{l+1}`, which does not exist until `self.mlp` has
run.

So at the moment `prepare_mlp` returns, two things are issuable at once:

- the **feed-forward of layer `l`**, which needs only `LN2(h_l)`, and
- the **sweep of layer `l+1`**, which needs only `q_{l+1} = W_q LN(h_l)` and the cached keys and
  values.

Layer `l+1` is ready when both return. That is the overlap, and it is one feed-forward wide -- no
more, because `k_{l+1}` and `v_{l+1}` are projected from `x_{l+1} = h_l + MLP(LN2(h_l))` and the
current position still has to be folded into the swept state.

**If you take one thing from this file:** the hook point is the `residual` returned by
`prepare_mlp`, not the layer's input and not its output. Reading the layer output gives `x_{l+1}`,
which is the standard wiring with extra steps; a port that quietly does that produces a correct
model with no overlap and no error message.

## What `--afd-q-shift-layers` means

The knob is a layer count `N`. Three things are the same number, which is why it is one knob:

```
offset       = N - 0.5 layers      how far back the query is read
source       = h_{l-N}             the residual it is read from
group size   = N layers            what the shift spans
half-layers  = 2N - 1              the study's own unit for the same depth
```

`N = 0` is the standard wiring and the only value that means "off". `N = 1` reads `h_{l-1}`, half a
layer back -- the operating point. There is no fractional setting: a query read from between the
two sub-layers of a block would be read from a point where the residual stream has no value, so the
knob is an integer and a non-integer is an error rather than a rounding.

The group reading is what makes `N` natural on a hybrid stack. Qwen3.8-27B has
`full_attention_interval = 4` and 64 layers, so its `layer_types` run
`[linear, linear, linear, full] x 16` and the full-attention layers are 3, 7, 11, ... -- the LAST
of each group of four. At `N = 4` each of them reads `h_{l-4}`, which is the previous
full-attention layer's own post-attention residual: layer 7 reads `h_3`, layer 11 reads `h_7`. The
sweep of one softmax layer then overlaps the entire group between it and the last one -- three
linear-attention layers and four feed-forwards -- which is the long shadow a hybrid stack offers
and a dense one does not.

Layer 0 is exempt at any `N`: its query reads the embedding under either wiring. A layer with fewer
than `N` layers beneath it clamps to the bottom of the stack -- at `N = 4` that is layer 3, the
first full-attention layer, which has no `h_{-1}` to read. **Record every clamp.** A run that
silently converts fewer layers than asked reports the cost of a shallower shift under a deeper
shift's name, and the number looks better for it.

The study measured this depth axis directly and the residual after repair rises with it -- roughly
four times larger at `N = 4` than at `N = 1` on a dense stack. `N = 1` is the conservative setting
and the one to deploy; the larger values exist because the question "how far back may it go" has to
be answerable, not because a server should use them.

## Argument naming

Follow `disaggregation_*`. Concretely:

```
--afd-mode {null,host,pool}          mirrors --disaggregation-mode {null,prefill,decode}
--afd-pool-addr HOST:PORT            where the host reaches the pool
--afd-bootstrap-port PORT            mirrors --disaggregation-bootstrap-port
--afd-q-shift-layers N               the read point: N-0.5 layers back; 0 is standard
--afd-transfer-backend {tcp,...}     mirrors --disaggregation-transfer-backend
```

Validate in a hook under `srt/arg_groups/`, the way `pd_disaggregation_hook.py` does, not inline in
`ServerArgs.__post_init__`. Two checks that must be there: a negative or non-integer `q_shift_layers` is an error,
and `afd_mode=host` without a reachable `afd_pool_addr` is an error at startup rather than at the
first token.

## The asynchronous part, which is the part that is easy to get wrong

The naive port issues the feed-forward and waits for it. That reproduces the synchronous
arrangement exactly and every measurement then shows no benefit, correctly, because there is none.

The host must:

1. **issue** the feed-forward for layer `l` and get a handle back, not a tensor;
2. **compute** `q_{l+1}` from the same `h_l` and run the sweep;
3. **wait** on the handle only when it needs `x_{l+1}`, which is when it projects `k_{l+1}` and
   `v_{l+1}`.

Between 1 and 3 the pool is not reserved for this request -- it is stateless, so another request's
call may be served in between, and that is the point of pooling it at all.

A receiver thread that fills a per-layer slot is enough; the slot is keyed by (request, layer) and
a wait is a condition variable on that slot. Do not key by layer alone: two requests in flight at
the same layer will overwrite each other, and the symptom is a model that produces fluent text with
a slightly wrong distribution, which no assertion catches.

## Testing, in the order the failures actually appear

1. **The read point is the tensor you think it is.** Capture `h_l` on a converted layer and the
   same layer's input `x_l`; assert they differ, and that `h_l` equals the layer's output minus its
   feed-forward contribution. Zero convertible layers is the failure this catches -- a conversion
   that hooks nothing costs nothing, and a cost of zero reads as tolerance.
2. **Coverage.** Assert the number of converted layers equals what the shift asks for, and that
   every clamped layer is in the record. On a hybrid stack, assert that the linear-attention layers
   were handled as the arm intends rather than skipped by accident.
3. **The split is exact.** Sweep plus join against one fused attention call, on real weights, to
   within the dtype's rounding. Do this before any performance claim; a protocol that is 1e-2
   relative is not the same model.
4. **Asynchrony actually happened.** Record the issue time and the wait time of each pool call. If
   `wait - issue` is the pool's service time for every call, nothing overlapped and the async path
   is a synchronous path with extra machinery.
5. **Two requests, one pool.** The slot-keying bug above only appears with concurrency. One request
   proves nothing about it.

## What this arrangement does not buy

The overlap is one feed-forward wide, so a sweep longer than a feed-forward is only partly hidden,
and a sweep much shorter than one hides nothing that mattered. Whether it pays is a ratio between
an interconnect's latency and a feed-forward's duration; a single card cannot answer it, and a
benchmark on one card that reports a speedup is measuring its own memory bus.
