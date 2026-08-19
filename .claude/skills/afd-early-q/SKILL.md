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

## What may leave the host, and what may not

The split is by STATE, not by cost. A pool is worth having because it is stateless: a caller that
stalls blocks nobody, and the pool need not be reserved for a request between that request's own
calls. Anything holding per-request state destroys that property the moment it moves.

    host    every attention. A softmax layer holds a KV cache; a linear-attention layer holds a
            recurrent state. Both belong to the request, and a pool holding either could no
            longer be released and retaken between one request's calls.
    pool    the feed-forward, and only it. It reads the same weights whatever the caller's
            history, so a request sweeping a million positions and one sweeping a thousand cost
            it exactly the same. That is why they can share it.

A hybrid stack makes this concrete rather than abstract: Qwen3.8-27B is 48 linear-attention layers
and 16 softmax ones, and all 64 keep their token mixer on the host. Only the 64 feed-forwards move.

## Coverage is not the same question as overlap

Three questions look alike and have different answers. Answering them with one function silently
capped a run's coverage at a quarter of the stack:

    which layers sweep a cache        the softmax ones. This is the DEPLOYMENT question: only a
                                      sweep over cached keys can start before the feed-forward
                                      preceding it, because only it needs no more than the query.
    which layers' queries move        every layer that has a query, including linear attention.
                                      This is the QUALITY question, and the rewiring is the same
                                      rewiring on both kinds.
    which layers hold state           all of them, for the reason above.

On this model the two coverages are distinguishable and were measured: 16 of 64 layers costs
+0.0181 bits per byte, 63 of 64 costs +0.0211 -- four times the coverage for 1.17 times the cost.
Reporting the cheaper coverage's number under the fuller coverage's name is a real error, not a
rounding, and `--afd-coverage {all,softmax}` exists so the choice is written down in the run rather
than implied by which function someone reached for.

A linear-attention layer's query is the first slice of a fused `in_proj_qkvz` that splits
`[key, key, value, value]`. Splicing at the projection's output is safe because the conv1d that
follows is depthwise: each channel is filtered on its own, so replacing a contiguous channel range
does not mix it with its neighbours. Check that before porting to a family whose convolution is
not grouped.

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

### Step 2 is a partition of attention, not a different kernel

`self.attn(q, k, v, forward_batch)` is one call: the sweep over the cache and the fold-in of this
step's token happen inside one kernel, and there is nothing to put between the issue and the
collect until that call is split. Softmax attention is a mergeable aggregate, so the split is
exact:

    sweep   the positions already in the cache. Needs q and the cache. Runs in the window
    join    this step's token. Needs k and v, which need x_l, which comes back from the pool
    merge   `merge_state(o_sweep, lse_sweep, o_join, lse_join)`

Do both halves through the SAME `forward_decode`, with `kv_indptr`/`kv_indices` swapped for the
partition's own, rather than writing a sweep kernel. That function is a hundred lines of branching
-- kv scales, logit capping, sinks, sliding windows, MLA, the unified pool's loc translation -- and
a reimplementation gets some subset right and drifts from the rest. The failure is not a crash; it
is attention that is slightly wrong on the configurations the reimplementation forgot.

### The order inside the window is the whole schedule

    issue, sweep, collect     the window
    sweep, issue, collect     nothing hidden: `issue` copies the hidden states to the host and so
                              synchronises the stream, and the send waits for the sweep
    issue, collect, sweep     nothing hidden: the synchronous port

All three give the same tokens. Assert the order directly; a benchmark can only tell them apart by
a smaller number, and a smaller number has many other explanations.

### Which layers get a window

A layer j opens one only when layer j+N sweeps a cache. On a hybrid stack that is the softmax
layers alone -- 16 of 64 on Qwen3.8-27B -- because a linear-attention layer's query multiplies a
recurrent state, and partitioning THAT is a separate piece of work. The other 48 feed-forwards are
still issued and waited for. Report the window count beside any speedup, or the speedup is read
against 64 layers of hiding that never existed.

## Testing, in the order the failures actually appear

1. **The read point is the tensor you think it is.** Capture `h_l` on a converted layer and the
   same layer's input `x_l`; assert they differ, and that `h_l` equals the layer's output minus its
   feed-forward contribution. Zero convertible layers is the failure this catches -- a conversion
   that hooks nothing costs nothing, and a cost of zero reads as tolerance.
2. **Coverage.** Assert the number of converted layers equals what the shift asks for, and that
   every clamped layer is in the record. On a hybrid stack, assert that the linear-attention layers
   were handled as the arm intends rather than skipped by accident.
3. **The split is exact**, and the number that says so has a control. Sweep plus join against one
   fused attention call, on real weights, to within the dtype's rounding -- before any performance
   claim, because a protocol that is 1e-2 relative is not the same model. A small gap only means
   something against what a WRONG partition would have given: measure the same comparison with the
   boundary moved by one position in each direction. In float64 at this model's shapes the honest
   partition sits at 6e-16 and either boundary error at 2e-1, so the two cannot be confused.
   Check the partition in index space too -- complete and disjoint -- because the merge is exact
   for any partition, including one that drops a position, and the output stays plausible.
   Greedy-decode agreement is NOT this test: batch composition alone parts the same model's greedy
   output on 3 of 4 prompts, so an arm that differs on 2 of 4 has said nothing yet.
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
