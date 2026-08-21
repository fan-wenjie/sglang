# AFD with an Early-Q read point: what was measured

Everything here was run on Qwen3.8-27B-FP8. Host is an RTX PRO 6000 Blackwell (97 GB); the pool,
where a second machine is involved, is an RTX 5090 (32 GB) across a 10 GbE link with a 0.3 ms TCP
round trip and 4.4 Gbit/s of bulk bandwidth. Numbers with no arrangement named are colocated.

Findings are grouped by what they decide. Several of them refuted the hypothesis that motivated
them, and those are marked, because a refuted hypothesis that stays in the record is the only
protection against re-adopting it.

## 2026-08-21, the residual against its own reference: the collapse is at the first middle span

The pool holds the whole model and also runs the spans, so both sequences come from one process,
one set of weights and one prompt. The model's own per-layer residual, row 0 of a 122-token
prefill, against the residual the span keeps at each boundary:

    the model, per layer          the span, per boundary
    layer 0    28.8
    layer 1    51.5
    layer 2    68.9              span 0 (after layer 2)      76.8      +11%
    layer 3    83.2
    layer 4    86.0
    layer 5    90.5
    layer 6   112.0              span 1 (after layer 6)      34.7      -69%
    layer 7   110.9              span 2                      40.4
    ...                          ... rising slowly to 242.7 at span 15

The model's residual grows monotonically, 28.8 to 237 over twenty layers, which is what a residual
stream does. The arrangement's FALLS by two thirds across its first middle span and then climbs
back from a floor it should never have reached.

Adding vectors cannot halve a norm unless what is added points against what it is added to. So the
first middle span contributes something strongly anti-correlated with the stream it joins -- and
the first middle span is the first one whose attention output came from the HOST. The prologue,
which has no host attention in it, is 11% off rather than 69%.

That is the tightest localisation this search has had. What it does NOT yet say is which of the
three things only a middle span does is responsible: the output gate applied to an attention
output computed elsewhere, `W_o` on that output, or the residual taken from the pool's own table
rather than carried in the call.

Two instrument errors were made getting here and both are worth more than the reading:

    the trace walked ROWS     `_RESIDUAL_SEEN` incremented once a row, so a 122-token prefill
                             logged 122 lines from ONE span. The sequence was read as span
                             boundaries and compared against a per-layer reference: 76.8, 44.2,
                             30.9 falling against 28.8, 51.5, 68.9 rising. Two correct
                             measurements of different quantities, and a comparison of neither
    the norm is gemma-style  it scales by (1 + w), not w. The final hidden's RMS is 1.99 and
                             rms(1 + w) is 1.95. The double-norm fix is unaffected, but what it
                             was said to cost -- a sign flip on negative channels -- was wrong


## 2026-08-21, second run: the arrangement copies its input, and it is not the state

The first run served 24 tokens and repeated the last prompt token. Two real faults were found and
fixed on the way to the second run, and NEITHER of them was the cause.

    OP_STATE_SCAN        a prefill chunk is one request's tokens and each reads what its
                         predecessor wrote. Batched through OP_STATE_READ the state never
                         advanced. Fixed, with a case against N separate read/update pairs.
    route_frame          the pool recognised replies by the literal `== OP_STATE_READ`, so the
                         scan's reply BOARDED as a feed-forward request: 122x6144 readings into a
                         layer expecting 5120. Fixed by routing on INBOUND_OPS, with the test
                         harness now calling the production router instead of a copy of it.

What the second run measures, with the pool serving the same prompt colocated as the control:

    prompt                  colocated                 the arrangement
    "The capital of         " Paris.\nThe capital     " is is is is is is ..."
     France is"              of Germany is Berlin."
    input logprobs          -8.86 -0.464 -3.738       -30.465 -19.063 -30.939
     (positions 1-3)                                   -0.590 -> -0.000 at position 4
    "The" (one token)       " following"              "The"

The single-token prompt is the one that decides it. With ZERO history the arrangement returns its
own input token at -0.206. A state that never advances cannot produce that -- there is no state to
advance yet. So the scan was a real bug and a different one, and the fault is in the per-token
forward of the span itself.

A third real fault was found and fixed after this and is also NOT the cause. The pool's
`run_epilogue` applied `model.norm`, and sglang's own forward applies it again after the layer
loop because the closing head hands back `residual=None`. The final RMSNorm ran twice, which for a
learned weight is that weight squared elementwise -- and this checkpoint's runs from -0.285 to
1.711, so squaring flips the sign of the negative channels. Fixed in d686d39e, verified red on the
bug and green on the fix, deployed: the output is unchanged, still `is is is`.

Three faults now, each real, each verified, none of them it. What they have in common is how they
were found -- a stack trace, a shape, a reading of the model's own forward -- and what none of
them had was a check that could have failed before deployment.

Alternatives killed, each by measurement rather than by argument:

    the checkpoint          both ends: config md5 0ecc077f, identical shard manifest
    the query shift         at a REAL 0. The first attempt at this control was void: the span
                            never read the flag -- `W_q` moved to the pool with the group cut, so
                            the pool decides the read point, and the flag was on the host. Wired
                            in bd4082eb, rerun, and the arrangement copies its input at 0 exactly
                            as at 1. NOW excluded
    the pool's weights      the pool serves " Paris." colocated from the same process
    the host colocated      cannot be run as a control: the host card is 31.36 GiB and the model
                            needs ~50, which is the premise of the arrangement and not a defect
    the pool's whole chain  cleared IN PROCESS. `benchmark/afd/span_moves_its_input.py` runs
                            prologue, middle span and epilogue over a tiny stack of the model's
                            own decoder classes: ||x-e||/||e|| goes 0.514, 1.094, 1.048 and cos
                            0.887, 0.687, 0.681. Every stage carries a contribution
    the host's tensors      all present and correctly shaped, under SGLANG_AFD_TRACE on the live
                            arrangement: q (122, 6144) = 24 heads x 256, k and v (122, 1024) =
                            4 x 256, attention output (122, 6144), norms growing 309 -> 4837 down
                            the stack. Nothing is empty and nothing is the wrong width
    the attention split     not implicated at PREFILL, where the host runs the fused path and
                            counts it (`self.fused += 1`). The fault is present at prefill

Copying the input means the final hidden state is approximately the input embedding -- the stack's
contribution is not reaching the residual stream. That is the next thing to localise, and the
instrument for it is the one this cut still does not have: the skill's check 3, a span run against
the model's own layers in one process, with a control that moves the boundary. Both bugs above
were found by a stack trace and a shape, not by that check, which is why they were found one at a
time and after deployment rather than together and before it.


## Read this before any number below about the GROUP CUT

The group cut -- sections 18, 19 and 20, and everything about spans -- **has never produced an end
to end number.** Its identities are verified and its parts are timed; the arrangement itself has
not once served a token.

    verified, against the kernel or a reference       what it says
    the read comes apart from the update  2.2e-03     a linear layer can wear a softmax layer's
                                          9.7e-08     interface
    the query coefficient                 8.5e-08     one contraction where there were two
    the one-pass kernel                   1e-06       260 us against the 252 of the kernel it
                                                      replaces, while emitting one more tensor
    a prefill chunk equals N decodes      exact       the difference three bugs turned on
    Early-K in the coefficient only       +0.013% bpb pre-registered, and it survived

    measured, on hardware                             what it says
    a span at batch 1..64                 2046 us     flat to 16; the bus size follows from this
    a span's parts                        361 / 200   feed-forward against linear attention
    the state read, split and fused       3.6-5.4x    why the fused kernel is required
    fp32 against bf16 for the state       11-25%      float32 stays

    NOT measured                                      why it matters
    step time, throughput, tokens         --          every figure in sections 18-19 is arithmetic
    token-identical against a local
    control                               --          the arrangement has never been shown correct
                                                      end to end

The per-layer cut (sections 1-17) is different: it ran, it was measured, and section 10's stopwatch
is why the group cut exists at all. Do not read a group-cut figure as if it had the same standing.

### 2026-08-21: it runs, and the output is wrong

Ten integration faults later the arrangement served its first tokens. It should not be read as
more than it is:

    prompt      "The history of computing begins"
    output      " begins begins begins begins ..."   24 tokens, all the last prompt token
    e2e         5.376 s for 24 tokens
    the pool    zero failed departures

**The plumbing works and the model does not.** Every fault before this one was a hang or a crash --
a shape, a name, a thread, a protocol key -- and none could be mistaken for a working system. This
one produces text.

The symptom names its own prime suspect. `linear_history.prefill_scan`'s docstring says what
treating a prefill chunk as a batch does: "runs every token against the same starting state, never
advances it, and produces a model with no memory of its own prompt". Repeating the last prompt
token is what no memory looks like. That is a hypothesis and it is written here as one; the state
is advanced in three places now -- the pool's conv ring, the host's HistoryCache, the deferred
OP_STATE_UPDATE -- and any of them failing to advance gives this.

A previous round of this project spent four rounds on the same symptom under a different
arrangement, concluded the pool was broken, and was wrong: the read point differed between the arms.
So the first move is to check what the ARMS are, not to hunt in the pool.

---

## 1. The read point costs what the study said, and only where the decode rule looks

    bits per byte     0.71301 -> 0.73240     +2.72%
    perplexity        8.7011  -> 9.2284      +6.06%

Reproduced to six decimals by a second, independent run, and again through the pool. It is the
UNREPAIRED checkpoint at a moved read point -- the forward-only arm, not a deployment.

Whether that cost is visible in generated text depends entirely on the decode rule:

| decode rule | model vs stock | its own noise floor | ratio |
|---|---|---|---|
| sampling, T=1.0 top-p 0.95 | MAUVE 0.7619 vs 0.7230 | 0.0461 (same model, 2nd sample) | **0.84x** |
| greedy | MAUVE 0.1929 vs 0.2685 | 0.0156 (same model, reordered batch) | **4.86x** |

Under sampling the difference is smaller than the metric's own noise and points the wrong way: a
null. Under greedy it is nearly five times the noise floor and points the right way. Greedy reads
the argmax, so any reordering of the top two logits changes the token and compounds; sampling
buries the same difference in its own randomness.

Greedy agreement, 500 prompts, 128 tokens:

    stock vs stock, requests reordered   110/500 identical, median 25 shared tokens
    stock vs shifted                      10/500 identical, median  3 shared tokens

**A serving claim has to say which decode rule it is making.** Neither number is wrong.

---

## 2. The partition is exact, and the controls are what make that mean anything

Attention is a mergeable aggregate, so sweep-then-join reproduces one fused call:

    float64, this model's shapes          5.8e-16
    boundary moved one position short     1.6e-1
    one position counted twice            1.8e-1
    runtime, bf16, 768 real joins         7.8e-3   (about two bf16 ulps)

The controls are the point. The merge is exact for ANY partition of the keys, including one that
drops a position -- the output stays a plausible attention over the wrong set -- so the boundary
is checked in index space, complete and disjoint, and the small number is read against what a
wrong boundary would have given.

**Refuted along the way:** I asserted a reversed cache would change the answer. It does not, 3.8e-16.
Attention is permutation-invariant over cached positions. Order does not live in the cache; it
lives in the rotation already applied to each key. The controls that DO bite are crossing the
key-value pairing (1.19) and projecting at the wrong positions (0.72).

---

## 3. Six of seven milliseconds were not the wire

A pool call was 6.5-7.1 ms and every explanation offered for it was wrong: not the network (0.3 ms
round trip), not the copies (d2h 0.011 ms, h2d 0.014 ms on a 40 KB frame), not the departure timer
(min_batch=1 departs immediately), not the feed-forward (267 MB of weights, 0.15 ms to read).

It was the pool's own sglang scheduler. A pool answers frames on a thread and never receives a
generate request, so its scheduler loop spins -- holding the GIL the departure thread needs.

    round trip, spinning scheduler        7.08 ms
    round trip, --sleep-on-idle           1.18 ms
    round trip, after removing three
      copies and one thread handoff       0.889 ms
    bare socket, no protocol              0.470 ms

End to end this moved decode from 6.4 to 21.4 tokens/s, and the Q-First arm from 6.5 to 25.5.
The pool now sets --sleep-on-idle itself, because nothing about the symptom points at it.

---

## 4. Q-First buys 1.46x across two machines, and only because the host was idle

    two-sided, fused        19.7 tok/s
    two-sided, q-first      28.8 tok/s     1.461x
    hidden per pool call    22.1%

The window is the sweep, launched between the feed-forward's issue and its collect. It exists at
all 63 converted layers -- 16 carrying a cache sweep, 47 carrying a linear layer's fused input
projection, which a converted layer runs twice anyway and which therefore costs nothing to move
inside the window. Going from 16 windows to 63 moved the two-machine ratio from 0.974x to 1.020x,
and the wire work above took it to 1.461x.

**Refuted:** I expected more concurrency to hide the round trip, since the host idles during it.
It does the opposite.

| requests | colocated | two-sided | ratio | colocated wall |
|---|---|---|---|---|
| 4 | 57.3 | 26.7 | 0.466 | 4.47s |
| 24 | 338.5 | 107.4 | **0.317** | **4.54s** |

Colocated's wall clock is FLAT from 4 to 24 concurrent -- the host has headroom at every level --
while two-sided's grows, because each departure's frame grows with the batch. The binding
constraint is not that the host has nothing to do during a round trip. It is that a decode step
contains **64 round trips in series**, one per layer, and concurrency does not shorten that chain.

---

## 5. Two frame streams, and only two

    1 stream    686 frames/s   1.452 ms each
    2 streams  1657 frames/s   1.128 ms each     2.41x, and LOWER latency
    4 streams   886 frames/s   4.591 ms each     1.29x
    8 streams   525 frames/s  15.482 ms each     0.77x

Two concurrent streams are superlinear and cheaper per frame -- a two-stage pipeline. Four
collapse. Any design that wants parallelism here should want exactly two of it.

---

## 6. Only half of a combined pool's work amortises

The feed-forward is weight-shared: one read of a layer's 535 MB serves every caller. The sweep is
not: each request reads its own history and there is nothing to share.

| context | feed-forward, 1 -> 64 requests | sweep | sweep's share of per-token cost at 64 |
|---|---|---|---|
| 512 | 59.5x | 29.5x | 19% |
| 4096 | 60.5x | 3.88x | 65% |
| 16384 | 60.4x | 3.44x | **88%** |

The sweep amortises only by keeping the card busy with one kernel instead of N launches, which
stops helping once it is memory-bound. **Putting the KV cache on the pool moves its dominant cost
onto work that shares nothing, and drags the amortising half down with it.**

**Measured wrong twice before being measured right.** A Python loop of per-request einsums reported
324 us where the traffic says 9; a batched einsum still materialised the grouped-query expansion,
six times the bytes, and reported 558; sglang's own decode kernel says 42. Two of three readings
were measuring my implementation and calling it the architecture, and both were plausible enough
to ship.

---

## 7. Where to draw the line through attention

    stock                        5.10s
    A  feed-forward on pool     18.15s     tokens identical
    B  + KV cache on pool       27.38s     tokens identical (2/2)
    C  + only W_kv on pool         --      implemented, does not run

B frees the host the whole cache -- 4.29 GB a request at 128k -- and pays for it by putting
non-amortisable work on the amortising service (finding 6) and by closing the Q-First window,
since the sweep it used to fill that window with has moved away.

C keeps the pool stateless and the window open at the cost of returning the key, value and gate
every layer: 1.9 ms a token of wire to move 1.34 GB of weights off the host. It fails on the
pool's rotation receiving 15 positions for 5 rows of hidden states -- the host's layer input and
its positions are not the same length on the path the wrapper sits in.

**The query does not need to cross the wire**, and removing it fixed a real divergence. With the
query sent, the merge took its log partition from the pool's query and its score from the host's,
and those differ -- different cards pick different FP8 kernels, 2.5e-4 here. End to end that
showed as one of two prompts diverging by a line break. Computing the score where the sweep is
made both prompts byte-identical.

---

## 8. What the measurements say to build

Two pools, because findings 5 and 6 point at the same shape from different directions: the
feed-forward and the sweep have opposite scaling and belong in different services, and the useful
number of concurrent frame streams is exactly two.

    weights pool   feed-forward. Stateless, shared, batched
    cache pool     a history per (request, layer), a sweep, an append. Stateful, per-request

They are concurrent because of the read point: the sweep needs the query, the query comes from
h_(l-1), and the feed-forward that produces x_l has not returned. Issue order is asserted directly
-- issue FFN, issue SWEEP, collect SWEEP, collect FFN -- because all four orderings produce the
same tokens and only one overlaps.

Cutting the stage from W_o to the key/value projection instead, with the query issued before the
MLP, adds a property that is not about speed: **the ordering becomes correct by construction.** The
query for layer l+1 goes out before layer l+1's key and value exist, so the swept history cannot
include this step's own token. The version this replaces arranges that in arithmetic, in the one
line of that file nothing else guards.

## 9. The ranking flips with context length, and my previous one was wrong

One decode layer has four pieces and only their ratio moves. The feed-forward, the query/output
projections and the key/value projection are flat -- they read weights, whatever the history is.
The sweep is linear in the context and unbounded. An arrangement is a choice of which pieces go
to another machine; its per-layer wall clock is the MAX of the two sides, not the sum, provided
the round trip is covered -- and how much work STAYS decides how deep a pipeline has to be to
cover it.

Batch 4, bfloat16, sweep charged to the 16 softmax layers rather than all 64:

| context | | colocated | A ffn remote | B ffn+cache | two pools | E sweep remote |
|---|---|---|---|---|---|---|
| **1024** | per layer | 429 us | **365 (1.17x)** | 375 (1.14x) | **365 (1.17x)** | 418 (1.02x) |
| short | depth needed | -- | 15.7 | 18.7 | 18.7 | **2.4** |
| **131072** | per layer | 952 us | 588 (1.62x) | 899 (1.06x) | **535 (1.78x)** | **535 (1.78x)** |
| half full | depth needed | -- | **1.7** | 19.0 | 19.0 | **2.4** |
| **262144** | per layer | 1482 us | 1118 (1.33x) | 1429 (1.04x) | **1065 (1.39x)** | **1065 (1.39x)** |
| full window | depth needed | -- | **0.9** | 19.1 | 19.1 | **2.4** |

    the host's KV cache at batch 4:  0.3 GB at 1k,  34.4 GB at 128k,  68.7 GB at 256k

Three readings:

**At short context nothing is worth disaggregating.** The best arrangement is 1.17x and needs
fifteen items in flight to reach it. The sweep is 41 us -- there is nothing to offload -- so
moving the feed-forward only trades compute for wire.

**At half and full window E is best on both axes at once**: the top speedup AND a depth of 2.4,
which a dataflow queue supplies from three items in flight. Two pools tie on speedup and need
nineteen. A degrades as the context grows -- the sweep stays on the host and grows with it, so at
the full window the host does 1118 us against the remote's 364 -- while its depth requirement
improves for the same reason.

**At long context the memory argument outweighs the speed one.** 68.7 GB of cache at batch 4 is
more than most cards have; an arrangement that moves it is not competing on 1.39x.

### The correction

My previous ranking charged the sweep to every layer. It runs on 16 of 64 -- three of every four
layers here are linear attention, whose state this does not sweep.

    what I said (sweep on 64/64)     colocated 730us   A 1.97x   E 1.73x
    what it is  (sweep on 16/64)     colocated 499us   A 1.34x   E 1.18x

A's advantage was a quarter of the sweep's cost being attributed to it four times over. The
measured table above supersedes it. This is the fourth arithmetic in this tree that was checked
only after it had already been used to recommend something.

## Not measured

- The two-pool arrangement end to end. The mechanism and its ordering test exist; two processes
  and two addresses do not yet.
- Arrangement C, which does not run.
- The 500-prompt agreement statistic on any of the pool paths. Two prompts is not evidence.
- Any of this at batch sizes other than 4 at long context, where the cache alone is 68.7 GB.
- Whether any of this pays. Every arrangement here is slower than colocated, because every
  converted layer makes a synchronous round trip the colocated model does not make, and the
  overlap so far recovers a fraction of it. Whether disaggregation pays is a ratio between an
  interconnect and a feed-forward, and one pair of machines answers it for one pair of machines.

## 10. The measured ladder contradicts section 9, and section 9 is the one to distrust

Section 9 is arithmetic over a cost model. This is a stopwatch on the two machines. They disagree
about the sign.

Qwen3.8-27B-FP8, host RTX PRO 6000 against the RTX 5090 pool over 10 GbE, 8 concurrent requests,
decode measured as the difference between a 36-token and a 4-token generation on a warmed prompt:

| context | colocated | A, feed-forward on the pool | A / colocated |
|---|---|---|---|
| 1024 | 170.1 tok/s (47.0 ms a step) | 40.7 tok/s (196.3 ms) | **0.240x** |
| 8192 | 171.5 tok/s (46.6 ms) | 41.0 tok/s (195.3 ms) | **0.239x** |
| 32768 | 176.9 tok/s (45.2 ms) | 63.2 tok/s (126.6 ms) | **0.357x** |

Section 9 predicted A at 1.17x at 1k and better beyond. Measured, A is between two and four times
SLOWER, everywhere. The gap is not small and it is not noise.

**Why the model was wrong is stated in the model.** Its per-layer figure is the MAX of the two
sides "provided the round trip is covered", and the same table says covering it at 1k needs 15.7
items in flight. Eight requests do not supply that, and nothing in the implementation pipelines
across layers to make up the difference. So the model reports a ceiling reachable at a pipeline
depth this code does not have, and the stopwatch reports what the code does. Both numbers are
correct about different things; only one of them is about the software that would be merged.

**The ratio does improve with context, and by less than it needs to.** From 0.239x at 8k to 0.357x
at 32k -- A's step time falls from 195.3 ms to 126.6 ms while colocated stays flat. Per converted
layer that is 2.33 ms of remote overhead at 8k against 1.27 ms at 32k. The obvious reading is that
a longer sweep hides more of the round trip, which is what the schedule is for; that reading is a
hypothesis until `overlap_report()` is read on this run, and it is not measured here.

**Colocated decode is flat from 1k to 32k** (47.0 -> 45.2 ms). Three of every four layers in this
model are linear attention, whose cost does not grow with history, so context is a much weaker
lever on this model than the budget model's sweep term assumes. An arrangement that needs context
to grow in order to win has less room here than section 9 suggests.

### The harness fix that changed the baseline

An earlier run of this ladder reported colocated at 304.7 tok/s at 1k. That number was wrong. The
warm-up ran only at the short length, so the timed short run absorbed the unfinished warm-up,
which shrank `t_long - t_short` and inflated the rate by 1.8x -- in the direction that flattered
the colocated arm. The warm-up now runs at the longest shape first and the timed pair is discarded
once. The same defect stopped the A arm outright, with 4 steps taking 33.5s against 36 steps
taking 8.4s, which the harness's own check caught and refused to report.

## 11. A second model family does not install, and the gap is granularity

`check_supported` was run against a built Llama-3.2-1B-Instruct stack. It refused, and what it
refused over is not a missing capability but a difference in where the same capabilities live:

    contract expects                     Llama has
    layer.attn                           layer.self_attn.attn
    layer.forward_prepare_* (four)       layer.self_attn.forward_prepare_native, _npu (two)
    layer.layer_communicator.prepare_mlp absent; the residual is done inline in the layer

Counted across this checkout by `python -m sglang.srt.afd.portability`:

    216 model files
     91 (42.1%) have input_layernorm
     32 (14.8%) have layer_communicator
     30 (13.9%) have prepare_mlp
      7 ( 3.2%) have forward_prepare_native
      1 ( 0.5%) have all four prepare variants -- qwen3_5, the model this was written against

The wiring installs on the intersection, so today that is one model file. A second family is a
port, not a flag: it needs the read point and the query projection located per family instead of
assumed. That is a design change to the wiring and it has not been made.

Running this check also exposed a defect in the check. A layer it could not classify ended that
layer's probe, so a new family saw the first gate only and would have learned the rest one launch
at a time -- the discovery process the module exists to replace -- and findings were listed per
layer, so one problem repeated sixteen times and truncated at twelve would hide a second and third
behind it. Findings are now grouped by message across layer ranges, and an unclassified layer is
probed against both kinds so the report says which side's names it does have.

## 12. The reversed arrangement, measured end to end, against a control that is the same model

Sections 9 to 11 rank arrangement E first at long context on the strength of a cost model. This is
a stopwatch, on two machines, with neither side quantised and with the tokens checked before any
rate was read.

Qwen3.8-27B in bfloat16 -- NOT the FP8 checkpoint the earlier sections used -- host RTX PRO 6000
against an RTX 5090 cache pool over a 10 GbE overlay, 8 concurrent requests, shift 1:

| context | stock | local split | E | E / local | local / stock |
|---|---|---|---|---|---|
| 1024 | 209.6 tok/s | 157.9 | 85.1 | **0.539x** | 0.753x |
| 8192 | 202.2 | 160.6 | 84.8 | **0.528x** | 0.794x |
| 32768 | 177.7 | 147.2 | 50.3 | **0.341x** | 0.829x |

    tokens identical across three prompts: local split == E

**The control had to change, and finding that out cost most of a session.** E was first compared
against stock and produced different text -- "The capital of France is Paris. The capital of France
is Paris." against stock's "...Germany is Berlin. ...Italy is Rome." That was read as a broken pool
and chased through the protocol, the cache and the kernels. It was the read point. Stock serves
shift 0 and E serves shift 1, and on a checkpoint nothing has repaired those are two different
models -- which the launch warns about, in a warning this tree wrote. Against a shift-1 local split
arm, E is token-identical.

So there are two costs and they were folded together in every earlier number:

    local / stock    0.75 - 0.83x    moving the read point and partitioning the attention
    E / local        0.34 - 0.54x    moving the sweep to another machine

Only the second is what disaggregation costs, and it is the one this section measures.

### The unquantised model is FASTER, which inverts an assumption

Colocated decode at bfloat16 runs 209.6 / 202.2 / 177.7 tok/s against the FP8 checkpoint's 171.7 /
168.4 / 174.4. Twice the weight bytes and it is quicker: FP8's block-wise dequantisation costs more
here than the bandwidth it saves, with CUDA graphs off and the Triton backend. Dropping
quantisation for comparability made the baseline harder to beat rather than easier.

### Where the time goes, measured on the pool rather than inferred

    a sweep, before batching       3609 us    of which 934 us was transport
    a sweep, after batching        1452 us    of which 934 us is transport
    an append                       632 us    fired and never waited on
    raw TCP echo, same bytes        972 us    a round trip's floor on this link
    NCCL send/recv, same bytes      746 us
    NCCL gather, one direction      151 us

The pool's per-request loop was 2670 us of the first figure and did not move when the context grew
from 1k to 8k, because it was per REQUEST. Replacing it with one contraction over a slotted buffer
is section 13. Transport is now 64% of a sweep and was 26%.

### What is NOT explained, and is therefore not concluded

E adds 2712 us per softmax layer at 1k, 2782 us at 8k, and **6549 us at 32k**. The pool's own sweep
grows only from 1452 us to 1788 us across the same range, so the jump at 32k is not the sweep. It
may be the slot buffer meeting its 33000-position ceiling, or host memory pressure changing the
batching. Until that is measured the 0.341x at 32k is a reading with an unexplained shape in it,
and this section does not rank anything on it.

Separately, 1260 us of the 2712 us at 1k has no owner: the pool answers a sweep in 1452 us and the
host loses 2712 us to it. The issue, the wait, the merge and the append's share of the link are
between those two numbers and none of them is measured yet.

## 13. Three defects the reversed arrangement's first real run exposed

**The verifier never covered the path it was needed on.** `--afd-verify-split` recomputes each join
the fused way and reports the worst disagreement. `_join_remote` returned above the call, so the
flag covered the LOCAL partition only -- the exactness figure of 7.8e-3 in the record is a local
number, and the arrangement whose entire premise is that the sweep happens elsewhere was the one
path nothing checked. Fixed, with the caveat that its reference reads the host's KV cache, which
the remote arrangement does not write during decode; the verifier is therefore still not usable on
E until the reference has a history to read.

**The pool's sweep was two loops deep.** A Python loop over tokens with a `repeat_interleave` of
the grouped-query expansion inside it, materialising 805 MiB of keys in float32 at 32k context per
token per layer; and a loop over requests around that. The first is the same arithmetic written the
expensive way for the THIRD time in this tree -- twice before in measurement tools, where it looked
like a number nobody believed, and here in the serving path, where it looked like an architecture
that had been refuted.

**The cache grew by concatenation.** `torch.cat` per append copies the whole history to add one
position. Replaced by a slotted buffer written in place, which is also what lets a batch be one
kernel. The slotted and per-request paths agree bit for bit in float64.

## 14. What the transport can and cannot be made to do

Measured on the link rather than assumed, after the recorded 4.4 Gbit/s turned out to be neither
the link's capacity nor what the protocol achieves:

    8 parallel streams, one way    9.91 Gbit/s     the physical ceiling; this is a 10 GbE overlay
    1 stream, one way              4.58 - 6.62
    request-response at 193 KiB    3.56            what any synchronous protocol gets
    our SWEEP_Q, transport only    1.69

    1-byte round trip, TCP          130 us
    1-byte round trip, UDP          139 us         the latency is the overlay's, not TCP's
    snd_cwnd under load          209 KiB           2.2x one frame: congestion control is not the limit

Four things were tried and three were refused by measurement:

    sharding one frame over N connections    873 -> 1577 us. Worse: the bytes in flight do not
                                             change and the syscalls multiply
    UDP                                      934 -> 2358 us. Worse: TCP segments in the kernel,
                                             UDP forces 142 datagrams into user space. 0% loss
    zero-copy and io_uring                   they act on 20.7% of the round trip of which 8.9 us
                                             is our encoding; the ceiling is about 3%
    FP8 on the wire                          would halve the payload and change what the model
                                             computes, so a timing taken with it is not comparable
                                             to one taken without. Withdrawn

The binding constraint is the bandwidth-delay product: 9.91 Gbit/s at 130 us is 157 KiB, and one
sweep is 96 KiB out and 97 KiB back. A message smaller than the pipe cannot fill it, and this is
Little's law rather than an implementation.

**The one number that changes the conclusion is NCCL's gather at 151 us**, against 746 us for its
own send/recv and 934 us for ours. That is one DIRECTION of 96 KiB at close to line rate, and it
says the round trip costs 2.5x two one-way transfers. The lever is therefore to stop taking round
trips -- issue layer l+1's query while layer l's output is still arriving -- and that is worth more
than any change of library. For the arrangement that does not exist yet, several hosts sharing one
pool, gather is also the right primitive for the inbound and `all_gather` at 571 us is the wrong
one.

Neither machine has InfiniBand or libibverbs, so NCCL falls back to sockets and GPUDirect is
unavailable: the bytes still travel GPU to host to socket to host to GPU. What was measured is
NCCL's C++ path and its multiple sockets, not a shorter route.

## 15. Communication against the work it displaces, and the ratio that decides everything

The arrangement's whole proposition is that a feed-forward is worth sending away. So the number
that decides it is the round trip against the feed-forward it buys, and every part of that was
measured on this pair rather than derived.

One routed feed-forward, taken apart. The host's own measurement of the same feed-forward
computed locally is 360.8 us, which is what the pool's column should be compared against:

| batch | round trip | wire | pool's ffn | our protocol | wire / ffn |
|---|---|---|---|---|---|
| 1 | 716 us | 221 | 379 | 116 | **0.58x** |
| 4 | 987 us | 480 | 359 | 148 | **1.34x** |
| 8 | 1152 us | 707 | 361 | 84 | **1.96x** |
| 16 | 1520 us | 970 | 364 | 186 | **2.67x** |

**The feed-forward does not grow with the batch and the wire does.** A decode-time feed-forward is
a weight read: the same 510 MiB whether one row rides along or sixteen, so it sits at 360 us in
every row of that table. The payload is `batch x hidden x 2 bytes` each way and rises linearly.

That is the arrangement's central tension in one table. The host wants a large batch, because its
own weight reads are fixed and every extra row is free -- measured, batch 1 to 8 at a flat 49 ms a
step and 7.94x the throughput. The wire wants a small one. They are the same number.

The crossover is at batch 2: below it the wire is cheaper than the work it displaces, above it the
wire costs more than the work is worth. Every deployment-shaped configuration is above it.

### What a better fabric does to that ratio

Wire at batch 4, against the same 359 us of feed-forward:

    this overlay, measured     480 us     1.34x
    10 GbE RoCE                 69 us     0.19x
    25 GbE RoCE                 29 us     0.08x
    100 GbE RoCE                10 us     0.03x

RDMA does not make the arrangement fast. It makes the wire stop being the subject: at 0.19x the
question returns to whether one weight read serving several hosts is worth a round trip, which is
a question about how many hosts there are and not about the network.

### Our own protocol is 8 to 15 percent and is not where the cost is

The `protocol` column above is everything this project adds on top of the wire and the work: frame
headers, the device-to-host copy, the departure queue, the reply's copy back. It is 84 to 186 us.
Four transports were tried against it and three were refused by measurement -- sharding a frame
across connections (worse), UDP (2.5x worse, 0% loss, lost on syscalls), NCCL send/recv (970 us
against our 934, level). The remaining candidate is a collective spelling that measured 11% better
and costs a fixed communicator, which a pool serving dynamic callers cannot have.

## 16. The measurement that was wrong for four rounds, and what it was hiding

Every figure in section 12 and every "the arrangement is N times slower" statement before it was
taken with the pool waiting for eight callers. That setting was made deliberately, for the
amortisation test in section 17, and never put back. There has only ever been one caller, so every
layer waited out the full 4 ms departure timeout.

    pool setting            outstanding   host blocked   window hid   decode
    min_batch 8 (stale)        5.70 ms        5.19 ms         9.0%    2.0 tok/s
    min_batch 1 (correct)      1.06 ms        0.85 ms        20.0%    8.1 tok/s

**Four times.** And `outstanding` at min_batch 1 is 1.06 ms against the 987 us the same round trip
measures when probed on its own -- a 7% gap, where there had been a 5.8x one.

What it invalidates: the four "real host with synthetic fillers" numbers (2.0 / 1.9 / 2.2 / 2.0
tok/s), the conclusion drawn from them that batching does not reach a real host, and the estimate
built on that conclusion that layer alignment would need hundreds of hosts. The amortisation
measurement itself stands, because there the setting and the intent agreed.

This is the fourth failure of the same kind in this arrangement's history -- a verifier that never
covered the remote path, a construction hook that reached one of sixteen loaders, a module flag
that did not cross a process boundary, and now a setting left over from another experiment. Each
produced a plausible number and nothing that could have said otherwise. The fix is the same in
every case and it is not a code fix: **a measurement has to report the configuration it ran under,
in the same breath as the number.** The tools now print the pool's departure settings before their
first line.

## 17. What the window actually hides, measured across context

On the clean baseline, the same host, the same pool:

    prompt tokens   batch   ms a step   tok/s   window hid
               33       1      136.7ms     7.3      20.0%
              833       1      114.1ms     8.8         --
             6657       1      109.8ms     9.1      21.9%
            28449       4      126.9ms    31.5      18.7%

The step time falls as the prompt grows, which is the direction the corrected model predicts: the
host's sweep is the one term that grows with context, and it is what fills the window.

**But the hidden fraction does not grow.** It sits between 18.7% and 21.9% across a range where
the sweep alone should have moved it from about 5% to about 25%. The window is real -- the host
logs 63 of them a step, 16 carrying a cache sweep -- and it is not scaling the way the sweep
scales. That is unexplained, and the reason it is written here rather than resolved is that the
last four rounds of this arrangement's history were spent explaining numbers whose configuration
was wrong. This one is on a clean baseline and it still does not fit.

Until it is understood, the corrected model in section 14 predicts break-even at 71,000 tokens on
this overlay, and the longest context this hardware can hold on the attention side is 28,449 at
batch 4 -- the host is a 32 GiB card carrying 14.9 GiB of weights. **The arrangement cannot be
shown to break even on the machines it was measured on.** That is a statement about the machines.


## 18. The group cut, and the memory that decides it

Section 17 ends by saying the arrangement cannot be shown to break even on these machines: the
corrected model puts break-even at 71,000 tokens and the host can hold 28,449 at batch 4, because
it is a 32 GiB card carrying 14.9 GiB of weights. That sentence is about how much of the host is
NOT cache, and the group cut changes exactly that.

### The cut

A span runs from one softmax attention's output projection to the next one's input: on this model
four feed-forwards and three linear attentions, in one call. 17 round trips a decode step instead
of 63.

The rule is not "weights to the pool". It is that **a batch only has to be re-formed where latency
varies**:

| stage | cost depends on context? | measured |
|---|---|---|
| feed-forward | no | a weight read |
| linear attention | no | 7.0 us at 1k, 32k and 256k alike |
| softmax attention | **yes** | 19 us at 1k, 2397 us at 128k |

Through a fixed-latency stage everybody in a batch finishes together and holding the batch is free.
Through a variable one, one long-context rider makes every short one wait. So the cut goes where
latency stops being fixed, and on this model that happens to be where the KV cache is.

The per-layer arrangement cut by OWNERSHIP instead -- a recurrent state belongs to its request, so
it stays with the request -- which is why 48 linear-attention layers stayed on the host there. Their
state is per-request AND their latency is fixed; under this rule the second fact decides, and they
move. **That is what frees the memory**, and it is the reason the two cuts are different
arrangements rather than the same one at two granularities.

### The span, measured before anything was built on it

`benchmark/afd/span_cost.py`, synthetic weights at the real shapes, so the launch count is real:

    riders     span     a rider    over the sum of its weight reads
         1   2070 us   2070 us     455 us
         4   2046 us    512 us     431 us
        16   2249 us    141 us     634 us
        32   2849 us     89 us    1234 us   <- no longer launch overhead
        64   3927 us     61 us    2312 us

The arithmetic over weight bytes said 1683 us. Measured is 2046 at batch 4, **22% higher**, and the
flat part of the curve ends at 16 -- which is how large a batch should be, measured rather than
chosen. Above 16 a span stops being a weight read and starts being a matrix multiply.

Its internal split, which answers who pays for the wire:

    4 x feed-forward     2040 MiB   74% of the span   361 us each   measured
    3 x linear attention  660 MiB   24%               200 us each   measured
    W_o                    60 MiB    2%                35 us
    one round trip                                    628 us

(The per-layer figures were first divided out of the weight bytes as 298 and 129. Measured they are
361 and 200 -- 22% and 55% low. `benchmark/afd/span_parts.py`.)

A span covers a round trip 3.3 times over. The per-layer cut's feed-forward covered it 0.57 times
-- 359 us of work against 628 us of overhead -- and that ratio is the whole of why section 10
measured a loss. **5.7x more work per round trip** is the change.

### What the host stops holding, and what that is worth

Computed from the checkpoint's own shapes, anchored on the measured 28,449 at batch 4:

| the host holds | weights | freed | tokens a request at batch 4 |
|---|---|---|---|
| per-layer cut (measured) | 15.81 GiB | -- | 28,449 |
| group cut | 5.49 GiB | 10.31 | 70,689 |
| group cut, and the query projection also on the pool | 2.37 GiB | 13.44 | 83,489 |
| | | | *break-even: 71,000* |

The computed 15.81 GiB against the measured 14.9 is 6% out, so these are estimates and the last
column inherits that. But the gap between the rows is far larger than the error in any of them, and
it says something section 17 could not: **the group cut lands ON the break-even point and does not
clear it.** 70,689 against 71,000 is a coin toss.

Moving the query and key/value projections to the pool as well is what clears it, by 18%. It also
finishes the GEMV finding -- host decode is weight-read bound, and `qkv_proj` and `o_proj` are the
last two GEMVs the host still runs at decode. After that the host runs no weight matrix at all: it
holds a cache and sweeps it.

Two ways to arrange that, and they differ by less than they look:

    A  host drives   host sends o, receives q early and then k, v      14,336 columns a group
    B  pool drives   pool sends q early, host returns o_hist and lse   12,312 columns a group

B is 2,024 columns cheaper, which at batch 4 is 16 KiB and **13 us on this link** -- 0.6% of a
span. It costs the host's forward pass: under B the host no longer runs sglang's model loop at all,
it is a cache server. A is taken. Thirteen microseconds is not worth a fork.

Both take the key/value append off the critical path, which `OP_APPEND` was written for and nothing
had used: this step's join uses key and value the caller already has, and the cache only has to
hold them by the NEXT step.

### What is not yet measured

Everything above about the group cut except the span itself. The 17-round-trip step time, the
break-even, and the token budgets are arithmetic, and section 10 exists to record what happened the
last time arithmetic and a stopwatch disagreed here. The sweep window is also still shut --
`SpanRouting.report()["sweep_window_open"]` is False -- so a timing taken today would be of a
schedule with the overlap removed.

## 19. Both ends run at once, so the comparison is a max -- and section 18's are not

Section 18 reports step times built by ADDING the pool's work to the host's. The two machines run
at the same time; the arrangement's step time is the larger of them, not their sum. Every figure in
section 18 derived from a sum is superseded here, and so is every multiple anywhere above that was
taken against the per-layer cut's 126.9 ms.

    context      pool     host      max     host busy
      1,024   37.3 ms   0.3 ms  37.3 ms           1%
     28,449   37.3 ms   8.3 ms  37.3 ms          22%
    131,072   37.3 ms  38.4 ms  38.4 ms         100%

    colocated, bfloat16, measured               ~38 ms

**The ceiling of this arrangement is parity with colocated, and it reaches it.** 30.0 of the pool's
37.3 ms is reading the model's 50 GiB of weights once. A colocated server reads the same 50 GiB. No
arrangement of two machines makes that read smaller, so no arrangement of two machines beats one
machine at weight-bound decode.

Everything reported earlier as "N times faster" was against the per-layer cut's 126.9 ms. That is a
comparison against a bad implementation, not against a baseline. Fixing the per-layer cut does not
beat colocated; it returns to colocated.

The two ends balance near 131k context: below it the pool is the bottleneck and the host idles,
above it the sweep is and more pool does not help.

### What the idle host is for

The host is 1% to 22% busy across the range that matters, which is capacity rather than waste. The
same host, time-sharing more requests:

    context   requests that fit   pool ms   host ms   bottleneck   tok/s
      8,192            54           62.9      32.4      pool         858
     28,449            15           39.9      31.2      pool         376
     65,536             6           37.4      27.6      pool         160

At 8k that is **7.9x the throughput of batch 4**, and the host is still not the bottleneck at any
point in the table. What stops it is not time but KV memory.

### Where the projections go: a trade worth 3% either way

Moving the query and key/value projections to the pool costs 1.87 ms a step on the side that IS the
bottleneck and gives the host back 3.12 GiB. Whether that pays depends on whether the freed memory
buys another whole request:

    context   projections on the pool   projections on the host
      8,192          858 tok/s                 833 tok/s
     28,449          376                       371
     65,536          160                       169

Pool-side at the lengths that fit meaningful concurrency, host-side at 65k where 3.12 GiB does not
buy a seventh request. **The whole question is worth 3%**, which is inside this model's error, and
it is settled on the pool for a reason that is not throughput: it leaves the host holding no weight
matrix at all.

### What that is actually worth: the host stops needing to be a big card

    host weights   8 GiB   12 GiB   16 GiB   24 GiB   32 GiB
    per-layer cut, 15.81   ----     ----     ----     ----     98,959 tokens
    group cut,      2.37   ----     ----     57,016   188,088  319,160

Under the per-layer cut nothing below a 32 GiB card could be a host at all -- the weights alone did
not fit. Under the group cut a 16 GiB card is a host.

Whether an 8 or 12 GiB card is one depends on a number this has not measured. The table above
subtracts 10.2 GiB of non-weight, non-KV footprint, inferred from a single anchor: the per-layer
cut on a 32 GiB card held 14.9 GiB of weights and 6.95 GiB of cache. But that anchor was taken with
the host running the WHOLE model, feed-forward activations at intermediate_size 17408 included, and
under the group cut the host runs attention and nothing else. The footprint should be far smaller
and it is not known how much smaller. **Measure it before quoting a card.**

The consequence worth more than the card price: a host that holds no weights makes KV capacity
horizontally scalable. The pool is the bottleneck, it is sublinear in batch, and its departures
already take riders from any socket -- so several cheap hosts against one pool needs no protocol
change, and the 50 GiB read is still done once.

### Every number in this section is arithmetic

The only measurements here are the span (2046 us at batch 4) and the anchors it is combined with.
The host's sweep is extrapolated linearly in batch from one point, which batched attention kernels
almost certainly beat -- so the host is likely to have MORE headroom than shown, not less. Section
10 records what happened the last time a cost model and a stopwatch disagreed in this tree.

## 20. Which key may be early, decided against a rule fixed before the numbers

`benchmark/afd/EARLY_K_PREREGISTRATION.md` states the conjecture and freezes the decision rule.
This section records the outcome. The rule is repeated here as it was written, not as it reads
after the fact:

    SURVIVES        mixed within 0.5% of exact in bits per byte, AND all_early at least 4x
                    further from exact than mixed is
    FAILS           mixed worse than 0.5% -- the coefficient's key is not free either, and
                    Early-K is unavailable at any granularity
    UNINFORMATIVE   all_early ALSO within 0.5% -- the corpus is too short to show a state error,
                    which saturates only after about 1/(1-alpha) steps, and the question has to be
                    re-asked over a long generation

### The conjecture

A key that feeds a state REUSED ACROSS TIME must be the current one; every other use may take the
early one. One operator appears twice in a linear layer:

    P(k) = I - beta k k^T

    state    S_t = alpha S_(t-1) P(k) + beta v k^T      survives the step
    output   o_t = alpha S_(t-1) P(k) q + beta (k.q) v  does not

so the arms are three evaluations of the same operator: `exact` current in both, `mixed` early in
the output only, `all_early` early in both.

### Two defects the arms had before they measured anything

Both would have produced a publishable-looking number about something else, and both were caught
only because the pre-registration required the `exact` arm to reproduce the stock function first.

**A transposed state.** transformers holds the recurrent state as (heads, KEY dim, VALUE dim) and
sglang holds it as (value heads, head_v_dim, head_k_dim) -- its own comment says "to match what the
decode kernel expects". Both dimensions are 128 on this model, so no shape check can tell them
apart. Written against the wrong one, the state was 0.96 out and the model disagreed with itself on
eight positions in thirteen.

**A whole-sequence forward.** The harness scored each document in one call, which takes the CHUNKED
prefill kernel. The arms patch the recurrent step, which that kernel does not contain, so all three
would have returned the SAME number -- and a null result reads as "the approximation is harmless".

The second is the more dangerous shape and it is worth naming: a check that cannot see the thing it
is checking fails by passing.

### Result

12 documents of fineweb-edu, 72,261 bytes, 15,457 tokens, scored one token at a time:

    arm            bits per byte     delta     percent
    exact                 0.6232    0.0000       0.00%
    mixed                 0.6233    0.0001       0.01%
    all_early             2.3684    1.7452     280.04%

**The conjecture SURVIVES, and not marginally.** `mixed` is 0.013% from `exact`, which is
thirty-eight times inside the 0.5% the rule allowed. `all_early` is 280%, which is twenty-one
thousand times further from `exact` than `mixed` is, against the 4x the rule asked for.

The third branch did not fire and could not have: `all_early` is not close to 0.5%, so the corpus
was long enough to show a state error. 6,000-character documents run about 1,300 tokens, which is
past the 1/(1-alpha) saturation `state_precision.py` measured.

An honest reading of the small number: `mixed` is 4.2 nats worse over 31,214, in the direction
expected of an approximation, on one sample. The claim is that it is far inside the threshold, not
that it is exactly zero.

### What it means

The early key may be used in the query coefficient and must not be used anywhere the state is
advanced. Concretely, in one linear layer:

    q~ = P(k_early) q       formed on the pool, from h_(l-1), one feed-forward early
    S_t = alpha S P(k) ...  the current key, always

So the whole round trip fits inside the previous feed-forward's 361 us: the coefficient is ready
before `x_l` exists, the host's single contraction runs while the pool spends that feed-forward,
and the reading is back before it is needed. The convolution ring is protected the same way -- the
early key is convolved against it WITHOUT writing, because three steps of reuse is still reuse.

`all_early` at 2.37 bits per byte is worth stating plainly: the model is destroyed, not degraded.
That is what feeding an approximate key into something that compounds does, and it is why the
split is where it is rather than being a matter of taste.

### What this does NOT license

That `mixed` is free. It costs a second key projection on the pool -- about 12 us a layer, 0.56 ms
a decode step -- and buys latency, not throughput: under a max-of-both-ends accounting the pool is
the bottleneck below about 131k context, so hiding a round trip inside a feed-forward improves the
per-request latency and leaves the step time where it was. Whether to take it is a separate
question from whether it is correct, and only the second was measured here.
