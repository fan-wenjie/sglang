# AFD with an Early-Q read point: what was measured

Everything here was run on Qwen3.8-27B-FP8. Host is an RTX PRO 6000 Blackwell (97 GB); the pool,
where a second machine is involved, is an RTX 5090 (32 GB) across a 10 GbE link with a 0.3 ms TCP
round trip and 4.4 Gbit/s of bulk bandwidth. Numbers with no arrangement named are colocated.

Findings are grouped by what they decide. Several of them refuted the hypothesis that motivated
them, and those are marked, because a refuted hypothesis that stays in the record is the only
protection against re-adopting it.

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
