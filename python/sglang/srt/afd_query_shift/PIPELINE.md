# The stateless pipelined span

> **The tour-group model (2026-08-25, the user's formulation; the concurrency milestone's
> design).** One agency = the pool; a coach = a departure; fixed-itinerary cities = the
> context-free stages (feed-forward 361 us, linear attention 200 us); the free-roam city = the
> softmax sweep (19 us at 1k to 2397 us at 128k), where each request leaves the group, takes
> its own time against its own KV, and boards whichever of the agency's coaches next passes.
> Several groups are on the road at once. The pool half of this exists (`boarding.py` re-forms
> at every variable-latency boundary); the host half is the open build: today a scheduler batch
> sweeps in lockstep, so the longest context paces everyone. Any coach can carry any tourist
> BECAUSE the pool is stateless -- nothing of the request rides the bus -- which is also the
> whole basis of multi-pool routing. Pre-registered check: at mixed contexts, a short request's
> p99 approaches its solo latency instead of the longest passenger's.

> **2026-08-25, 04:00: the push arrangement wins, and the loss's history closes.** The user's
> redesign -- answer the EARLY frame with the reading instead of stashing it, assemble `core`
> on the pool from the query it kept, send the advance back unanswered as OP_STATE_APPLY --
> removed the blocking round trip itself, which no amount of cheapening its halves had touched.
> Five alternated pairs, all won, every push flight faster than every mech flight of the night
> but one:
>
>     push (rung 2, final)   32.36  32.90  32.78  33.34  29.02
>     mechanism (rung 1)     33.35  43.93  45.50  43.30  35.36
>
>     against mech's fast mode   about -11%
>     same-conditions adjacent   -35% (verified pair: 160 = 160 tokens, so the walls compare)
>
> The path there, each step measured: fused cook (twenty launches to one, 0.573 -> 0.165 ms
> live), the streamed wire (cudaLaunchHostFunc writes the socket from a driver thread; a
> per-fd mutex shared with Python's send_frame), one-query-two-keys (the mix projects [k|v|z]
> only; the ring's q channels advance with the early query's raw column), the push protocol
> (EARLY answered, MIX_READY retired and its number burned, APPLY unanswered with the next
> EARLY behind it for order), and the history as a partial sum (one 20 KiB column a layer
> serves both cooks; wall-neutral at batch one, 3x less down-wire for concurrency).
>
> What remains on the critical path: the reading's lateness (~0.9 ms a layer -- the
> early-contract-reply loop is ~1.1-1.4 ms against a 0.46 ms feed-forward window at TCP;
> RDMA folds it into the window), an unexplained bimodal issue-clock (1.7-2.0 ms in some runs,
> 0.07-0.17 in others, uncorrelated with wall time so far), and the machines' own ~10 s
> bimodal mode that made this night's absolute numbers a lottery and pairwise diffs the only
> currency.
>
> Validation is two axes, now written into the docs: CONSISTENCY against the serial
> same-semantics reference (the equivalence tests and the in-process round trip, within the
> declared reassociation regime), and APPROXIMATION against the standard model (the mechanism's
> +5.01% bits per byte and the linear operator's price, re-measured by the `serving` arm for
> the final semantics -- a property of a non-native checkpoint, absorbed at training time by a
> native one).

> **2026-08-25: the ladder's verdict, a retraction, a new accuracy figure, and a rebuild.**
> Everything below this block predates the attribution ladder; read it as history with these
> corrections over it.
>
> **The ladder** (one change a rung, each against an adjacent shift-0 baseline, replicated):
>
>     rung 0  shift 0's arithmetic at shift 1        -0.94%   the zero-difference control
>     rung 1  + the attention query read early       -3.56%   THE DESIGN'S OWN MECHANISM: free
>     rung 2  + the raw early frame, dropped        +11.53pp  the frame alone, nothing gained
>     rung 3  + the host contracts and the mix uses +10.69pp  preparation on the host
>     rung 4  + the early frame advances the ring    -0.52pp  noise; the write was overwritten
>                                                             by the mix's own, same token
>
> The 18.6% loss was never the design's. It was the linear layers' early read as built: the
> host prepared raw materials -- convolve 0.327, normalise 0.247, coefficient 0.088 ms -- against
> a 0.461 ms window, so the schedule's max moved to the host, and e = max(host, pool) priced the
> overflow on every layer. Rung 1 alone, four replicated runs, is at worst even with baseline.
>
> **Retraction: the "+17.9% for the mechanism alone" comparison.** SGLANG_AFD_NO_EARLY_STATE_READ
> cut the chain at `issue_host`, AFTER the pool's projection, packing and send -- so it measured
> a third configuration (rung 2-3's costs, none of their machinery removed), not the mechanism.
> The ladder cuts at the source; its rung 1 replaces that number.
>
> **The mechanism's accuracy, measured on the deployment for the first time** (bits per byte of
> the served model over a fixed passage, prefill logprobs; deterministic -- replicates agree to
> the last digit):
>
>     natural prose    1.559006 -> 1.637043    +5.01%
>     shuffled words   1.393054 -> 1.477491    +6.06%
>
> This is the SPAN mechanism's own approximation -- the next attention's query and gate projected
> from the hidden state one feed-forward early -- which the four-arm harness never touched (its
> arms price the linear operator's early read, +0.38%/+0.90%). Five percent on a model not
> trained for the shift is the strongest argument yet that `--afd-query-shift-layers` is for
> NATIVE query-shift checkpoints, and dangerous on everything else, exactly as the flag says.
>
> **The rebuild.** The linear early read is not structurally lost -- the loss was preparation on
> the wrong side. Rebuilt cooked: the host attaches each span's convolution windows (the ring's
> last three columns; the ring stays the host's, the pool still holds nothing between calls),
> the pool convolves/normalises/forms the coefficient and sends FINISHED materials
> (OP_STATE_EARLY carries q~ and q; OP_STATE_MIX_READY carries k, v, gates and the ring's next
> column), and the host's handlers are one contraction and one assembly. Every relocated piece
> is held to BIT equality with the sequence it replaced by
> `test_afd_cooked_is_the_same_arithmetic`; `TheHostHalfDoesNotCook` pins the rule statically
> and `span_routing`'s tripwire warns whenever the schedule's max lands on the host at runtime.
>
> **2026-08-25, later: the cooked rebuild was measured and lost too -- +23.3%** (55.40 against
> the mechanism's 44.93, three alternated pairs). The host's half worked exactly as designed
> (mix 1.848 -> 0.566 ms; the contraction absorbed by the feed-forward window; the max stayed
> on the pool), but each early send cost 0.95 ms of the pool's serial path against 0.16 ms in
> isolation -- interpreter-lock stretching from the reader and sender threads, a cost class the
> schedule's model does not carry. With the movable work at 0.38 ms a layer, even a zero-cost
> cook leaves the residual above the gain, so this is structural on a Python runtime at batch
> one, and the cooked rung now defaults OFF. Three implementations, one wall: raw-on-host
> +18.6%, cooked-on-pool +23.3%, theoretical floor still positive.
>
> **Two structural facts the ladder settled.** The linear layers' early-read gains are a
> CONSTANT (their compute does not grow with context; only the softmax sweep's does), so rungs
> 2-4 could never be bought back by longer prompts -- and native query-shift models change the
> accuracy of the shift, not its speed, so no future checkpoint changes this ledger either.

> **The accuracy figures below were wrong and are replaced.** -0.016% and +238.5% were measured
> by an arm that left the QUERY current while the key inside `P(k)` was early; the deployed
> arrangement forms the whole coefficient from the early projection. The arm has been made to
> compute what the code computes, and re-run:
>
>     exact                0.6272 bits per byte
>     mixed                0.6296     +0.38%     the early query and key in the coefficient
>     mixed_reused_norm    0.6329     +0.90%     and the previous layer's norm reused
>     all_shifted          4.0575   +546.89%     the control
>
> Twelve fineweb-edu documents, teacher-forced one position at a time through the recurrence, and
> the kill condition was +2.68%.
>
> Two things to read off it. The arrangement costs 0.38%, not nothing -- the old figure flattered
> it by an order of magnitude. And reusing the previous layer's post-attention norm, which saves
> 0.058 ms a layer, costs 0.52 points on top of that: a third of the whole budget for a twelfth of
> a millisecond, which on the current ledger is not a trade worth making.
>
> The control's +547% against its old +238% is the same correction from the other side: with the
> query shifted as well, letting the early tensors into the state is worse still.

> **Verdict, and read this before the design below.** The arrangement was built, it runs, and on
> the deployment it was built against it is **14.2% SLOWER** than the standard read point --
> 51.630 s against 45.222 s, two runs an arm, alternated, against a within-arm spread of 2.7 s.
>
> The mechanism is not absent and the design is not wrong: the host improves by exactly what was
> predicted. The pool pays more. The extra projection is real GPU work on the POOL, competing with
> the feed-forward it was meant to hide behind -- the overlap is with the host, the cost is on the
> pool's own card -- and that is invisible to every counter here, because nothing on this path
> synchronises and all of them measure launch time. They explain a third of it.
>
> The verdict is conditional. The saving is on the host and grows as the host weakens relative to
> the pool; the projection does not move. This pair is a 5090 against a PRO 6000, closer to parity
> than the small-card/large-card arrangement intends, so it should be re-taken further from it.
>
> Two sections below describe designs that were REFUTED rather than built -- the ring travelling
> as a call parameter is the main one, killed by its own traffic. They are kept because the
> measurement that killed each is the useful part. What was actually built is
> "The arrangement that was built, and what it measured".


How a group of layers is served so that the pool holds nothing per request AND the two ends
compute at the same time. Those two goals were in conflict in every arrangement before this one,
and the conflict is recorded in `span.py`'s own `_issue_read`, which refuses to open a window
"whenever `mix_host` is set -- the stateless arrangement sends the whole layer, and there is no
separate read to move".

Every number below is either MEASURED on the two-machine deployment (pool: RTX PRO 6000, host:
RTX 5090, same-rack LAN) or marked DERIVED. Derived means scaled from a measured whole by channel
count, and derived numbers are the ones to check first if this does not behave as written.

## What was wrong with each arrangement before

    ring on the pool        the pool remembers a request's last K projections, so the request is
                            STICKY: its next call must come back to the same pool. Multi-pool
                            routing, pool replacement and horizontal scaling all die here
    ring on the host,       the pool sends the whole layer in one frame and blocks. Measured:
    whole layer one frame   `work 0.000 ms, 99% of the wait`. No overlap at all
    ring on the host,       the read can be issued early, but then the host holds v and the pool
    separate early read     cannot mix locally -- v would have to cross back

## The arrangement

The ring travels as a CALL PARAMETER. The pool convolves, so it has v; it keeps nothing.

    host --- span frame, carrying the group's THREE rings ------------------> pool
                                                                             W_o
                                                                             norms
                                                                             project q,k  (early)
                                                                             convolve READ-ONLY
    pool <-- q~ ------------------------------------------------------------ pool
             host contracts S q~        |  pool runs the head's feed-forward     <-- the overlap
    pool <-- S q~ ----------------------------------------------------------- host
                                                                             project k,v,z
                                                                             convolve WRITING
                                                                             core = a(S q~) + s v
                                                                             norm(core, z), W_o
    host <-- k, v, advanced ring -- deferred, no reply --------------------- pool

The pool writes nothing per request at any point. The ring it convolves against arrived with the
call and is discarded after; the host advances its own copy from the k and v it is sent.

## CORRECTION: the ring was refused on the wrong comparison

The section below, and the one after it, killed ring-lending by weighing its traffic against the
TOTAL callback blocking in a token -- 30 MiB against 107 ms at eight callers, "71% of what we are
trying to save". That is not the comparison. What lending BUYS is the host work it removes, and
the two numbers are not close:

    1 caller     ring 3.75 MiB = 5.9 ms      host saves 95.8 ms      net +89.9 ms a token
    8 callers    ring 30.0 MiB = 47.0 ms     host saves 95.8 ms      net +48.8 ms a token

The host saving is what moving the convolution off it is worth: 2.212 ms a layer becomes 0.217 --
the contraction alone -- across 48 layers. It does not amortise across callers the way a callback
does, because it is serial work on one thread.

The schedule in the design says the same thing from the other end. Its recurrence is

    e_l = max(e_(l-1) + a, ffn_(l-1))

-- ONE thing on the host a layer, of duration `a`, and the sweeps abut. `a` is the contraction,
0.217 ms MEASURED. This implementation gives the host 2.212, because the convolution, the
normalisation and the coefficient are all on it, and the design puts those on the pool.

What follows below is kept because the traffic numbers in it are right; the conclusion drawn from
them is not.

## Why the ring rides the span frame instead of going on its own

Because the cover for the FIRST ring is only what runs between the span frame arriving and the
read-only convolution -- and that is short:

    W_o, 24x256 -> 5120                            23.2 us   MEASURED
    add_and_norm (residual add + RMSNorm)          14.3 us   MEASURED
    input_layernorm                                 9.4 us   MEASURED
    projection of q,k, sliced                      15.9 us   MEASURED
    ----------------------------------------------------------
    cover                                          62.8 us

    one ring on its own frame, 80 KiB             337 us    MEASURED, round trip
    the same ring bundled into the span frame     198 us    MEASURED, marginal bytes only

Neither number is a margin against 62.8 us, and that settles it: a ring cannot arrive inside the
first window at all. It rides the span frame because then it does not have to -- it arrives WITH
the frame, and the window stops being a constraint instead of being met.

The cover is also optimistic: `W_o`'s head count is assumed, the norms are `torch.nn.RMSNorm`
rather than the model's own, and nothing counts the Python dispatch around them.

The two norms matter more than they look: 23.7 us of a 62.8 us cover, 38%. Without them the cover
is 39 us and no ring fits at all.

## Why the projections are sliced, and why that TIGHTENS the first ring

`in_proj_qkvz` is fused and lays out `[q | k | v | z]` at widths `[2048 | 2048 | 6144 | 6144]`.
The early path needs q and k -- 25% of the output. The current path needs k, v and z -- 88%, all
but q.

    fused, once                                   112.1 us   MEASURED
    sliced q|k        (25% of the output)          16.2 us   MEASURED
    sliced k|v|z      (88% of the output)          92.3 us   MEASURED

    two fused projections                         224.3 us
    the two slices                                108.5 us
    saved                                         115.8 us   per layer

Time tracks output width almost exactly (25% wide -> 14% of the time), so this shape is bound by
writing the output rather than by launch overhead, and "the fused one is faster anyway" -- which is
true for many shapes -- is false for this one.

The convolution slices the same way. It runs over `[q | k | v]`, 10240 channels; the early pass
needs q and k (40%), the current pass needs k and v (80%). `z` never enters it.

    two full convolutions                         534 us     DERIVED from 267 us measured whole
    early 40% + current 80%                       320 us     DERIVED
    saved                                         214 us     DERIVED

**Slicing shrinks the cover.** The unsliced early projection is 112 us and the cover would be
158.8 us, which swallows a 61 us ring on its own. Sliced, the cover is 62.8 us. The two
optimisations are not independent: taking the projection saving REQUIRES bundling the ring.

## Splitting the ring: right about the windows, wrong for this link

The ring's `[q | k]` channels are wanted at the read-only convolution and its `v` channels only at
the writing one, and those two moments are far apart:

    q|k half, 32 KiB      window 62.8 us     W_o + norms + the early projection
    v half,   48 KiB      window ~567 us     the above, plus the read-only convolution, the head's
                                             feed-forward and the current projection

Nine times the window for the half that is 60% of the bytes, so the halves should not travel
together. What decides it is the shape of a transfer on this link, MEASURED as a round trip on the
persistent connection the wire actually uses:

    empty frame     139 us
    12 KiB          248 us
    32 KiB          304 us
    48 KiB          355 us
    80 KiB          436 us
    240 KiB         820 us
    ---------------------------------------------------------------
    fixed 139 us a frame, marginal 2.48 us/KiB

A transfer is NOT fixed-cost dominated at these sizes: one ring carries 198 us of byte cost
against a 139 us frame cost, and three carry 595 us. Bytes are the larger half, so moving bytes
off the critical path is worth an extra frame.

    all three rings on the span frame              595 us of marginal bytes, all before the span
                                                   can begin
    q|k thirds on the span frame                   238 us, and the span begins 357 us earlier
    v thirds on a second frame                     139 + 357 = 496 us, against a ~567 us window

The v frame fits its window with 71 us to spare and takes 357 us off the span frame. That is the
whole of the argument, and it is 2.6% of a 13.6 ms span -- real, and smaller than it looks.

    An earlier version of this section concluded the opposite from a table that read 12 KiB at
    0.052 ms and 192 KiB at 0.077 ms, and called transfers fixed-cost dominated on that basis.
    Those numbers were taken on a FRESH connection per sample and timed `sendall`'s return. That
    measures a copy into the socket buffer and a TCP window still in slow start -- not arrival,
    and not this wire, which sends every frame down one long-lived connection. Re-measured as a
    round trip on a persistent connection, the fixed cost is 139 us rather than 50 and the
    marginal cost is 2.48 us/KiB rather than 0.17. Both halves of the old conclusion were
    artefacts of the harness.

    The ring's size was also wrong there -- 64 KiB, from head dimensions this model does not have.
    It is 80 KiB: q and k at 2048 channels each and v at 6144, four taps, bfloat16.

## What the overlap buys

The head's feed-forward runs while the host contracts:

    q~ round trip                                 1.35 ms    MEASURED (after the inbound-thread fix)
    head's feed-forward                           305 us     MEASURED

So the feed-forward hides 305 us of a 1.35 ms round trip -- 23% of it. The remaining 1.05 ms is
still exposed, and that is the honest statement of what one window is worth here. Three windows a
span, one per linear layer, and the span's wall time is 13.6 ms MEASURED.

The wire is not the whole of that 1.35 ms: bare TCP is 0.33 ms and the host's own work is 0.59 ms,
so about 0.43 ms is framing and dispatch on both sides. That is the next thing to attack and it is
not addressed here.

## The window on the way back is large

The host advances S and the ring from the deferred k and v. That has to finish before the same
layer is called again, which is one decode step later -- 244 ms MEASURED. The update itself is
tens of microseconds. There is no pressure on this direction at all, which is why the ring can be
sent back the cheap way (with the deferred update) rather than the expensive way (in the reply).

## What must not be got wrong

    the early key must NOT write the ring     it is a state reused across K steps, and only
                                              per-step values may be early. `_convolve(write=False)`
    the current key MUST write it             and it is the current one that advances the state
    v is never early                          it feeds `beta (k.q) v`, which is this step's own
                                              contribution, and it is projected after the
                                              feed-forward completes `x_l`
    the state's own k is never early          `S <- alpha S + beta (v - alpha S k) k^T`. The
                                              pre-registration measured +280% for an early one
                                              there against +0.013% for an early one in `q~`

The rule behind all four is one line, and it is the pre-registration's: **a key that feeds a state
REUSED ACROSS TIME must be the current one; every other use may be early.**

## Status

Designed and measured, not implemented. `_issue_read` and `_early_view` exist on the branch this
one was ported from and are being changed there; the ring-as-parameter, the sliced projections and
the sliced convolutions do not exist anywhere yet.

## What it measured, once both ends had the mechanism

Two machines, pool on a PRO 6000 and host on a 5090 across a LAN, `--afd-span-cut`, stateless
pool, five prompts of 32 tokens each, identical output text:

    --afd-query-shift-layers 0      48.981 s      3.27 tok/s      span 16.18 ms
    --afd-query-shift-layers 1      49.024 s      3.26 tok/s      span 16.36 ms

Query-shift is 0.09% SLOWER, which is noise. The early read fires, the window opens, and nothing
comes through it. The reason is in the callback:

    one callback                    2.731 ms
      the host's own compute        1.706 ms      MEASURED on the host, state_mix served
      the host's reply              0.318 ms
      wire and framing, both ends   0.707 ms
      ---------------------------------------------
      the host's own work is 74% of a callback

    what query-shift has to cover it with:
      the head's feed-forward       0.248 ms      9% of one callback

Three callbacks a span, 8.19 ms of a 16.36 ms span. Even if all three were covered perfectly by a
feed-forward, the ceiling is 4.5% -- and the cover is 9% of what it would have to be.

    An earlier run compared 48.9 s against 48.9 s and recorded "+0.015%, no gain". That
    measurement was invalid: the build under test had no `_issue_read`, so shift 1 differed from
    shift 0 only in which hidden state was projected, with no parallelism mechanism to exercise.
    This run has the mechanism on both ends and reaches the same verdict for a reason that can
    be read off the numbers.

### Why this deployment defeats it, and what would not

Query-shift buys a window on the POOL and pays for it in accuracy. The window is filled with the
pool's feed-forward. On this arrangement the pool's feed-forward is 0.248 ms and the thing it must
hide is the host's 2.03 ms of compute -- an 8x gap, and the shift depth that would close it is
around ten layers, far past where accuracy is already gone (all-shifted measured +280%).

That gap is not an accident of tuning; it is the architecture working as intended. The host is
deliberately the weaker card -- that is what makes a small card viable -- and the pool is
deliberately the one holding the weights. So the pool is fast at exactly the work query-shift uses
as cover, and the host is slow at exactly the work query-shift is trying to hide.

The lever is therefore the host's 1.706 ms, not the shift depth. That number is the stateless
pool's price: it is convolution, contraction and advance moved to the host, against 0.594 ms for
the plain state read the stateful arrangement used. The statelessness is worth keeping -- it is
what lets a request stop being sticky to one pool -- but it is what query-shift now has to
overcome, and 0.248 ms of feed-forward will not.

## Where a callback's time actually goes, and the 0.415 ms that left it

MEASURED inside `_mix` on the host, five stages, none of which synchronises -- so every number is
CPU time, launch and Python, not the card:

    cast        0.169 ms
    slots       0.009
    convolve    0.297
    core        0.826      the state read and the mix
    update      0.415      the recurrent advance
    ------------------------------
                1.716      against 1.764 the pool sees as this side's compute

The advance was the one stage nothing in the reply depends on: the read above it takes S_(t-1)
deliberately, so the pool was blocked on a write it never reads. Parked and applied after the
reply goes out, on the same single inbound thread that will serve the next frame:

    total, five prompts x 32 tokens     49.534 s  ->  45.701 s      7.7% faster
    tokens per second                       3.23  ->      3.50
    one callback                        2.731 ms  ->  2.222 ms
    the host's own compute              1.764 ms  ->  1.282 ms
    the update stage                    0.415 ms  ->  0.001 ms

Output is bit-identical across the change -- 4 of 4 prompts, 48 tokens each, against the same
build with the advance inline. It is a reordering of two independent operations and not an
approximation, and it is measured as one rather than argued as one.

The remaining 1.28 ms is still all CPU. `core` at 0.826 ms is twelve calls, which is about 69 us
each -- far above a kernel launch -- so there is something in there that is not launch cost, and
it has not been found yet. That is the next thing to measure, and it is a larger number than
anything query-shift can win.

### The early read is not what produced this

Worth stating plainly because three separate measurements were reported before anyone checked:
`SpanRunner._issue_read` returns None in every configuration this branch supports, because
`mix_host` is installed unconditionally and `issue_host` is installed nowhere at all. The early
read has never executed. Every "shift 1 vs shift 0" number taken so far compares two builds that
differ only in which hidden state gets projected, with the parallelism mechanism inert on both
sides, and none of them says anything about query-shift.

## The arrangement that was built, and what it measured

The early frame carries the PRE-CONVOLUTION `[q | k]` and the write strength, and nothing comes
back. The host convolves them against its own ring without writing to it, forms the query
coefficient, contracts the state and keeps the result; the MIX that follows for the same request
and layer takes what was kept instead of contracting again. Ordering is what makes it safe and
the transport already gives it -- both frames go down one socket in order, and the far end serves
them on one thread.

The materials rather than the coefficient, because forming it needs the convolved q and k, which
needs the ring. See the section above for why the ring does not travel.

Two machines, pool on a PRO 6000 and host on a 5090, five prompts of 32 tokens. REPLICATED and
alternated, 1-0-1-0, because the within-arm spread turned out to be larger than every difference
this project had reported from single runs:

    --afd-query-shift-layers 0      43.915  46.530      mean 45.222 s
    --afd-query-shift-layers 1      53.022  50.239      mean 51.630 s
    -----------------------------------------------------------------
    shift 1 is 14.2% SLOWER; the arms differ by 6.4 s against a 2.7 s spread

    Single runs said "net zero" (49.006 against 49.029) and later "2.6% faster" (47.725 against
    49.029). Both were inside that spread and neither was evidence of anything. They were
    reported here as findings and they were not.

What each end did with it:

    the host                        shift 0     shift 1
      core, the read and the mix     0.828 ms    0.587 ms
      the whole mix served           1.326       0.990
      one callback, as the pool sees it
                                     2.288       2.027
      share of a span spent waiting on the host
                                       49%         38%

    the pool
      head mlp, which now carries the early send
                                     0.236       0.986

The host improves by exactly what the design predicted: the state read leaves the callback and
happens while the pool spends a feed-forward. The pool pays more than that for the send.

### Why the pool pays more, MEASURED

    the early send                  0.718 ms total
      the projection                0.423
        norm                        0.058
        q|k, sliced                 0.075
        b and a                     0.080
        gates                       0.154      <- the decay, which this path discards
      the frame                     0.294      a blocking copy to CPU, which drains the queue

against 0.261 ms it takes off a callback. The largest single step computed something nobody
wanted: `gates` returns the decay and the write strength, and the early view uses beta alone.
That is fixed -- `write_strength`, and `in_proj_ba` sliced to b -- and the remainder is the
extra projection itself, which is the acknowledged price of the shift.

The honest shape of it: on this deployment the extra projection costs MORE than the overlap
saves, and by enough to see.

    per layer, the early send      0.280 projection + 0.175 send = 0.455 ms
    per layer, the callback saves  2.385 - 2.221               = 0.164 ms
    sends a span, MEASURED         2.67
    ------------------------------------------------------------------
    predicted from those          +0.78 ms a span
    MEASURED                      +2.18 ms a span  (13.80 -> 15.97)

The prediction accounts for about a third of it, and the rest is a cost these counters cannot
see. Nothing in `_send_early` synchronises, so every number above is LAUNCH time -- and the extra
projection is real GPU work on the pool, competing with the feed-forward it was meant to hide
behind. The overlap is with the host; the projection is paid on the pool's own card, which is the
resource the feed-forward wants.

That is structural, not a tuning failure. The condition that would flip it is a host slower
relative to the pool: the saving is on the host and grows as the host weakens, while the
projection stays where it is. That is the direction this architecture is meant to go -- the host
is deliberately the cheaper card -- so the arrangement should be re-measured on a host further
from parity before it is judged for good.

## What the shifted read point costs the model

Latency is half the question. The other half is what the model computes, and that is measured on
ONE machine, teacher-forced a position at a time through the recurrence, with only
`recurrent_gated_delta_rule` replaced -- `benchmark/afd/early_k_bpb.py` over
`afd_query_shift/key_shift_arms.py`. Twelve fineweb-edu documents, 48168 bytes, 9927 tokens:

    exact          0.6272 bits per byte     the model
    mixed          0.6271                   -0.016%   the arrangement
    all_shifted    2.1229                  +238.5%    the control

The pre-registered decision rule holds, and holds by three orders of magnitude. The shifted key
may enter the OUTPUT's copy of `P(k)` -- an error spent in one step -- and may not enter the
STATE's, where it compounds through every step after. The control arm is what that looks like:
not a degraded model, a different one.

    Bits per byte rather than perplexity: it divides by UTF-8 bytes, so the three arms are
    comparable without the tokenizer entering.

    The kill condition was +2.68%, fixed before the measurement. Mixed came in below zero, which
    is noise and not an improvement -- the honest reading is "indistinguishable", and it is the
    same conclusion either way.

This is measured on a checkpoint that declares no read point, which is exactly the case
`--afd-query-shift-layers` exists for and warns about: serving a query projection an input it was
not trained on. A checkpoint repaired for the shift would be served at its own read point with no
setting given and no warning, and would be expected to do better than -0.016%, not worse.

## Making the early send cheap enough to be worth sending

The first version cost 0.718 ms against the 0.261 ms it saves. Two changes and one mistake:

    first                 project 0.423   send 0.294   total 0.718
    projection trimmed            0.265        0.273         0.538
    send queued                   0.244        0.151         0.395

**The projection** was computing what it discards. Timed in four steps -- norm 0.058, q|k 0.075,
b and a 0.080, gates 0.154 -- the largest was `gates`, which returns the decay and the write
strength, and this path uses beta alone. The decay is five elementwise kernels against beta's one
sigmoid, and `a` was projected only to feed it. So `write_strength`, and `in_proj_ba` sliced to b
the way `in_proj_qkvz` is already sliced to q and k.

**The send** was a blocking copy. `protocol._payload_of` does a plain `.to("cpu")`, which
synchronises the pool's stream -- so an inline send drains the queue in front of the feed-forward
this frame exists to run beside. The pool has had a sender thread since the deferred update was
written, for exactly this reason; the early frame now takes it, staged non-blocking behind a CUDA
event.

**The mistake** is worth keeping. Ordering the mix after the queued frame was done with
`_outbox.join()`, and that queue also carries the deferred updates -- which are queued there
precisely so that nobody waits for them. Joining it put them back on the critical path: the span
went 15.16 -> 16.41 ms, and the change meant to make the arrangement cheaper made it dearer.
`post_unawaited` returns an event now and the mix waits on its own layer's frames.

    An overtake is not an error at either end. The far end finds no contraction waiting and does
    its own, correctly, for the arrangement this one replaces -- so the failure is the overlap
    silently absent, which is the same shape as every other failure this line has had. The test
    requires the narrow wait AND rejects the join, so it cannot widen back.

## How these numbers are taken, after several were taken wrong

Every latency comparison in this file is now REPLICATED and ALTERNATED: at least two runs an arm,
in 1-0-1-0 order, means reported with the within-arm spread beside them, and no difference claimed
that is smaller than that spread.

That is not caution for its own sake. On these two machines the within-arm spread is about 2.7 s
on a 45 s benchmark -- 6% -- and three separate conclusions were drawn here from single runs whose
differences were 0.05%, 2.6% and 0.09%. All three were noise reported as findings, and one of them
was reported as the arrangement finally paying off.

`/tmp/measure.sh` and `/tmp/summarise.py` in the working session did the last round; the
summariser prints NOT SEPARABLE rather than a percentage when the difference is inside the spread.

The per-stage numbers are a different matter and remain trustworthy: they are timed inside one
run, over thousands of calls, so ambient drift moves both arms of a comparison together. What
they cannot see is GPU time, because nothing on that path synchronises -- which is exactly the
gap between the +0.78 ms those counters predict and the +2.18 ms the span actually costs.

## What moving the work off the critical path actually did

Everything the early send does has until this layer's MIX to finish, which is a feed-forward
away, so none of it belongs in front of the feed-forward -- and all of it was there. Two moves
were tried together and they do not behave the same way.

    the build measured              shift 1     shift 0     difference
    ------------------------------------------------------------------
    before either move              51.630 s    45.222 s    +14.2%
    worker thread AND side stream   56.488      45.450      +24.3%

Two runs an arm, alternated. The second round's spread was 0.29 s and 0.84 s, so the difference
is not in doubt.

The side stream is REVERTED. It was there for the obvious reason -- on the main stream the early
projection's kernels are not beside the feed-forward's, they are queued in front of them, and a
stream is first-in-first-out whichever thread filled it -- and it made the arm ten points worse.
Why is not established. The candidates are that two streams contending for a small-batch GPU cost
more in scheduling than they win in overlap, and that the side stream's event wait serialises
where the main stream's implicit ordering did not. Neither is measured, and the honest statement
is that the change was tried, measured, and taken out.

The worker thread stays. What it moves is real -- 0.395 ms of Python and launches off the thread
that is about to spend the feed-forward -- and it is the half of the change that costs nothing to
keep.

## The norm the early projection no longer runs

Its input is the previous layer's post-attention norm OUTPUT, which the model produced for its own
feed-forward from the same residual this used to normalise itself. So the caller hands over the
normalised tensor and one RMSNorm a layer goes: 0.058 ms, and a tensor that was already there.

It is a second approximation on top of the shift, not a free saving. The two norms are the same
arithmetic over the same vector and differ in the learned per-channel scale, so the query is now
projected from a vector carrying a neighbour's. `key_shift_arms.mixed_reused_norm` is the arm that
measures what that costs, and a test requires it to exist for as long as the code takes the
saving.

## Where the loss actually is: the host serves two ops on one thread

The pool's extra projection was the suspect for a long time and it is the smaller half. What the
HOST does per layer:

    shift 0     state_mix                        1.384 ms
    shift 1     state_early  1.234
                state_mix    0.990               2.224 ms

Its inbound worker is one thread -- deliberately, because a second reader of the same socket is
not a race that sometimes loses but a deadlock that always happens -- so the two serialise.

The sequence inside a span makes that expensive:

    mix(N) returns  ->  send early(N+1)  ->  feed-forward  ->  mix(N+1) is sent

The early read gets exactly the feed-forward as its window, and the feed-forward and the
projection and the send together are about 0.54 ms. The contraction takes 1.234. So the mix
arrives at a host that is still working on the early read and waits out the remainder -- roughly
0.3 ms a layer, on top of the 0.37 the pool pays for the projection and the frame. At 2.67 sends
a span that is the larger part of the 2.18 ms a span the arrangement costs.

This is what makes the arrangement's own condition sharp. Query shift wants a host that is SLOW
in the way that makes its callback expensive, and it needs that same host to be FAST enough to
finish an early contraction inside one feed-forward. Those pull in opposite directions, and the
gap between them is what any version of this has to fit into.

The stages of that 1.234 ms are timed in `early_contraction.py`, so the next change is aimed.

## One query, and where s comes from

The operator has ONE query:

    S_t = alpha S_(t-1) P(k) + beta v k^T          P(k) = I - beta k k^T
    o_t = S_t q = alpha S_(t-1) P(k) q + beta (k.q) v
                = alpha (S q~) + s v

Two facts fall out of writing it that way, and the implementation had neither of them right.

**The k in `s` is the state's k.** `beta (k.q) v` and `beta v k^T` carry the same key -- they are
the same term of the same product -- so by the pre-registered rule that key is the CURRENT one. It
is not the early key that may enter the output's copy of `P(k)`.

**There is one q, and under query shift it is the early one.** The current projection has no use
for a query at all. The implementation projected one anyway and computed `s` from it, which is
neither of the arms and is not the operator either.

So the arrangement that is both self-consistent and sendable early:

    early     project q_e and k_e            q~ = q_e - beta_e (k_e . q_e) k_e
                                             contract S q~, keep the reading AND q_e
    current   project k and v, NOT q         s = beta (k . q_e)
                                             core = alpha (S q~) + s v
    state     k and v, current                 unchanged, and the rule is why

The early key has to be in `q~` -- without it the coefficient cannot be formed before the
feed-forward, and there is nothing to send early. That is the whole reason the key's read point
moves at all, and it is what the pre-registration was about.

What it saves, beyond being right:

    the current projection      drops q          16384 -> 14336 outputs
    the MIX payload             drops q          10240 -> 8192 channels, -20%
    the host's convolution      drops q's        -20%
    the host's mix              no normalise, no expand, no query_coefficient on current values

and the early projection stops being an addition: the early q REPLACES the current q rather than
joining it. What is genuinely extra is the early k alone.

## Where the host's 1.234 ms goes, and what reading it wrongly cost

    cast        0.053      convolve    0.413      read     0.215
    slots       0.012      gates+norm  0.521      stash    0.019

and the largest, split again:

    beta 0.073      normalise 0.247      expand 0.065      coefficient 0.085

The expansion was the suspect -- `repeat_interleave` is a copy and it runs twice, sixteen heads to
forty-eight -- and it is 0.065 ms. That is the fifth time on this path that reading the code named
the wrong thing and timing it named the right one. The entry that matters is `normalise` at 0.247,
which is two casts, two normalisations and a scale: eight small kernels, and all of it launch time
on a host whose Python is the slow part.

`convolve` at 0.413 held something cheaper to fix. It built its index by copying a Python list to
the device, once per layer per step, for a tensor that does not change while a batch holds its
slots -- 48 copies a token. It is built once now and kept, keyed on the slot NUMBERS so that a
slot handed from one request to the next gets the same tensor and it is the right one.

None of this closes the gap on its own. The contraction has 0.54 ms of window and takes 1.234, and
the pieces that would have to go are the convolution and the normalisation -- which are the
arithmetic, not overhead. What that leaves is the other direction: the host serves the early read
and the mix on ONE thread, deliberately, and they are different layers touching different slices.

## Why nothing in this direction can win, which is the arithmetic and not the hardware

The early read exists so the host contracts the state while the pool spends a feed-forward. What
it costs the host is not the contraction -- it is that the contraction needs a query coefficient,
and building one needs a convolution and a normalisation and a coefficient, on a projection that
is not the one this step is going to use. So the host makes that whole pass TWICE: once over the
early `[q | k]`, once over this step's `[k | v]`.

MEASURED, per layer:

    what the host does once at shift 0     convolve + normalise + coefficient + read + mix
                                           1.384 ms
    what it does at shift 1                the same pass twice, minus the second's coefficient
                                           1.207 (early) + 1.005 (mix) = 2.212 ms

    what moving the read out of the mix SAVES        1.384 - 1.005 = 0.379 ms
    what building the coefficient a second time COSTS
                                           convolve 0.327 + normalise 0.247 + coefficient 0.088
                                           = 0.662 ms

The trade is 0.662 to save 0.379, and it is set by the shape of the arithmetic rather than by the
machine. The two convolutions cost nearly the same -- 0.327 for the early pass over 4096 channels
against about 0.29 for the mix's over 10240 -- because both are launch-bound, so making the early
pass narrower does not make it cheap. That is also why the payload trim helps and does not decide:
it takes a fifth off the mix's side, not off the duplicate pass.

The window is the second problem and the smaller one. The pool's feed-forward, its projection and
its send give the early read about 0.54 ms, and the contraction takes 1.207, so the mix arrives at
a host still working and waits out the remainder -- 0.667 ms a layer. Even with a contraction that
fit the window exactly, the ledger is:

    host saves 0.379      pool pays 0.395      net +0.016 ms a layer

so a perfect early read is break-even, and the duplicate pass is what makes it a loss.

What would change it is not a faster host or a slower one -- both scale the saving and the
duplicate together. It is a host on which a pass costs less than its launches: a fused kernel, or
a captured graph. Every number above is launch time, on a path where nothing synchronises, and
the same arrangement on a host that spent its time in the arithmetic rather than in front of it
would read differently.

## One linear layer at shift 1, as a schedule

Measured, per layer, against shift 0's 2.759 ms:

    pool   |-- own work 0.969 --|------------- blocked on the callback 2.541 -------------|
                    ^ sends the early frame to a worker, then the feed-forward, then this
                      layer's projection and gates, then sends the mix

    host          |wire|-- early contraction 1.207 --|-- mix 1.005 --|reply|
                              |-- the mix waits in the queue 0.622 --|

    per layer      3.510 ms      (shift 0: 2.759)

The extra 0.751 splits two ways and BOTH were things this session had already called solved:

    the pool's own work      0.461 -> 0.969    +0.508
    the callback             2.298 -> 2.541    +0.243

The pool's own work was supposed to be unchanged: the early projection went to a worker thread so
that it would not sit in front of the feed-forward. It does not sit in front of it and the main
thread is 0.508 ms a layer slower anyway, because the worker contends for the same GPU and the
same interpreter. Moving work to a thread moves its place in a queue; it does not move its cost
off a machine. That is also the likeliest reason a side CUDA stream measured ten points WORSE
rather than better -- a second stream makes the contention more complicated, not less.

The callback was supposed to shrink: the state read moved out of the mix and the mix is indeed
cheaper, 1.384 -> 1.005. It grew anyway, because the read it moved out now runs in front of it on
the host's one thread and only about 0.46 ms of that hides behind the pool's own work.

Three symptoms, one cause. Moving a 0.217 ms state read earlier requires 0.99 ms of preparation --
convolve, normalise, coefficient -- and the preparation must happen where the ring is. On the host
it blocks the mix; on the pool it contends with the feed-forward. It is 4.6x the thing it makes
early, and that ratio is what any version of this has to change. Where to put it is not the
question.
