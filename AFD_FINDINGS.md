# AFD with an Early-Q read point: what was measured

Everything here was run on Qwen3.8-27B-FP8. Host is an RTX PRO 6000 Blackwell (97 GB); the pool,
where a second machine is involved, is an RTX 5090 (32 GB) across a 10 GbE link with a 0.3 ms TCP
round trip and 4.4 Gbit/s of bulk bandwidth. Numbers with no arrangement named are colocated.

Findings are grouped by what they decide. Several of them refuted the hypothesis that motivated
them, and those are marked, because a refuted hypothesis that stays in the record is the only
protection against re-adopting it.

## 2026-08-22, the linear attention is clean on both paths, and the 1-5% was a misread of my own table

    layer 0   whole-layer relative 0.00320    layer 1   0.00346    layer 2   0.00457
    layer 4                        0.00447    layer 5   0.00475

Cosine 1.0000 at every layer. That is bfloat16, and it is the same 0.4% the two prefill algorithms
cost each other in isolation.

So the "1 to 5 percent" that has driven the last several rounds was never the layer's error. It
came from a PER-HEAD, PER-ROW breakdown, where a head whose own norm is small shows a large
relative difference while contributing almost nothing to the layer. I read a diagnostic
decomposition as if it were the quantity itself, chased the heads it happened to rank first, built
a story about key head 10 out of it, and retracted that story only when the ranking moved with the
row. The whole-tensor figure was in the same log line the entire time.

`_linear_attention` is now exonerated on both paths -- decode and prefill, arithmetic and
composition -- against the model's own forward, in the same process, on the same weights, seeded
from the same state. What it does NOT exonerate is the deployed data flow: this comparison installs
a LOCAL `ask_host` stub, so the wire, the deferred update's ordering, and the host's own state
never enter it.

The ring-seeding control that prompted this round did not discriminate -- reversing the history
gives an identical number to five figures, which means the ring's content does not reach the
result here at all, most likely because a fresh request's conv state is zero. It is recorded as a
control that could not fail rather than as evidence the seeding is right.


## 2026-08-22, the two prefill algorithms agree at deployment size, so the 1-5% is not theirs

    8 tokens, zero initial state          relative 0.00416
    122 tokens, onto a warmed state       relative 0.00416

The same to three figures. The model's prefill runs `chunk_gated_delta_rule` and the pool walks
the chunk token by token, and the difference between those two algorithms does not grow with the
chunk or with the state -- so the 1 to 5 percent seen in the deployed arrangement is NOT what two
implementations of one recurrence cost. It is an order of magnitude above that, and it belongs to
something else.

The comparison had to be built carefully in two places, and both would have produced a number
rather than an error:

    the state layout      the chunked kernel keeps (heads, key dim, value dim) and `read_one`
                          keeps (heads, value dim, key dim) -- transposed. Handing one tensor to
                          both would have measured the transpose. Each side builds its own initial
                          state from the SAME warm-up tokens instead, which is also what the
                          deployment does
    the state handoff     the kernel returns three things, the second is None, and the final state
                          is not usefully among them: it advances `initial_state` IN PLACE, which
                          is how the backend chains chunks. Feeding `got[1]` forward made triton
                          fail to compile on a null pointer

Kept as a case at the deployed size, because "they agree" was previously known only for eight
tokens from nothing.


## 2026-08-22, three rows instead of one, and the key-head reading is retracted

    layer 0   row 0   42/48 within 2%, worst h30 h31 h1     mid  44/48, worst h37 h36 h38
              last    48/48, worst 0.014
    layer 1   row 0   45/48, worst h30 h32 h31              mid  48/48
              last    46/48, worst h3 h5 h17
    layer 2   row 0   48/48                                 mid  46/48, worst h18 h20 h32
              last    48/48

Two things fall out, and both cut against what was written here yesterday.

The error does NOT grow along the chunk. Layer 0 goes 42/48, 44/48, 48/48 -- it improves. So the
scan is not accumulating anything, which is the second independent way accumulation has now been
ruled out.

And the worst key head CHANGES WITH THE ROW: key head 10 at row 0, key head 12 at the middle, no
grouping at all at the end. "Key head 10 is mis-sliced" is dead. The contiguous triples are not
evidence of a slicing bug -- three value heads share one key head on this model, so they share q
and k, so their errors are correlated by construction. That is a property of the model's shape
and it would appear for ANY error in the (k.q) term, rounding included. The previous entry read
it as structure. It was not.

What the three rows leave is a scatter of 1 to 5 percent that does not grow, does not favour a
head, and sits above the 0.4% the decode path gives. The model's prefill runs
`chunk_gated_delta_rule` while this side walks the chunk token by token; those agree to 0.0042 on
eight tokens from a zero state, and 122 tokens from a live state is not that test.


## 2026-08-22, the stub was fixed, nothing moved, and the reason is that the comparison sees one row

The comparison's local `ask_host` did a BATCHED read over every row of a chunk -- right for a
decode batch and wrong for a prefill, whose rows are one request's consecutive tokens. That is the
first bug of this whole search, reappearing inside the instrument built to hunt it, and it looked
like the explanation for "42 of 48 heads on a prefill against 48 of 48 on a decode".

It was not. With the stub walking the chunk token by token -- proven, not assumed, by a marker
that logs "scanning 122 rows sequentially" -- every figure is bit-identical: 42/48, h30 = 0.057,
h31 = 0.053, h32 = 0.051.

The reason is in `per_head`: it computes the error over the chunk and then takes `err[0]`. ROW
ZERO. Row 0 reads the initial state and nothing else, so batched and sequential give it the same
answer, and every prefill reading this comparison has produced describes the FIRST TOKEN of the
chunk. The per-key-head structure at row 0 is real -- it reproduces across layers and across runs
-- and it has nothing to do with the scan, because at row 0 there is no scan yet.

So two things are now known that were not: the batched-stub hypothesis is dead, disproved rather
than abandoned, and the prefill comparison has never looked at a row where the recurrence has run.
Comparing a late row is the next measurement, and it is one line.


## 2026-08-22, the error enters between the convolution's output and the layer's, on the wire

Three boundaries measured on the deployed 122-row prefill, each against the model's own value:

    the packed projection, BEFORE the convolution   16/16 key heads, error 0.000 -- identical
    the convolution's output                        16/16 key heads, worst 0.004 -- bfloat16
    the layer's output                              42/48 value heads, worst 0.057 on one
                                                    key head's group

So the projection, the packing and the convolution are exact, and everything between the
convolution's output and the layer's output is where the per-key-head error is born. That段 is the
scan and the mix -- and in the deployment the scan crosses the WIRE: the query coefficient is
computed on the pool, sent to the host, walked token by token there through `read_one` and
`update_only`, and the readings come back.

`test_afd_prefill_scan.py` verifies the scan's ALGORITHM -- synthetic tensors, eight tokens, zero
initial state, one process. The wire path has never been verified, and it is where the ordering of
the two uses of `k` lives: the query coefficient's, computed locally and at once, and the state
update's, deferred and sent separately.

Getting the first two boundaries took catching the seventh instance of comparing two correct
measurements of different quantities. The model's `mixed_qkv`, the tensor that reaches
`self.attn(forward_batch, mixed_qkv=...)`, is PRE-convolution -- the convolution runs inside the
backend -- and the span's `mixed` is post-convolution. Compared directly they read 100% apart on
every key head while the layer's output agreed to 2%, which is impossible, and the impossibility
is what exposed it. The reference for the convolution's output is built instead by running
sglang's own `causal_conv1d_fn` on the packing both sides agree on.


## 2026-08-22, the worst heads are one key head's group, and it is not accumulation

A prefill scan walks a chunk token by token, so a head whose decay is closest to 1 carries its
rounding furthest -- "the worst heads are the slowest-decaying heads" would be a completely
different finding from "one key head's channels are wrong". The per-head decay separates them:

    layer 0   worst h30 h31 h32   their alpha 0.173, 0.129, 0.008   rank corr(err, alpha) -0.269
    layer 1   worst h30 h31 h32               0.715, 0.775, 0.979                        +0.126
    layer 2   worst h24 h39 h11               0.750, 0.606, 0.749                        +0.276
    layer 4   worst h17 h16 h15               0.748, 0.935, 0.647                        +0.253

The correlation is weak and changes sign, and layer 0's three worst heads are among the FASTEST
decaying of the 48 -- alpha 0.008 accumulates nothing at all. Accumulation is out.

What is left is a structure, and it is the same structure three times:

    layer 0   h30, h31, h32   ->  30//3 = 31//3 = 32//3 = 10   key head 10
    layer 1   h30, h31, h32   ->                               key head 10
    layer 4   h15, h16, h17   ->  15//3 = 16//3 = 17//3 =  5   key head  5   (error 0.16)

48 value heads over 16 key heads, three value heads to a key head, and the worst three are a
CONTIGUOUS TRIPLE every time -- exactly one key head's group. Which key head varies by layer. So
whatever is wrong is per-key-head: the q and k channel slicing out of the convolved packed
projection, or the key-head-to-value-head expansion. And it is prefill-only: the same comparison
on a decode row gives 48/48 heads within 2% at every layer.


## 2026-08-22, the prefill path compared at last, and one key head's group is the worst twice

The comparison's gate was single-row, so everything it had ever said covered the DECODE path. The
gate is lifted and the deployment's own case -- a 122-row prefill through `prefill_convolve` and
OP_STATE_SCAN -- is inside it now:

                        decode                 prefill
    layer 0   48/48 within 2%          42/48, worst h30 .057 h31 .053 h32 .051
    layer 1   48/48 within 2%          45/48, worst h30 .034 h32 .034 h31 .033
    layer 2   48/48 within 2%          48/48, worst h24 .010

Heads 30, 31 and 32 are the three worst at TWO different layers. This model has 48 value heads
over 16 key heads, three value heads to a key head, and 30 // 3 = 31 // 3 = 32 // 3 = 10 -- they
are exactly the three value heads of KEY HEAD 10. One key head's group being the worst at two
independent layers is a structure, not scatter.

Read the per-head norm ratio and not the elementwise median on this table. The elementwise figure
is |d| / (|b| + 1e-6) and its median sits at 0.47 to 0.86 while every head's norm agrees to a few
percent -- which happens when the differences live in elements whose reference is near zero, and
says nothing about the ones that carry the signal.


## 2026-08-22, the pool's linear attention is arithmetically correct, and the axis that said otherwise

Fed the model's own input and seeded from the model's own state, `_linear_attention` agrees with
`Qwen3_5GatedDeltaNet.forward` head by head:

    layer 0   48/48 heads within 2%, worst h40 = 0.007
    layer 1   48/48 heads within 2%, worst h27 = 0.016
    layer 2   48/48 heads within 2%, worst h2  = 0.019

Including layer 1, the layer whose deployed output has been 6.3% high throughout. So the
reimplementation is right and what feeds it in the arrangement is not.

The first run of this comparison said 0/48 heads within 2% and an elementwise median of 1.06 --
100% error on the input of a LINEAR projection whose output agreed to 0.5%. A linear map cannot do
that, and the contradiction is what exposed the instrument rather than the code: the hook capturing
the model's pre-projection value was registered permanently, and `compare` calls `span_of` twice --
the second time on a shuffled input, for its control. The model's value was overwritten by the
control's. Fixed by capturing once and removing the hook.

That is the SIXTH time in this search that two correct measurements of different things were set
side by side, and the second caught by an internal contradiction rather than by luck. The full
list, kept because the pattern is the finding: a trace walking rows against a reference walking
layers; a control reversing a filter on both sides at once; x+attn+mlp against x+attn; a whole row
against a single head; a 1-row occasion against a 122-row one; and this.

The two uses of `k` in the linear attention -- the query coefficient's `I - beta k k^T` and the
state update's `k k^T` -- currently draw from ONE tensor, the current attention input after the
convolution. Whether the intended arm reads them from two different points, the feed-forward's
input vector and the attention's input vector, is a design question and not a defect: the standing
constraint has been that only the query moves.


## 2026-08-22, the per-layer cut is token-identical to colocated, and the group cut is not

The bisect that should have been run first. Same two processes, same weights, same prompts, the
only change being `--afd-span-cut` dropped and the shift at 0:

    prompt                          colocated                        per-layer AFD
    "The capital of France is"      " Paris.\nThe capital of         IDENTICAL
                                     Germany is Berlin.\nThe"
    "The"                           " following table lists the      IDENTICAL
                                     average annual salaries..."
    "Explain why the sky is blue"   "\n\nThe sky appears blue        IDENTICAL
                                     because shorter wavelengths..."

Token for token, on three prompts. The conventional arrangement -- feed-forward on the pool,
attention on the host -- is exact.

So everything both cuts share is clear, and it is most of the machinery: the protocol, the
transport, the socket and its reply routing, the pool server and its departure batching, the
host's attention and KV cache, the per-layer feed-forward offload, the host's model wiring. Every
hang and every misrouted frame fixed over the past days was real and none of them is what is
wrong now.

What is left is what only the GROUP cut has: `SpanRunner`'s prologue, run and epilogue, the
residual and gate tables on the pool, the convolution ring on the pool, the state callbacks, the
head-and-pass-through install in `SpanRouting` -- and, above all, the fact that under the per-layer
cut the linear attention is THE MODEL'S OWN CODE running on the host, while under the group cut it
is the pool's reimplementation of it.

That reimplementation has been verified piece by piece against external references -- projection
split bit-identical, convolution to 0.003, gates identical, scan against the chunked kernel to
0.004, head expansion, the z-gated tail -- and it is still the thing that differs. Which means the
next measurement is not another piece. It is the whole of `_linear_attention` against
`Qwen3_5GatedDeltaNet.forward` on identical inputs, elementwise, which the dump added in 7e9ad914
exists to do.


## 2026-08-21, every early read turned off, and the arrangement is still wrong

    colocated              " Paris.\nThe capital of Germany is Berlin.\nThe"
    arrangement, shift 1   " France is France is France is France is"
    arrangement, shift 0   " France is France is France is France is"

The linear layers never had an early read in this cut -- `_linear_attention` takes q, k and v from
the current hidden (span.py:772) -- and the per-layer wiring that did convert 63 layers now stands
down, with the per-layer stream unchanged to five figures when it does. So shift 0 turns off the
last of it: the softmax query, projected from the current `x` like everything else.

Still wrong. The fault is in the span's plumbing and not in any early read.

This control is worth something the earlier one was not. The first shift-0 run was taken before
the flag was wired into the span, so it was not a shift-0 run at all; this one is, and it is also
after the row-keyed tables and the double transformation were fixed.

One observation, unexplained and recorded rather than interpreted: shift 0 and shift 1 produce
identical greedy text for twelve tokens. Two different models agreeing token for token is possible
when both have collapsed onto the same attractor, and it is also what a shift that reaches nothing
would look like. The span's `_finish` does branch on `query_shift`, and the branch was verified by
reading; whether it changes the output has not been measured on its own.


## 2026-08-21, a second transformation was found on the pool, and it was inert

Asking whether the baseline also uses Early-K turned up something else. It does not -- the
colocated model uses neither, and the arrangement moves only the softmax query: `q` and the gate
come from the read point, `k` and `v` from the current `x` (span.py:665 and 685), and the linear
layers take q, k and v entirely from the current hidden (span.py:772). So layer 1's divergence is
not a designed difference.

But the pool's own startup log said, every single time:

    afd early-q installed: coverage=all, shift=1, 63 layer(s) moved, 0 clamped

63 of 64 -- every layer except LAYER 0, which is exempt because nothing sits beneath it. The pool
was installing the PER-LAYER early-q wiring and then running the GROUP cut over the same model,
and the exempt layer is precisely the one layer that has matched the colocated model exactly
throughout this search. It looked like the answer.

It is not. With the per-layer wiring stood down, the pool's stream is unchanged to five figures:

    layer 0, x + attn   28.773  (was 28.773)
    layer 1, x + attn   54.736  (was 54.736)

`install_early_q` converts by wrapping `layer.forward`, and the span never calls `layer.forward`
-- it calls the submodules directly. The second transformation was real, was installed, and was
inert for the path being measured.

Fixed anyway: the group cut implements its own read point, so the per-layer wiring now converts
nothing under `--afd-span-cut`. Two runs of a knob that means two different things on one model is
a defect whether or not it currently bites.

Also recorded: `--afd-coverage` appears nowhere in `span.py` or `span_routing.py`. Every run of
this arrangement has passed `--afd-coverage all` and the group cut has ignored it, the same way it
ignored `--afd-query-shift-layers` until that was wired. A run that reports a setting it never
applied is the failure mode the skill names, and this is the second instance of it here.


## 2026-08-21, precision is not the mismatch, and the divergence is a direction not a scale

Every buffer and every stage, logged on both sides rather than assumed:

    the model    ssm float32, conv bfloat16, hidden bfloat16
    the pool     state float32, conv bfloat16, hidden bfloat16, reading float32,
                 core bfloat16, gated bfloat16, out bfloat16

The same. The pool contracts in float32 and rounds to bfloat16 after the mix; the model holds its
state in float32 and its hidden in bfloat16. The 11% to 25% bfloat16 state error measured in
`state_precision.py` does not apply, because neither side keeps its state in bfloat16.

The like-for-like stream comparison, 122 rows, row 0:

    embedding                 0.85246  against  0.85246   identical
    layer 0, x + attn        28.773    against 28.775     exact
    entering layer 1         38.501    against 38.750     -0.6%
    layer 0 feed-forward in  11.570    against 11.587     -0.15%
    layer 1, x + attn        54.736    against 51.493     +6.3%
    layer 1 feed-forward in  11.554    against 12.047     -4.1%
    entering layer 2         55.413    against 52.189     +6.2%

Layer 1 is handed a stream 0.6% low and returns one 6.3% high, so the excess is its own attention
contribution and not something it inherited.

The last line is the informative one. `post_attention_layernorm` removes the scale, so its output
is nearly dimensionless -- and it still differs by 4.1%. What leaves layer 1 is not the right
vector made too large; it points somewhere else.


## 2026-08-21, the scan equals the chunked kernel, so the prefill algorithm is cleared

    scan against the chunked kernel          relative 0.0042   cos 0.999991
    control, two rows swapped on one side    relative 0.5412   cos 0.852554

`chunk_gated_delta_rule` over a chunk of eight against the token-by-token `read_one`/`update_only`
scan, zero initial state, at Qwen3.8-27B's own head shapes. bfloat16 rounding, with a control that
moves by two orders of magnitude. The prefill algorithm is not the fault, and the reference gap
named in the entry below is now closed -- in the direction of no fault.

One correction to that entry. The asymmetry it records belongs to the FlashInfer kernel, and this
deployment runs `linear_attn_backend='triton'`, whose `extend` calls `chunk_gated_delta_rule(...,
use_qk_l2norm_in_kernel=True)` -- normalising inside the kernel, the same as decode. The entry
below described a path the deployment does not take.

Kept as `test/registered/unit/test_afd_prefill_scan.py`, because the two implementations are free
to drift apart under any upstream change to either and nothing else in the tree would notice.

## 2026-08-21, the prefill and decode paths are not the same call, and only decode has a reference

Reading what the model actually calls, rather than assuming the two paths agree:

    the gate      `g = -exp(A_log) * softplus(a + dt_bias)`, alpha = exp(g), beta = sigmoid(b).
                  Identical to `linear_history.gates`. Ruled out.
    decode        `fused_recurrent_gated_delta_rule_packed_decode(..., scale=None,
                  use_qk_l2norm_in_kernel=True)` -- the kernel normalises q and k itself
    prefill       `self._prefill_fn(..., scale=None, use_qk_l2norm_in_kernel=False)`, with
                  `q_fi = l2norm_fwd(q[0])` and `k_fi = l2norm_fwd(k[0])` done OUTSIDE it

Same recurrence, two implementations, and the normalisation crosses the kernel boundary in
opposite directions between them. `gdn_split.py` verifies the span against the DECODE one. The
model's prefill uses the other, and nothing has ever compared the span to it.

That is where the next measurement goes, and the shape of it matters: the span walks a chunk token
by token through `read_one`/`update_only` while the model runs a chunked delta rule over the whole
chunk at once. Those are equal in exact arithmetic and need not be in bfloat16, and "need not be"
has to be measured rather than argued -- with a control, because a loose threshold passes anything.


## 2026-08-21, the stages of a linear layer, and where the divergence cannot be

Layer 0's stages beside layer 1's and layer 2's, row 0 of a first prefill:

    layer   hidden   alpha mean   beta mean   q~      reading   s        core     out
    0       69.19    0.747        0.724       0.606   0         0.0672   0.375    28.75
    1       29.72    0.655        0.641       0.607   0         0.0639   0.0326   16.75
    2       30.59    0.693        0.676       0.603   0         0.0860   0.0221   16.60

Nothing steps out of line. The gates, the query coefficient and the write strength are the same
order across all three, and `reading` is exactly zero everywhere, which is what row 0 of a first
prefill must read from an empty state.

That last zero is the useful part. If row 0 reads nothing and its core is `s * v` alone, then
whatever makes layer 1 differ from the model cannot be at row 0 -- it has to come from the LATER
rows of the chunk, which is the sequential scan. And the scan is the one piece whose reference has
never been the right one: `gdn_split.py` verifies the recurrence against
`fused_recurrent_gated_delta_rule_packed_decode`, the DECODE kernel, while the model's prefill
runs a CHUNKED delta rule. Two algorithms for one recurrence, verified against the wrong one.

A fourth instance of comparing two correct measurements of different quantities happened writing
this. `core` is (rows, heads*dim) and `gated` is (rows*heads, dim), because the z-gated norm
reshapes for its own kernel, so `t[0]` was a whole row of one and a single head of the other --
printed as 0.0107 beside 1.4261 on the same line. Everything is reshaped to (rows, -1) first now.
The running count of this mistake in this search is four.


## 2026-08-21, the feed-forward is ruled out and layer 1's attention is what is left

Layer 1's input is layer 0's residual plus layer 0's FEED-FORWARD output, and that output had
never been compared to anything. An attention that is perfectly correct computes a wrong answer
from a wrong input, so the feed-forward between the exact layer and the inexact one had to be
ruled in or out before the attention was blamed -- and these are MoE layers, where "same module,
same weights, therefore same output" is an assumption rather than an argument.

    layer 0 feed-forward   10.588  against 10.654    -0.6%, bfloat16
    layer 1 feed-forward    1.685  against  1.789    -5.8%

Layer 0's is fine. Layer 1's is off by the same order as everything downstream of layer 1, which
is what a correct function of a wrong input looks like.

So layer 0 is right in all three of its parts -- x + attn to five figures, the feed-forward to
0.6% -- and layer 1's attention is the first thing that is wrong. That is now agreed by three
measurements that share no code: the per-layer residual bisect, the feed-forward comparison here,
and `watch_linear_attention`, which feeds one hidden state into both implementations and gave
0.0003 at layer 0 against 0.049 at layer 1.

What layer 0 has that layer 1 does not, on this side, is still the question. Both enter the same
loop with a zero state and an empty ring; the only visible difference is that layer 0 is entered
with residual=None.


## 2026-08-21, the state starts clean, so layer 1's 6.3% is the arithmetic and not the start

The layer-0-exact-layer-1-off pattern has two shapes of explanation: the recurrence computes the
wrong thing, or it computes the right thing from the wrong starting point. A slot handed to a new
request without clearing would give exactly the second, and would leave layer 0 alone if that
layer's buffer happened to be untouched.

One number decides it. The first token of a request's FIRST chunk contracts a state nothing has
written to, so its reading is exactly zero or the slot carries a previous occupant's history:

    first-read layer 0   |0|   rms 0
    first-read layer 1   |0|   rms 0
    first-read layer 2   |0|   rms 0

Exactly zero. The start is clean and stale state is out.

The first version of that probe fired on any single-row call and read 6.21, which looks like the
answer and is not: a single-row call is usually a DECODE step, whose state is legitimately
non-zero. It is now gated on a multi-row chunk from a request with no residual yet, which is a
first prefill and nothing else.

Both allocators were read while forming this. `HistoryCache.release` zeroes state and conv for
every layer unconditionally. `LinearStates.release` zeroes only the (slot, layer) pairs in
`_touched`, and `note_touched` is called for the convolution and nowhere else -- which is correct
only because the group cut puts the recurrent state on the host and the convolution on the pool.
It is right by coincidence of who holds what, not by construction, and moving either would break
it silently.


## 2026-08-21, layer 0 is exact and the divergence enters at layer 1

The prologue bisect, comparing the same quantity on both sides at last:

    embedding             0.85246  against  0.85246   identical
    layer 0, x + attn    28.773    against 28.775     exact to bfloat16
    layer 1, x + attn    54.736    against 51.493     +6.3%

The first attempt at this compared the arrangement's x + attn + mlp against the model's x + attn
and called it +34%. sglang's layer returns the stream BEFORE its own feed-forward -- 
`postprocess_layer` hands back (mlp output, x + attn) and the next layer's `prepare_attn` does the
addition -- so those are different quantities. That is the THIRD time in this search that two
correct measurements of different things were compared: a trace that walked rows against a
reference that walked layers, a control that reversed a filter on both sides at once, and this.

With that fixed, the reading is sharp. The embedding the host sends is exactly what the model
computes, layer 0 agrees to five figures, and layer 1 does not.

AND IT RETRACTS THE PREVIOUS ENTRY'S CONCLUSION. The linear attention compared against the model's
own, seeded from the model's own state, gave 0.0003 at layer 0 and 3 to 18 percent everywhere
else; that was read as bfloat16 accumulation noise because the per-channel ratios were scattered
rather than constant. The same pattern -- layer 0 exact, every other layer a few percent off --
now appears in a completely independent measurement. Noise does not come out systematically zero
at one particular layer in two unrelated comparisons. It is not noise.

So the question is what layer 0 has that layer 1 does not, in the span's path. Both run the same
loop in `_linear_run`; the visible difference is that layer 0 is entered with `residual=None`
while layer 1 is entered with a real residual, and that layer 0's state and convolution ring are
the first to be touched for the request.


## 2026-08-21, the residual grows again, and the prologue is the only span left out of line

The residual comparison was last run BEFORE the row-keying fix. Rerun after it, against the same
per-layer reference from the same process:

    the arrangement, per span boundary      the model, per layer
    span 0  (after layer 2)   76.76         layer 2    68.93      +11.4%
    span 1  (after layer 6)  110.65         layer 6   112.00       -1.2%
    span 2  (after layer 10) 168.15         layer 10  170.32       -1.3%

The collapse is gone -- it was the tables keyed by request, and 34.73 at span 1 has become 110.65
against the model's 112.00. Spans 1 and 2 now agree to about one percent, which is the size of the
linear attention's own accumulation noise.

Span 0 does not. It is 11% HIGH, and it is the only span produced by a different function:
`run_prologue` starts from the embedding and has no `W_o`, no output gate and no attention output
at its head, where every other span begins with all three. Every other span agreeing to 1% while
the one with its own code path sits at 11% is the sharpest single reading this search has
produced.

The linear attention itself was compared against the model's own in the same process, seeded from
the model's own state, and the difference is 3 to 18 percent with a per-channel ratio near one and
a few percent of scatter -- the shape of bfloat16 accumulation over a 128x128 contraction plus a
rank-one update, not of a wrong factor. Layer 0 agrees to 0.0003. That is consistent with the
composition being right and leaves the assembly around it as what to look at.


## 2026-08-21, the composition compared at last, and the confound in the first reading

Every piece of `SpanRunner._linear_attention` had been checked against something outside itself and
every piece was right. The composition had not, because the comparison needs a real ForwardBatch
and a span never runs inside a model forward. The pool has one on its own port: during a colocated
request `Qwen3_5GatedDeltaNet.forward(hidden, forward_batch)` is the real thing, so a hook there
can run the span's reimplementation on the same tensor, in the same process, on the same weights.

First reading, one row, the model against the span:

    layer 0   rel 0.194  cos +0.987      control (shuffled input)  rel 0.462  cos +0.889
    layer 1   rel 0.644  cos +0.883                                rel 0.765  cos +0.792
    layer 2   rel 0.748  cos +0.665                                rel 1.139  cos -0.070
    layer 4   rel 0.770  cos +0.914                                rel 0.954  cos +0.686
    layer 5   rel 0.839  cos +0.615                                rel 0.977  cos +0.217

19% to 84% relative, against bf16 rounding of 0.003 in the convolution comparison. The span is
nearer than the shuffled control but nowhere near the model.

CORRECTED, and the corrected reading is below. The paragraph that follows was right to refuse the
first one.

With the state seeded from the model's own -- a PRE-hook, because a forward hook fires after the
layer has advanced its cache -- and reaching the LINEAR backend rather than the hybrid wrapper
that holds no forward_metadata of its own:

    layer 0   rel 0.000293  cos 1.0000    control 3.50    seeded state |222.67|  conv |735.26|
    layer 1   rel 0.0493    cos 0.9993    control 0.91                  |4.50|        |69.21|
    layer 2   rel 0.1793    cos 0.9861    control 1.22                  |2.63|        |73.49|
    layer 4   rel 0.0768    cos 0.9979    control 1.20                  |2.48|        |84.60|
    layer 5   rel 0.0576    cos 0.9996    control 1.21                  |1.51|        |82.05|
    layer 6   rel 0.0339    cos 0.9995    control 1.82                  |2.94|        |88.04|

Layer 0 is exact to bf16 rounding. Every other layer is 3% to 18% off at a cosine of 0.986 to
0.9996 -- so what differs is mostly MAGNITUDE and hardly at all direction.

The state's magnitude was logged to decide a fork: if the error appeared only where the state is
non-zero, the suspect would be the seeding rather than the span. It resolves the other way. Layer
0 carries the LARGEST state of all (222.67 against 1.5 to 4.5) and is the one that agrees exactly,
so the error does not track the state at all and the seeding is not what it measures.

What distinguishes layer 0 from every other linear layer is now the question. It is the first
layer of the stack, its convolution history is ten times the others', and it is the only one whose
span input comes from the embedding rather than from a previous layer's output.

The refused first reading, kept because refusing it was the right call:

THIS IS NOT YET A FINDING. The span's state and ring are ZEROED for the comparison and the model's
are whatever its own mamba cache holds -- and the warmup ran before this, on the same slots. Two
recurrences started from different states differ for that reason alone, and the size of the
difference says nothing until they start from the same one.

The fix is available and specific: the hook already receives `forward_batch`, which carries the
cache indices, so the span can be seeded from the model's own conv and recurrent state rather than
from zero. Until that is done these numbers are a measurement of an uncontrolled difference.

The hook also has to be unable to kill what it measures. Its first version reached for
`LinearStates.state`, which does not exist -- the class keeps `_states` behind `buffer(layer)` --
and raising inside a forward hook took the scheduler down rather than skipping the diagnostic.


## 2026-08-21, four suspects cleared and no fault found

A round that eliminated rather than fixed. Each of these was a plausible per-token fault -- wrong
in the simplest case, one token and no history, which is the constraint the remaining fault has to
satisfy -- and each is now measured rather than argued.

    the fused split         `Qwen3_5GatedDeltaNet.forward` takes a FUSED branch when
                            num_v_heads // num_k_heads is in _GDN_FUSED_QKVZBA_RATIOS, which on
                            CUDA is (1, 2, 3, 4). The deployed model is 48/16 = 3, so it does; the
                            span always calls `fix_query_key_value_ordering`, the other branch.
                            Both are pure functions of the projections, so the comparison needed no
                            ForwardBatch: mixed_qkv, z, b and a are BIT-IDENTICAL between them
    the z gate at the tail  the model applies `self.norm(core_attn_out, z)` before `out_proj`. The
                            span does the same, on the same reshape
    the output gate         cleared last round and restated here: 15x attenuation, matching the
                            model's own o_proj input in the same process
    the q/k scale           `normalise` applies `head_k_dim ** -0.5` to the QUERY only, and leaves
                            the key at unit norm. That is what `gdn_split.py` verified against the
                            fused kernel. Scaling both would have made every (k.q) 11x too small
                            and gutted all 48 linear layers -- which fits the symptoms exactly, and
                            is not what the code does

The convolution is cleared too, and now has the coverage it never had. `_convolve` against
sglang's own `causal_conv1d_update` and `causal_conv1d_fn` on the tiny stack: 0.0029 relative for
one token against an empty ring, 0.0025 for a chunk of four where every tap participates, both at
bf16 rounding. The control reverses the taps on ONE side and gives 1.40 and 1.45 -- the first
version reversed them on both, which changes the same tap in each and agrees again. That is the
second control in this line of work that could not fail, and the second caught by looking at what
it actually discriminates. The checkpoint has 48 conv1d tensors and no bias, so the span adding
none is right; its docstring says `bias[c]` and is wrong about a model that has none.

With that, every PIECE of `_linear_attention` has been checked against something outside itself:
the projection split (bit-identical), the convolution (here), the query and key scaling and the
gates and the recurrence (`gdn_split.py`, against the fused kernel), the head expansion
(`repeat_interleave`, matching that same reference), the z-gated norm and the output projection
(the model's own tail). What has NOT been checked is the COMPOSITION -- correct pieces in the
wrong order, or with one missing, is still wrong -- and that needs a real ForwardBatch, which the
pool never has because it is not inside a model forward.

The way to get one without building it: the pool serves colocated requests on its own port, and
during such a forward `linear_attn.forward(hidden, forward_batch)` runs with a real one. An
env-gated hook there can run the span's `_linear_attention` on the same hidden states and compare,
in the same process, on the same weights, against the real thing.

The older note this replaces:

What that leaves is the one part of the linear attention that NO check covers. `gdn_split.py`
starts from `mixed`, which is the projection AFTER the convolution, so the span's `_convolve` has
never been compared to anything. With one token and an empty ring the convolution is decided
entirely by the last tap, and taking the wrong one is wrong in exactly the simplest case.


## 2026-08-21, the fourth fault: a chunk's rows shared one residual

Found by matching two log lines that had never been printed side by side. Same request, same
boundary, one value:

    76.76   written leaving the prologue
    18.833  read entering the next group

The pool's residual and gate tables were keyed by REQUEST. A decode batch is one row a request, so
that was right there and only there. A 122-token prefill is 122 rows of ONE request; each row
overwrote the last, the FINAL token's residual survived, and the next span handed it back to all
122 positions. Every position in the prompt ran the rest of the stack on the last token's state.

Fixed in 27806f37 with five cases, red on the per-request table. Deployed, and the output CHANGED
without becoming right:

    "The capital of France is"   before  " is is is is is is"
                                 after   " France is France is France is"
    colocated                            " Paris.\nThe capital of France is Paris."

It now echoes a longer stretch of the prompt instead of only its last token, which is what a
partly-restored causal structure looks like. The single-token prompt is unchanged at "TheTheThe",
and with one row this fault could not have applied to it -- so at least one more fault is present
in the simplest case there is: one token, no history, tables trivially consistent, scan equal to
read, convolution over a single position, attention attending to itself.

Two suspects were cleared by measurement on the way here, and clearing them is what left this one
visible:

    the output gate         attenuates the attention output 15x, which looked damning until the
                            model's own o_proj input was hooked in the same process: 3.40 at layer
                            3 against the span's 2.14, 2.68 against 1.82, 2.47 against 2.25. A
                            trained gate on this model is simply that closed
    anti-correlation        the residual falling by two thirds was read as something being added
                            against the stream. The cosines are mostly POSITIVE (+0.92, +0.67,
                            +0.46). The inference was refused by the measurement it prompted

Four faults now, each real, each verified red and green, none of them the whole answer. Every one
was found by an instrument built after the previous one was fixed, and none by a check that
existed before deployment. The next thing to build is still the same thing: the span against the
model's own layers, on one row, where nothing else can be blamed.


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


## 2026-08-22, the group cut's fault: a slot handed on with the previous request still in it

Found by the ladder, on the rung it was built for, and it is one call that nobody makes.

Both slot tables have a `release` that zeroes the recurrent state and the convolution ring --
`LinearStates.release` on the pool, `HistoryCache.release` on the host -- and `OP_RELEASE` has
existed in the protocol since the beginning, carrying the comment that says exactly why: "sglang
reuses slots and the next one is not this one". It reached the KV cache and the attention holder.
It never reached the span, and no host ever sent it.

A KV cache survives a reused slot, which is why this went unnoticed: a length of zero already
excludes stale positions. A recurrent state has no length. Whatever is in the buffer IS the
history, so the second request through a slot is conditioned on the first one's prompt.

### How it was cornered

Each step is a control the step before it made possible.

    rung 2, all 48 linear layers on the pool     not identical, relative 1.28, cosine 0.03
    rung 2, ONE linear layer on the pool         not identical, relative 1.27 -- so not
                                                 accumulation across layers
    rung 2, ZERO linear layers (same installer)  IDENTICAL, drift 0.005 -- so not the installer,
                                                 the wire, the feed-forward offload or the harness
    the moved layer against the model's own,     fresh pool, first request: relative 0.0026 mean,
    same input, same call, same occasion         0.012 worst, cosine 1.0000
                                                 second request, same slot: 0.31, and the text
                                                 became " the same as the same as the same as"

The last line is the whole finding. The first request through a fresh pool was always right, and
every in-process check ever run was a first request.

That is also why the fault looked like it moved: rung 3 and rung 4 measured 1.23 and 1.17 with
cosines of 0.13 and 0.22, and those numbers are not a shifted read point being slightly worse than
a grouped one. They are two runs of the same contaminated slot.

### The fix

`slot_reset.forget_starting_requests`, called once a forward pass by both cuts, at the first
routed layer. It releases every row id that BEGINS a request -- a prefill chunk with no cached
prefix -- on this end and, with OP_RELEASE, on the pool.

At the start of a request rather than at its end: an aborted or crashed request never reaches its
end, and the slot it leaves behind is indistinguishable from one in use.

### After the fix, both ends confirmed installed from their own logs

    arrangement                       "The capital of France is"   "Explain why the sky..."
    rung 2, 48 layers on the pool     IDENTICAL, drift 0.0062      IDENTICAL, drift 0.0198
    group cut, shift 0                IDENTICAL, drift 0.0062      IDENTICAL, drift 0.0092

The third prompt, a bare "The", parts at token 1 under both -- and its continuation is fluent and
unrelated ("the following excerpt is taken from a philosophical text" against "the following table
lists the average annual salaries"), which is the near-tie a one-token prompt is, not the
repetition loop. The drift at token 0 is 0.008, at bfloat16 rounding, before the argmax parts.
The skill's own warning applies to it: batch composition alone parts the same model's greedy
output on 3 of 4 prompts, so a single parted prompt is not yet a finding either way.

### What this says about the instruments

Every in-process check that passed while the deployment degenerated was correct AND blind, by one
shared property: it ran one request. A recurrent state's whole failure mode is what the SECOND
request sees. Any check of a stateful cut has to run at least two requests through one slot, and
the second one is the test.

### The fix, checked the way the fault would have been caught

Group cut at shift 0, both ends confirming their own installation, each prompt asked THREE times
through the same slots, and each side compared against its own earlier answer:

    prompt                              pass 0      pass 1      pass 2
    "The capital of France is"          identical   identical   identical
    "The"                               part @1     part @1     part @1
    "Explain why the sky is blue..."    identical   identical   identical

Every per-token relative difference is repeated to the digit across passes (0.0092, 0.0096, 0.0108
...), and neither side ever disagreed with its own pass 0. Before the fix the second pass was a
different arrangement.

Two requests in flight together through one pool answer exactly as they do one at a time, on both
prompts -- so the slot table holds up when two of them are live at once, which is the case a
single-request check cannot reach either.

`rung_verdict.py` now asks every prompt twice by default and VOIDS the verdict, non-zero, when a
side disagrees with its own earlier answer. The instrument that missed this for days could not
have reported it: it asked once.

What is NOT yet covered: a request that is aborted or retracted mid-generation. Its slot is
released by the next request that lands on it, which is the design, but nothing has exercised the
path.


## 2026-08-22, the two lines get two branches

`afd/main`, forked from origin/main, carries standard AFD alone: the host runs every attention and
owns the KV cache and the recurrent states, the pool holds the static weights and answers
feed-forward frames. 60 files, 248 unit cases, and no derived package anywhere in the tree. It is
named `afd/main` rather than `afd` because `afd/span-cut` already occupies that namespace and git
cannot hold both.

`afd/span-cut` keeps the derived line and is what the deployment runs.

Forking is what found the contamination. The property "standard AFD runs with the derived code
ABSENT" had a ratchet, the ratchet was green, and it was grepping `srt/afd` alone -- while two
files outside that directory imported the derived package by name, one of them the argument hook,
which runs before anything else in the process. A test that cannot see the offender reports the
property it was written for.

    arg_groups/afd_hook.py   validated two of the arm's flags, imported its installer by name
    model_runner.py          FROZEN, imported the derived package in a `try`, read three of the
                             arm's server args off the runner, and reached two levels into the
                             arm's own object for the sweep schedule
    server_args.py           still holds the arm's three flags on the derived branch; absent on
                             afd/main. The remaining gap, recorded rather than closed

The repair is three named entry points on the shared registry -- `check_args`, `install_transforms`,
`sweep_schedule` -- each answering a question a shared file legitimately has, none of them naming
an arm. The checks themselves moved unchanged into `afd_query_shift/arg_checks.py`.

Re-deployed on the rewired installation, group cut at shift 0, two passes: every per-token
relative difference is bit-identical to the run before the refactor (0.00615874, 0.00787484,
0.00920967, and the same cosines). A relocation that changes a number is not a relocation.

Still open on this line: the travel-group batch re-forming as a documented property of the shared
half (the pool's departure already re-forms per layer, which IS the coach model -- it has never
been stated or measured as one), and the two-batch staggered time-division multiplexing, which
doubles the in-flight requests and the host's KV cache and must be measured rather than assumed.


## 2026-08-22, two callers on one pool: what re-forming across callers actually buys

The pool holds no per-request state, so two callers' calls are the same kind of transaction and
can leave on one bus -- which is what makes re-forming across CALLERS, and therefore across NODES,
possible at all. `benchmark/afd/two_callers.py` measures it: N independent connections, which is
what a second host is to this pool, issuing at one layer at once. Qwen3.8-27B, 4 tokens x 5120 a
call, 30 rounds, median round trip:

    callers     min_batch 1      min_batch 2
       1          0.54 ms          5.33 ms      <- the lone caller waits out max_wait 4 ms
       2          0.93 ms          0.76 ms
       4          1.77 ms          1.67 ms

    calls/s     min_batch 1      min_batch 2
       1            882              166
       2           1354             1250
       4           1240             1061

Three readings, and the third is the one worth having.

At min_batch 1 the departure leaves the instant the first frame lands, so two callers are
SERIALISED: the median rises 1.72x for two and 3.28x for four. That is the deployed configuration,
chosen for latency, and it means the bus never fills -- the coach model buys nothing in the
arrangement as it runs today, which is a fact about the setting rather than about the idea.

At min_batch 2 two callers do ride together and each one's call is 18% cheaper than when they were
serialised. The lone caller pays 5.33 ms for it -- max_wait, in full, waiting for a partner that
never comes -- so the setting trades a single caller's latency for a pair's.

And the throughput columns say the pool is NOT bound by the weight read at these widths: sharing
the read across two callers did not raise calls/s at all (1354 -> 1250). Whatever binds the pool
at 4 tokens a call is per-call cost, not bandwidth, so the case for batching across callers has to
be made at widths where the read dominates -- `pool_amortisation.py` shows the per-token cost
still falling at 512 tokens, and that is where two callers should be measured next.

The measurement is two CONNECTIONS, not two model hosts: a second host needs a second card, and
two hosts on one card contend for that card's bandwidth, so a flat aggregate could not tell a
saturated pool from a saturated host. It measures the pool's side, which is the side the claim is
about.


## 2026-08-22, #62: co-batching across callers does not pay at any width measured

The claim under test was that a second caller riding the first one's 267 MB weight read is what
makes a shared pool worth having. Qwen3.8-27B, one dense layer, 20 rounds, tokens/s (calls/s x
tokens a call), two independent connections against the real pool:

    tokens a call     min_batch 1                 min_batch 2
                      1 caller   2 callers        1 caller   2 callers
         4              3096       7122 (2.30x)      518       4100
       128             53490      27640 (0.52x)    14396      31180
       512             85641      88212 (1.03x)    30930      37837

Read the 512 row across: one caller alone at min_batch 1 does 85641 tokens/s; two callers FORCED
into one departure do 37837. Co-batching more than halves it. At 128 it is 53490 against 31180.
The min_batch 2 single-caller column is the same setting's other cost -- 4 ms of max_wait paid in
full, every call, waiting for a partner that never comes.

Why the read has nothing left to share: a 512-token frame already amortises the layer's weights
across its own 512 rows, and at that width the pool is saturated -- a second caller adds 3%. The
read is shared WITHIN a caller's frame long before two callers can share it between them.

What the second caller does buy, at 4 tokens: 2.30x, with min_batch at 1 and no co-batching
anywhere. That is PIPELINING -- one caller's wire transfer overlapping the other's compute -- and
it needs nothing but a pool that takes whoever is ready.

So the arrangement's answer is the stateless pool plus flexible re-forming, and NOT a minimum
batch. min_batch above 1 would need a width where the weight read dominates AND the pool is idle;
the range measured here does not contain one. The seating rules (`seating.py`, `boarding.py`) stay
correct and stay useful -- they decide who rides when several are ready at once -- but the payoff
they were assumed to deliver, a shared weight read, is not where the money is.


## 2026-08-22, #46: what the host actually needs on its card

Both sides of the running deployment, Qwen3.8-27B, read from their own startup accounting rather
than from nvidia-smi (which cannot separate weights from caches):

                            weights     mamba state    KV cache     total resident
    pool (everything)       51.05 GB      10.47 GB     11.86 GB        73.4 GB
    host, GROUP cut         19.18 GB       2.86 GB      3.34 GB        25.4 GB

CORRECTION, same day: the host row was first written as "feed-forward on the pool", the per-layer
cut. It is not -- that host was launched with --afd-span-cut, and its own log says "the group cut
is installed". So 19.18 GB is the GROUP cut's host, which also has its attention projections on
the pool. The per-layer cut's host keeps those and is therefore LARGER, and it has not been
measured. Attributing a number to the wrong arrangement is the error this tree has spent days on
from the other direction; it is corrected here rather than quietly re-run.

The cut removes 31.87 GB of weights, 62% of them. That is the arrangement working exactly as
described -- and it is NOT enough for the claim this line has been carrying.

**An 8 or 12 GB card cannot serve this model under either cut.** Under the group cut -- the
smaller of the two -- the weights alone are 19.18 GB: attention and linear-attention projections, embeddings, the vision tower and the norms
all stay on the host, and only the feed-forward leaves. The claim needs one of

    move more than the projections       W_q/W_kv/W_o are ALREADY on the pool in this number.
                                         What is left on the host is the embeddings, the vision
                                         tower, the norms and the linear-attention projections,
                                         and that is what 19.18 GB is
    a smaller model                      the 62% is a property of this stack's shape, not of AFD
    quantised weights                    FP8 would put the host's weights near 9.6 GB, which is a
                                         12 GB card with very little left for KV

There is a second-order cost visible in the same logs and it is easy to miss: with less memory
left over, the host's own limits shrink. `max_running_requests` was capped to 3 by the mamba state
cache against the pool's 8, and `max_total_num_tokens` is 54688 against 194276. So the cut buys
weight memory and then spends part of the gain on a smaller working budget -- any throughput
comparison between the two sides is also a comparison between a 3-request host and an 8-request
one, and must say so.

### The per-layer cut's host, measured the same way: 19.18 GB. IDENTICAL.

    host, GROUP cut         19.18 GB      (attention projections computed on the pool)
    host, PER-LAYER cut     19.18 GB      (attention projections computed here)

The prediction above was wrong and the reason is worth more than the prediction. The group cut
moves where the projections are COMPUTED; it does not change what the host ALLOCATES. The loader
makes exactly one thing absent -- "afd loader: feed-forward built with storage=False" -- so under
the group cut the host holds attention projection weights it never multiplies by anything.

That is an unclaimed reduction sitting in plain sight, and it is the only remaining route to the
small-card claim now that "move the projections too" turns out to change no memory at all: make
them absent in the loader as well, on a host that has a pool to compute them. `absent_ffn.py`
already does this for one weight class and nothing else uses it.

(The cache rows differ between the two runs -- mamba 3.52 GB against 2.86 GB, KV 4.12 against 3.34,
max_running_requests 4 against 3 -- because the group-cut host was launched with
--enable-return-hidden-states and this one was not. Those rows are not comparable across the two
runs; the weight row is, because it is decided by the loader alone.)


## 2026-08-22, #67: the host stops holding what the pool computes

The loader made one thing absent, the feed-forward. Everything else the pool computes was still
allocated on the host: a full set of attention projections nobody multiplied by anything. Released
after the routing is installed, from the modules the routing itself names -- passenger layers
whole, and a head layer's qkv_proj, o_proj and mlp; the head keeps its norms and its weight-free
attention core.

    host, group cut          before        after
    weights                  19.18 GB      5.69 GB       (13.49 GiB released)
    mamba state               2.86 GB      9.18 GB
    KV cache                  3.34 GB     10.46 GB
    max_running_requests            3            8
    max_total_num_tokens        54688       171524       3.1x

Same card, same everything else: the freed weights become cache, which is the only form the gain
can take. The deployment answers " Paris.\nThe capital of Germany is" as before, so nothing was
released that was being used.

This is what the small-card claim needed. A 24 GB card carrying 5.69 GB of weights has ~18 GB for
cache, and a 512K context of standard attention in bf16 is 13.5 GB -- so one request of that size
fits with room, on a card that cannot hold a twentieth of the checkpoint. The claim was refused
this morning on 19.18 GB and it is the release, not the cut, that was missing.

Two things this does NOT do, stated because both are easy to assume:

    the construction peak      unchanged. The projections are QKVParallelLinear and
                               RowParallelLinear, classes shared with modules this host still uses
                               and with the vision tower, so they cannot be built on meta by class
                               the way Qwen2MoeMLP is. The model is still materialised whole and
                               then released, which needs the card to survive one full
                               construction. On a card too small for THAT, this does not help yet.
    the per-layer cut          untouched. Its host computes its own projections and needs them.

A correction on the way: the first reading said 45.36 GiB released, on a host whose whole
checkpoint is 19.18 GB. The counter moved every parameter and counted every parameter, and the
feed-forward was already on meta from the loader -- counted twice. Numbers like that get quoted.


## 2026-08-22, #65: the pool's ceiling is its own serving loop, and it FALLS under concurrency

Connections against the deployed pool, 4 tokens a call, 25 rounds:

    connections   median ms   calls/s
        1            0.53       931
        2            0.91      1403     <- the peak
        4            2.00      1241
        8            6.06       844
       16           12.14       844

GPU utilisation at the end of the sweep: 0%. The pool is nowhere near compute or bandwidth bound
at this frame width -- it is bound by its own per-call path, and it does not merely plateau, it
DEGRADES: sixteen connections get 60% of what two get, and the median goes 23x. That is the shape
of lock and interpreter contention, not of saturation.

Against what the sixteen-small-cards design needs: each 4090D-class host doing 512K context runs
~74 decode steps/s (13.5 GB of KV at ~1 TB/s), and each step needs 64 feed-forward calls. Even
with a node-level dispatcher merging all sixteen hosts' rows into one call per layer -- the best
case, and the reason #68 exists -- that is ~4700 calls/s from one node against a measured ceiling
of 1400. **3.4x short, and adding connections makes it worse rather than better.**

So the critical path for that design is the pool's per-call cost, not its bandwidth and not the
interconnect. Three candidates, in the order their evidence points:

    the serving loop      Python, per-frame, one thread a connection. 0% GPU at 844 calls/s says
                          the work is not where the cost is
    CUDA graphs (#44)     7.3 ms of launch overhead was measured once; at 0.5 ms a call that is
                          not the whole story, but it is on the same path
    frame width           at 512 tokens the pool DOES saturate (85641 tokens/s). The narrow-frame
                          regime is the one that is broken, and decode is exactly that regime

What this settles for #68: a per-node dispatcher is not an optimisation, it is a requirement. Not
because it batches -- co-batching was measured to be a pessimisation -- but because the pool gets
SLOWER with connection count, so sixteen hosts must reach it as one connection, not sixteen.

### The construction peak, closed by the fifth entry point

`#67` left the peak untouched: the model was built whole and then released, so a card too small to
build it once was still too small. The arms registry now has a fifth entry -- an arm names classes
whose EVERY instance it computes remotely, and the loader builds those with no storage at all.
The query-shift arm names `Qwen3_5GatedDeltaNet`: under the group cut every linear-attention layer
is a span's passenger, so the host never multiplies that class by anything, and unlike the
projections it is shared with nothing else in the model.

    host, group cut          before      release only     + never allocated
    peak at load             19.18 GB      19.18 GB           8.81 GB
    steady state             19.18 GB       5.69 GB           5.68 GB
    released after routing        -        13.49 GiB          3.13 GiB
    KV cache                  3.34 GB      10.46 GB          10.38 GB
    max_total_num_tokens        54688       171524            169943

Same output, " Paris.\nThe capital of". The steady state is unchanged, as it should be -- the same
weights end up absent either way. What moved is the peak, 19.18 -> 8.81 GB, and that is the number
deciding whether a card can build this host at all: a 12 GB card now can, with about 3 GB left for
cache; a 24 GB card has ~18 GB, which holds a 512K context's 13.5 GB with room.

Two mechanisms, and which one a weight can use is a property of its CLASS, not a preference. The
loader wraps classes rather than instances, so a class shared with anything the host still uses --
QKVParallelLinear, RowParallelLinear, shared with the vision tower -- can only be released after
routing. A class the arm owns entirely can be made absent before it exists.


## 2026-08-22, where a pool call's time actually goes

py-spy cannot attach to the scheduler process here (root refused), so the pool charges its own
phases. Deployed pool, 4 tokens a call, 3000 rounds a connection count:

    connections   calls/s    wire in     work     wire out
        1          1759      0.098 ms   0.108 ms  0.321 ms
        2          2320      0.122 ms   0.160 ms  0.303 ms
        8          1264      1.380 ms   2.256 ms  1.605 ms

Two things fall out, and neither was visible from the outside.

**WITHDRAWN, same day: the reply does not cost three times the work.** The instrument was wrong.
Kernels launch asynchronously, so timing `self.forward(...)` measured the LAUNCH, and the reply
path paid for the compute because `_payload_of` copies to the CPU and that synchronises. With a
stream synchronise closing the work phase, the same pool reads:

    connections   calls/s    wire in     work     wire out
        1          1663      0.102 ms   0.394 ms  0.064 ms
        2          2224      0.121 ms   0.413 ms  0.141 ms
        8          1197      1.357 ms   2.667 ms  1.620 ms

The reply is 0.064 ms of a 0.59 ms round trip. The forward is 0.394 ms -- and for a FOUR-token
frame that is not arithmetic, it is the layer's 267 MB of weights being read: 267 MB at ~0.68 TB/s
is 0.39 ms. The pool's per-call floor is a weight read, which is a hardware fact and not an
overhead to optimise away.

That also explains the 0% GPU utilisation without contradicting it: a memory-bound read occupying
0.39 ms of a 0.59 ms window is not what an occupancy sampler counts as busy.

**Under concurrency every phase inflates together, ~10x at eight connections.** Work goes 0.108 ->
2.256 ms doing exactly the same matmul on an idle GPU. A phase that takes ten times longer while
the work is unchanged is a thread waiting its turn, not a thread working: this is the GIL, and it
is why throughput FALLS from 2320 to 1264 rather than plateauing.

That settles the direction for #68. A per-node dispatcher is not about batching -- co-batching was
measured to be a pessimisation -- it is about reaching the pool as ONE connection instead of
sixteen, because the pool's own accounting says the sixteenth connection makes the first one
slower.

Note on absolute numbers: this pool instance peaks at 2320 calls/s where the #65 sweep on an older
instance peaked at 1403. The absolutes move between instances; the SHAPE -- a peak at two
connections and decline after -- is what has reproduced.

### What the corrected numbers say about the sixteen-card design

The per-call floor is one layer's weight read, ~0.4 ms, however few rows ride. So a pool serving
one layer tops out near 2500 calls/s, and the measured peak of 2224-2320 at two connections is
that floor rather than a software ceiling.

A node of sixteen 4090D-class hosts at 512K context needs 64 layers x ~74 decode steps/s = ~4700
calls/s. A per-node dispatcher merges the sixteen hosts' ROWS into one call, which is worth doing
for the GIL reason already measured -- but it does NOT reduce the call RATE, because every layer
still needs its own call every step. ~4700 calls/s against a ~2500 ceiling is **one pool per about
eight hosts at this operating point**, and that is now a number rather than a guess.

Two ways out, and neither is the reply path:

    more pools (#64)     the obvious one, and the ceiling says how many: one per ~8 hosts here
    shard layers         a pool holding a QUARTER of the layers reads a quarter of the weights per
                         call, so its floor falls with its share. Four such pools serve the same
                         node with the same total memory and four times the call rate. Untested,
                         and it changes what a host addresses per layer -- which is exactly the
                         multi-pool routing #64 has to build anyway

`#69` is closed by this measurement rather than by work: at 0.064 ms of 0.59 ms, the reply path is
not where the ceiling is. The wide-frame reading confirms the shape from the other end -- 512-token
frames give 107571 tokens/s against 6651 at four tokens, because the same read is amortised over
128 times as many rows.

### #70's premise was wrong: sharding does not shrink the read

The floor could have been two things -- a true HBM read of the layer's weights on every call, or a
cache effect that hammering one layer hides. Measured by cycling the calls across different layers
on the deployed pool, 4 tokens a call, 2000 rounds:

    layers touched     round trip     work
          8              0.58 ms     0.380 ms
         48              0.59 ms     0.383 ms

Identical. The read is real and it happens every call, so a pool holding a QUARTER of the layers
still reads a whole layer per call and its per-call time is unchanged. "Shard the layers so each
call reads less" was my own reasoning and it does not survive its first measurement.

What sharding actually buys is different and still worth having: four pools each holding a quarter
of the layers are four DEVICES serving four different layers at once, for the same total memory as
one full pool -- 4x the aggregate call rate with no weight duplicated. The gain is parallelism
across devices, not a smaller read, and that distinction decides the experiment: it cannot be
tested on one GPU, where two pools contend for the same HBM. It needs two cards, like #47.

So the sixteen-card arithmetic stands as it was: ~4700 calls/s needed, ~2500 per pool, one pool per
about eight hosts -- and the way to more is more pools, sharded to keep the memory bill flat.


## 2026-08-22, #59: an aborted request's slot, and a crash found on the way

The design clears a slot when a request BEGINS rather than when it ends, precisely because a
request that is aborted or crashes never reaches its ending. That was reasoning, and the abort
side had never been walked. `benchmark/afd/abort_then_reuse.py` walks it: start a 400-token
generation, abort it 1.5 s in with `/abort_request`, then ask the same short prompt again and
compare against colocated.

    colocated          " Paris.\nThe capital of Germany is Berlin.\nThe"
    host, no abort     identical
    after abort 0-2    identical, all three

So the slot an aborted request leaves is cleared before the next request uses it, and the design
holds on the path that motivated it.

### The crash the run exposed, which was not the abort

The first attempt found the host dead. Not from the abort -- from a pool restart earlier in the
session. Every other call on the host's path degrades when the pool goes away: the feed-forward
falls back to running locally and the router counts the degradation. The RELEASE did not. A
`PoolClosed` raised inside a forward reaches sglang as an exception in the model and the scheduler
dies, so restarting the pool killed the host on its next token -- the exact failure
`test_afd_pool_failure.py` exists to rule out, reached by a path added after it and not covered by
it.

It now reconnects once, and if that fails it refuses with a message that says why. NOT a silent
skip: the release is what stops the next request inheriting a stale recurrent state, so a host
that cannot deliver it must fail loudly rather than serve fluent text conditioned on somebody
else's prompt. A new call on an old path inherits none of that path's tolerance, and nothing warns.


## 2026-08-22, #68: the stop, end to end, and what its hop costs

A stop on the host's own node, upstream to the real pool, host pointed at it instead of at the
pool. Per-layer cut, greedy, 12 tokens:

    colocated              " Paris.\nThe capital of Germany is Berlin.\nThe"
    host through the stop  identical

So the merge, the split and the relay are right on real traffic. The cost, from the host's node,
4-token calls:

    straight to the pool     1.06 ms      878 calls/s
    through the stop         1.40 ms      694 calls/s

**0.34 ms a call.** The module docstring had guessed "tens of microseconds" for a loopback hop and
that was wrong by an order of magnitude: the hop is not a kernel copy, it is a full decode, queue,
re-encode, decode and re-encode in Python.

Which changes what the component is worth. It saves the pool's degradation under connection count
-- 0.45 ms a call at two connections against 0.84 ms at eight -- so sixteen hosts through one stop
is roughly a WASH at today's cost, and it is clearly wrong below eight hosts. The idea survives;
the implementation has to get cheaper, and the number to beat is 0.34 ms.

A second thing the run corrected: the stop reported `riders_per_merged: 1536.0` from 1536
departures of one rider each, because it divided by a merged count of zero. With min_batch at 1
every offer completes a bus on arrival and nothing merges -- correct behaviour, reported as its
spectacular opposite. It reports riders per DEPARTURE now.
