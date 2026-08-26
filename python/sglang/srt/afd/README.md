# AFD: attention and feed-forward on different machines

The host runs every attention and owns everything that belongs to a request. The pool holds the
static weights and answers feed-forward calls for whoever asks, holding nothing between calls.

    host    every attention -- softmax and linear alike -- plus the KV cache and the recurrent
            states. These belong to the request, and a pool holding either could not be released
            between one request's own calls
    pool    the feed-forward, and only it. It reads the same weights whatever the caller's
            history, so a request sweeping a million positions and one sweeping a thousand cost
            it the same

The split is by STATE rather than by cost, and that is what makes the pool shareable.

## Running it

Two servers, one flag each. The pool first:

    python -m sglang.launch_server --model-path MODEL \
        --afd-mode pool --afd-bootstrap-port 8999 \
        --disable-cuda-graph --port 31000

Then the host, pointed at it:

    python -m sglang.launch_server --model-path MODEL \
        --afd-mode host --afd-pool-addr POOL_IP:8999 \
        --disable-cuda-graph --port 31001

The host serves the OpenAI-compatible API as usual; the pool serves one too, and answering
requests itself is a legitimate way to use it as a colocated reference.

**Restart the host whenever you restart the pool.** The host reconnects for feed-forward calls,
but a pool that disappears mid-generation fails the requests in flight.

**With ONE host, add `--afd-min-batch 1` to the pool.** The default is 2, which asks every
departure to wait for a second caller, and a single host never provides one -- so each of the 64
layer calls pays `--afd-max-wait-ms` before leaving. Measured on the deployment below, same host
flags, same prompt, identical output text:

    pool --afd-min-batch 1      32 tokens in  4.256 s     7.5 tok/s
    pool --afd-min-batch 2      32 tokens in 18.205 s     1.76 tok/s

4.3x, and the 436 ms a token it costs is 64 layers times the 5 ms wait. The default is right for
a pool with several hosts on it, which is what a pool is for; it is wrong for the two commands
above, which is what anybody runs first. At `--afd-min-batch 1` a departure still carries whatever
is already queued, so it batches opportunistically rather than not at all.

## What it costs and what it buys

Measured on Qwen3.8-27B, bfloat16, 64 layers, across two machines on a LAN: the pool on an
RTX PRO 6000 Blackwell (97 GiB), the host on an RTX 5090 (31.36 GiB). sglang reports memory as
`/(1 << 30)`, so every "GB" it prints -- and every number below -- is a GiB.

    whole model, resident on the pool        51.05 GiB
    AFD host, resident weights               19.18 GiB
    the host therefore does not hold         31.87 GiB, 62% of the checkpoint
    the same model colocated on the 5090     FAILS TO CONSTRUCT, inside create_weights, before
                                             any token: 340 MiB wanted, 238.50 MiB free

That last row is the comparison that matters. It is not that AFD is cheaper on the host card; it
is that there is no colocated run on that card to be cheaper than.

What the freed memory becomes, on the host card at `--mem-fraction-static 0.88`:

    KV cache                                 4.12 GiB (K 2.06 + V 2.06)
    max_total_num_tokens                     67519

A pool call costs what its weight read costs, and the read does not depend on how many tokens
ride it. Loopback, one layer, width 5120, `--afd-min-batch 1`:

    tokens a call      1      4     16     64    256
    median ms       0.55   0.54   0.61   0.86   2.14
    tokens/s        1233   6244  18232  29271  80731

Sixteen tokens cost what one costs. That flat region is the whole argument for a pool: the layer's
weights are read once and everyone on that departure shares the read.

    one pool serves    ~1400-1800 calls/s for one layer at one or two callers, falling to ~990 at
                       eight -- connections waiting, not work. See push-back item 5 in the pull
                       request description

Tokens match the colocated model under greedy decoding on the prompts checked. They are NOT
guaranteed to under concurrency, and the reason is measurable rather than mysterious: over 1811
decoded positions, **36 (1.99%) had the top two candidates at exactly equal logprobs** -- not
close, equal to the last bit of the float32 the server returns, because the logits are bfloat16
and two tokens land on the same value. At such a position the argmax is decided by the order of
a reduction, and this arrangement reduces in a different order than a colocated one. About one
position in fifty is available to turn over; none of them means a request read another's history.
`benchmark/afd/how_often_tied.py` is that measurement.

## What it does not work with yet

`compatibility.py` refuses these at startup rather than at the first token, and says why:
tensor and pipeline parallelism, data-parallel attention, speculative decoding, LoRA, multimodal
inputs, and CUDA graphs on the decode path (hence `--disable-cuda-graph`). Each refusal names the
reason it is refused rather than the flag it saw.

## Reading the source

    protocol.py       the wire: frames, ops, and what each carries
    pool_server.py    the pool: departures, the boarding rule, and what it refuses
    roles.py          the composition root -- which side this process is, and what it installs
    pool_client.py    the host's end of the socket, including reconnection
    absent_ffn.py     why a routed host never allocates the weights it routes away
    boarding.py       when a call departs, and why the answer is recomputed rather than cached
    dispatcher.py     an optional stop in front of the pool, for a node with many hosts
    arms.py           where a derived arrangement announces itself. Nothing here names one

The tests under `test/registered/unit/afd/` are written to be read: each case's docstring
says which failure it guards and, usually, when that failure actually happened.

## Serving another model family

The host is an attention service: it builds from the pool's ATTENTION
MANIFEST (`manifest.py`) and speaks two state algebras -- `softmax_kv` and
`gated_delta`. What a new family costs depends only on what it is made of:

1. **Its layers speak the known algebras** (any mix of standard attention and
   gated-delta linear attention): the HOST needs nothing. The POOL needs the
   family's span adaptation -- `layer_kinds` must recognise the decoder layer
   classes (the kind is read off the class NAME: "Linear" / "Attention"), and
   the span runner must find the modules it cuts around (`linear_attn.conv1d`,
   `layer.attn`, the mlp). A family that follows those names serves as-is.
2. **A new state algebra** (Mamba-2, GLA, ...): teach both ends the kind --
   its manifest spec (widths), its host state kernel (what `HistoryService`
   does for gated_delta), and its pushed-weight slots. The manifest's `kind`
   field is where it announces itself; a host that meets an unknown kind
   refuses by name rather than serving a guess.
3. **The early read (query shift)** is per-operator: a new linear operator
   that wants shift 1 also needs its cook (`afd_query_shift/pool_cook.py` is
   the reference shape). Shift 0 -- standard AFD -- needs none of it.

The verification ladder, in order, all of which exist as reusable pieces:
unit suite against the model-less FAKE POOL (`test_afd_fake_pool.py` speaks
the real wire); then a live pair with 32-token greedy probes; then top-5
logprob capture against the family host on ONE pool, compared bitwise --
capture on a FRESH server right after warmup, because request history
(radix cache) shifts logprobs at the 0.1 level and reads as a regression;
and when logits disagree, the wire-level ENTER dump on the pool is what
locates the divergence (the M-RoPE position scheme was found exactly there:
v, which carries no rope, matched to the bit while k drifted).
