# Measuring the pool

Three scripts, each answering one question that decided something. Every number below was taken on
Qwen3.8-27B across two machines, an RTX PRO 6000 Blackwell serving as the pool.

## `two_callers.py` -- what a second caller costs, and what it buys

Independent connections against a real pool, which is what a second host is to it.

    python benchmark/afd/two_callers.py --pool HOST:8999 --callers 1 2 4 8 16 --tokens 4 512

The readings that mattered: throughput PEAKS at two connections and falls after -- 931, 1403,
1241, 844, 844 calls/s at 4 tokens -- with every phase inflating tenfold at eight and the GPU
idle. That is threads waiting their turn, not saturation, and it is why `dispatcher.py` exists.

And co-batching does not pay: at 512 tokens a call, one caller alone reaches 85641 tokens/s while
two forced into one departure reach 37837. A wide frame amortises the layer's weights across its
own rows long before two callers can share them.

## `pool_amortisation.py` -- does a wider frame cost more?

    python benchmark/afd/pool_amortisation.py --pool HOST:8999

One caller, frames of increasing width. A departure carrying 128 times the tokens costs 34 times
as much, not 128: the per-token cost falls 5.6x from 4 tokens to 512. That is the case for pooling
at all, and it says a single host talking to a single pool is the arrangement's WORST operating
point rather than its normal one.

## `launch_or_read.py` -- is the floor a read, or is it launch overhead?

    python benchmark/afd/launch_or_read.py --mib 64 267 512 --rows 4

Runs matmul chains of known size plainly and through a captured CUDA graph. On this card: 250 MB
of weights with four rows takes 0.195 ms plainly and 0.189 ms as a graph, at 1345 GB/s. So graphs
buy 3% and the floor is the read.

That number then explained the pool: its 0.394 ms per call is this model's dense feed-forward,
3 x 5120 x 17408 in bfloat16 = 510 MiB, read at the measured 1345 GB/s = 0.398 ms. The pool runs
at the card's bandwidth. Do not take the bandwidth from memory -- it was assumed at half this
value once, which made a wrong explanation fit.

## What is not here yet

The rest of this project's benchmarks measure a derived arrangement or have not been re-run since
the arrangement changed. They stay on the working branch until each one is checked against what it
claims to measure, which is a smaller job than it sounds and a real one: several of them were
written against an earlier cut.
