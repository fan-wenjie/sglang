"""Does a departure carrying more tokens cost more? That is the whole case for pooling.

A pool serving one host reads a layer's 267 MB of weights to transform four tokens. If the same
read serves forty, the pool's cost per token falls by ten and disaggregation starts to pay; if the
service time rises linearly with tokens, pooling buys nothing and the arrangement is only ever a
way to fit a model on smaller cards.

Measured by sending the real pool frames of increasing width. This is deliberately NOT two hosts
sharing the pool: two host engines on one card contend for that card's own bandwidth, so a flat
aggregate would not distinguish a saturated pool from a saturated host, and the experiment could
not answer the question it was run for.

Measured across the two machines, Qwen3.8-27B-FP8, one dense layer:

    tokens per departure      round trip      per token
             1                 0.81 ms         809 us
             4                 1.21 ms         302 us      <- what one host sends today
            16                 2.10 ms         131 us
            64                 5.20 ms          81 us
           256                17.17 ms          67 us
           512                27.42 ms          54 us

So a departure carrying 128 times as many tokens costs 34 times as much, not 128: the cost per
token falls 5.6x. That is the case for pooling, and it is why one host talking to one pool is the
arrangement's worst operating point rather than its normal one.

## What this does NOT separate, and why the obvious control fails

The fall has two causes -- the pool's weight read being shared, and the fixed per-call cost of the
wire being shared -- and only the first amortises across HOSTS, because each host sends its own
frame. Subtracting a pool whose forward is the identity looked like the way to separate them and
is not: that pool runs on the CPU, and above about 32 tokens its own copies cost more than the GPU
pool's feed-forward, so the subtraction goes negative. A valid control needs an identity pool on
an idle GPU, which this pair does not have spare. The split is unmeasured and is left unmeasured
rather than reported from a control that does not hold.
"""
import json
import statistics
import sys
import time

import torch

HIDDEN = 5120
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def main():
    sys.path.insert(0, "/home/user/experiment/sglang/python")
    from sglang.srt.afd.pool_client import PoolClient

    client = PoolClient(sys.argv[1], 10.0)
    rows = []
    try:
        for tokens in BATCHES:
            hidden = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
            for _ in range(10):
                client.collect(client.issue(1, 0, hidden), "cuda")
            spans = []
            for i in range(60):
                t = time.perf_counter()
                client.collect(client.issue(i + 1000, 0, hidden), "cuda")
                spans.append(time.perf_counter() - t)
            ms = statistics.median(spans) * 1e3
            rows.append({"tokens": tokens, "round_trip_ms": ms, "us_per_token": ms * 1e3 / tokens})
            print(f"    {tokens:4d} token(s): {ms:7.3f} ms round trip = "
                  f"{ms * 1e3 / tokens:8.1f} us/token", flush=True)
    finally:
        client.close()

    base = rows[2]["us_per_token"]           # 4 tokens, what one host sends today
    best = min(r["us_per_token"] for r in rows)
    print(f"\n  a departure of 4 tokens costs {base:.1f} us/token; the best here is {best:.1f}",
          flush=True)
    print(f"  so one pool read shared across callers is worth up to {base / best:.1f}x",
          flush=True)
    json.dump({"hidden": HIDDEN, "rows": rows, "amortisation": base / best},
              open(sys.argv[2], "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
