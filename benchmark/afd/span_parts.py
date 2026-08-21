"""What a span is made of, timed piece by piece against what a round trip costs.

The span's total is measured (`span_cost.py`); its split into feed-forward and linear attention was
arithmetic over weight bytes. That split decides which part of the span can pay for the wire, so it
is measured here rather than divided.

The question it answers: is a linear-attention layer, plus the round trip it would have to carry,
still cheaper than a feed-forward? If it is, the wire can be hidden at linear-attention granularity
and the feed-forward chain never has to be interrupted.

    python benchmark/afd/span_parts.py --config .../config.json --batches 1,4,16 --repeats 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

# Measured on the two-machine link, batch 4: 987 us for a feed-forward round trip, of which
# 359 us was the pool's own work -- so 628 us is wire and protocol. Recorded rather than
# re-measured here: this box has only one card today and a loopback socket would report a
# number about localhost.
ROUND_TRIP_US = 628.0


def timed(fn, repeats: int) -> float:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t) * 1e6)
    return statistics.median(sorted(xs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", required=True)
    ap.add_argument("--repeats", type=int, required=True)
    a = ap.parse_args()
    raw = json.load(open(a.config))
    c = raw.get("text_config", raw)
    H, I = c["hidden_size"], c["intermediate_size"]
    KH, KD = c["linear_num_key_heads"], c["linear_key_head_dim"]
    VH, VD = c["linear_num_value_heads"], c["linear_value_head_dim"]
    dev, dt = torch.device("cuda"), torch.bfloat16
    randn = lambda *s: torch.randn(*s, device=dev, dtype=dt)

    gate, up, down = randn(H, I), randn(H, I), randn(I, H)
    lin_in, lin_out = randn(H, 2 * KH * KD + 2 * VH * VD), randn(VH * VD, H)

    print(f"    {'batch':>6} {'1x FFN':>9} {'1x linear':>11} {'linear+wire':>12} "
          f"{'vs FFN':>8}   {'3x lin+wire':>12} {'vs 4x FFN':>10}")
    for rows in [int(b) for b in a.batches.split(",")]:
        x = randn(rows, H)
        state = torch.zeros(rows, VH, VD, KD, device=dev, dtype=torch.float32)

        def ffn():
            return x + (torch.nn.functional.silu(x @ gate) * (x @ up)) @ down

        def linear():
            qkvz = x @ lin_in
            q = qkvz[:, : KH * KD].reshape(rows, KH, KD).float()
            head = q.repeat_interleave(VH // KH, dim=1)
            out = torch.einsum("bhvk,bhk->bhv", state, head)
            state.add_(torch.einsum("bhv,bhk->bhvk", out, head))
            return x + out.reshape(rows, VH * VD).to(x.dtype) @ lin_out

        f, l = timed(ffn, a.repeats), timed(linear, a.repeats)
        print(f"    {rows:6d} {f:7.0f}us {l:9.0f}us {l + ROUND_TRIP_US:10.0f}us "
              f"{(l + ROUND_TRIP_US) / f:7.2f}x {3 * l + ROUND_TRIP_US:10.0f}us "
              f"{(3 * l + ROUND_TRIP_US) / (4 * f):9.2f}x")

    print(f"\n  a round trip is {ROUND_TRIP_US:.0f} us: wire and protocol, from the 987 us")
    print("  feed-forward call measured on the two-machine link minus its 359 us of pool work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
