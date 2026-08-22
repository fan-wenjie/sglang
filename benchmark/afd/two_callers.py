"""Two independent callers on one pool: does the second one ride the first one's weight read?

This is the property the whole arrangement is built on, measured rather than argued. The pool
holds no per-request state, so a call from caller A and a call from caller B are the same kind of
transaction and can leave on the same bus -- and if they do, the layer's 267 MB weight read is
paid once for both. That is what makes re-forming across CALLERS possible at all, and re-forming
across callers is what re-forming across NODES is.

`pool_amortisation.py` answers the neighbouring question -- one caller sending wider frames -- and
says so explicitly. It cannot answer this one: a wider frame shares the weight read by
construction, whereas two callers share it only if the departure actually batches them, which is a
scheduling question and not an arithmetic one.

    python benchmark/afd/two_callers.py --pool 127.0.0.1:8999 [--callers 1 2 4] [--layer 0]

Each caller is its own socket and its own connection, which is exactly what a second host is to
this pool. It is NOT a second model host: that needs a second card, and two hosts on one card
contend for that card's bandwidth, so a flat aggregate could not tell a saturated pool from a
saturated host. What this measures is the pool's side.

## Read it against the pool's own settings

`--afd-min-batch 1` means a departure leaves the moment the first frame arrives, so two callers
share a bus only when the second's frame is already queued when the first departs. A pool deployed
for latency at min_batch 1 will show riders of 1 and no sharing at all -- which is a real reading
about that configuration, not a failure of the idea, and it is why this prints the settings it was
run against beside the numbers.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

import torch

from sglang.srt.afd.pool_client import PoolClient


def one_caller(pool: str, layer: int, width: int, tokens: int, rounds: int, out: list,
               spread: int = 1) -> None:
    """A caller's own connection, its own frames, its own timings.

    `spread` is how many DIFFERENT layers the calls cycle through. It exists to separate two
    explanations of the same per-call floor: a true HBM read of the layer's weights every time,
    which cannot be helped, or a cache effect that hammering one layer hides -- if 64 layers cost
    what 1 layer costs, the read is real; if cycling is slower, a pool holding fewer layers keeps
    more of them warm and sharding buys something the arithmetic did not predict.
    """
    client = PoolClient(pool, connect_timeout_s=10)
    hidden = torch.zeros(tokens, width, dtype=torch.bfloat16)
    try:
        for _ in range(3):                       # warm the connection and any JIT behind it
            client.collect(client.issue(1, layer, hidden), "cpu")
        for i in range(rounds):
            at = layer + (i % spread)
            start = time.perf_counter()
            client.collect(client.issue(1, at, hidden), "cpu")
            out.append((time.perf_counter() - start) * 1e3)
    finally:
        client.close()


def run(pool: str, callers: int, layer: int, width: int, tokens: int, rounds: int,
        spread: int = 1) -> dict:
    """All callers issue at once, which is the case the departure has to batch."""
    timings: list[list] = [[] for _ in range(callers)]
    threads = [
        threading.Thread(target=one_caller,
                         args=(pool, layer, width, tokens, rounds, timings[i], spread))
        for i in range(callers)
    ]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start
    every = [ms for row in timings for ms in row]
    return {
        "callers": callers,
        "round_trip_ms": statistics.median(every),
        "worst_ms": max(every),
        "calls": len(every),
        "calls_per_s": len(every) / wall,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--callers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--width", type=int, default=5120)
    p.add_argument("--tokens", type=int, nargs="+", default=[4])
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--spread", type=int, default=1,
                   help="how many different layers the calls cycle through")
    args = p.parse_args()

    print(f"  pool {args.pool}, layer {args.layer}, width {args.width}")
    print(f"  {'tokens':>7} {'callers':>8} {'median ms':>10} {'calls/s':>9} {'tokens/s':>10} "
          f"{'vs 1 caller':>12}")
    for tokens in args.tokens:
        first = None
        for callers in args.callers:
            got = run(args.pool, callers, args.layer, args.width, tokens, args.rounds,
                      args.spread)
            if first is None:
                first = got["calls_per_s"]
            print(f"  {tokens:>7} {got['callers']:>8} {got['round_trip_ms']:>10.2f} "
                  f"{got['calls_per_s']:>9.1f} {got['calls_per_s'] * tokens:>10.0f} "
                  f"{got['calls_per_s'] / first:>11.2f}x")

    print("\n  The question is the last column at the WIDEST frame. A second caller that rides")
    print("  the first one's weight read shows tokens/s near 2x; one that waits its turn shows it")
    print("  flat. At 4 tokens a call the read is not what binds the pool, so the comparison only")
    print("  means something where the read dominates -- which is what the widths are for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
