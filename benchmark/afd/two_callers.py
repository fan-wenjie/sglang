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


def one_caller(pool: str, layer: int, width: int, tokens: int, rounds: int, out: list) -> None:
    """A caller's own connection, its own frames, its own timings."""
    client = PoolClient(pool, connect_timeout_s=10)
    hidden = torch.zeros(tokens, width, dtype=torch.bfloat16)
    try:
        for _ in range(3):                       # warm the connection and any JIT behind it
            client.collect(client.issue(1, layer, hidden), "cpu")
        for _ in range(rounds):
            start = time.perf_counter()
            client.collect(client.issue(1, layer, hidden), "cpu")
            out.append((time.perf_counter() - start) * 1e3)
    finally:
        client.close()


def run(pool: str, callers: int, layer: int, width: int, tokens: int, rounds: int) -> dict:
    """All callers issue at once, which is the case the departure has to batch."""
    timings: list[list] = [[] for _ in range(callers)]
    threads = [
        threading.Thread(target=one_caller,
                         args=(pool, layer, width, tokens, rounds, timings[i]))
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
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--rounds", type=int, default=40)
    args = p.parse_args()

    print(f"  pool {args.pool}, layer {args.layer}, {args.tokens} token(s) x {args.width} a call")
    print(f"  {'callers':>8} {'median ms':>10} {'worst ms':>9} {'calls/s':>9} {'per caller':>11}")
    first = None
    for callers in args.callers:
        got = run(args.pool, callers, args.layer, args.width, args.tokens, args.rounds)
        if first is None:
            first = got["round_trip_ms"]
        print(f"  {got['callers']:>8} {got['round_trip_ms']:>10.2f} {got['worst_ms']:>9.2f} "
              f"{got['calls_per_s']:>9.1f} {got['round_trip_ms'] / first:>10.2f}x")

    print("\n  A second caller that rides the first one's weight read costs almost nothing: the")
    print("  median round trip stays flat and calls/s roughly doubles. A second caller that waits")
    print("  its turn shows the median rising with the caller count and calls/s flat -- the pool")
    print("  is serialising them, which is what min_batch=1 asks it to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
