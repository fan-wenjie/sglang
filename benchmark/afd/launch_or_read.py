"""Is the pool's per-call floor a weight read, or is it launch overhead?

The floor is 0.394 ms for a four-token frame. It was called a weight read on the strength of one
line of arithmetic -- 267 MB at 0.68 TB/s is 0.39 ms -- and that bandwidth was ASSUMED to make the
number fit. This card is an RTX PRO 6000 Blackwell, whose HBM runs near 1.8 TB/s, at which the
same read is 0.15 ms. So the arithmetic did not confirm the explanation; it was fitted to it.

    "The reference must match the thing measured" -- the bandwidth ceiling in this project has
    been wrong three times, always low, and always in the direction that made a reading look
    explained.

This measures instead. For a stack of dense matmuls of known size, with four rows -- the decode
shape, where the weights dominate and the arithmetic is nothing -- it reports:

    per call            wall clock a call, synchronised, which is what a caller waits
    achieved GB/s       bytes of weights divided by that, the effective read rate
    graph replay        the same work through a captured CUDA graph, which removes per-kernel
                        launch overhead and leaves the read

The gap between the two IS the launch overhead, and that is what #44 asks about. If a graph replay
runs at the same speed, the floor is the read and no amount of graph capture helps; if it is much
faster, the floor is launches and the pool should capture one.

    python benchmark/afd/launch_or_read.py [--mib 64 267 512] [--rows 4] [--iters 200]
"""

from __future__ import annotations

import argparse
import time

import torch


def build(mib: int, width: int, dtype=torch.bfloat16):
    """A chain of square matmuls whose weights come to about `mib` megabytes."""
    per = width * width * torch.finfo(dtype).bits // 8
    count = max(1, round(mib * 1024**2 / per))
    weights = [
        torch.randn(width, width, dtype=dtype, device="cuda") for _ in range(count)
    ]
    got = sum(w.numel() * w.element_size() for w in weights)
    return weights, got


def run_once(x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
    for w in weights:
        x = x @ w
    return x


def timed(fn, iters: int) -> float:
    """Seconds a call, synchronised once around the whole run rather than per call.

    Per-call synchronise would measure the synchronise. Around the run it measures what the GPU
    took, which is what a caller waiting on a reply actually pays once the launches are in flight.
    """
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iters


def as_graph(x: torch.Tensor, weights: list[torch.Tensor]):
    """Capture the same chain, so a replay costs one launch instead of one a matmul."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run_once(x, weights)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_once(x, weights)
    return graph


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mib", type=int, nargs="+", default=[64, 267, 512])
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--width", type=int, default=5120)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    print(
        f"  {args.rows} row(s) x {args.width}, bfloat16, {args.iters} iterations a point"
    )
    print(
        f"  {'weights':>9} {'kernels':>8} {'per call':>10} {'GB/s':>8} {'graph':>10} "
        f"{'graph GB/s':>11} {'launch share':>13}"
    )
    for mib in args.mib:
        weights, got = build(mib, args.width)
        x = torch.randn(args.rows, args.width, dtype=torch.bfloat16, device="cuda")

        run_once(x, weights)  # warm the allocator and any autotuning
        plain = timed(lambda: run_once(x, weights), args.iters)
        graph = as_graph(x, weights)
        replay = timed(graph.replay, args.iters)

        gbs = got / plain / 1e9
        graph_gbs = got / replay / 1e9
        share = 100.0 * (plain - replay) / plain
        print(
            f"  {got / 1024 ** 2:>8.0f}M {len(weights):>8} {plain * 1e3:>9.3f}ms {gbs:>8.0f} "
            f"{replay * 1e3:>9.3f}ms {graph_gbs:>11.0f} {share:>12.0f}%"
        )
        # rebound rather than deleted: `weights` is captured by the lambda above, in this same
        # scope, and deleting it is an F821 under the repository's ruff selection. The reference
        # is what holds the memory, so dropping it is what frees it.
        weights = graph = None
        torch.cuda.empty_cache()

    print(
        "\n  The last column is what #44 asks for: the share of a call that is launches rather"
    )
    print(
        "  than reads. A pool whose floor is reads gains nothing from capturing a graph."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
