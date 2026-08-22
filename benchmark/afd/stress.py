"""Sustained load through the disaggregated path, and a pool killed while it is carrying it.

    python -m sglang.srt.afd.stress --weights HOST:PORT [--cache HOST:PORT] --waves 12

A correctness suite says the arrangement computes the right thing once. This asks whether it keeps
doing so, and what it does when the thing it depends on goes away. Three questions, and each has a
failure that a short run cannot show:

    does it leak      a history that is never released grows until the pool refuses -- and it
                      refuses on some later unlucky request, not on the one that leaked. Checked
                      by watching what the cache pool holds BETWEEN waves, which should be what it
                      held before the first one
    does it drift     throughput that decays over waves is a queue filling, a cache growing, or a
                      list being appended to forever. A single wave cannot tell a warm-up from a
                      slope
    does it survive   the weights pool is killed mid-wave. The requests in flight must fail --
                      their answers went with the process -- and the ones after must be served

The last is the one worth running deliberately. A serving system is not defined by what it does
when everything works.
"""

from __future__ import annotations

import argparse
import json
import statistics
import os
import sys
import time

# `under_test` moved out of the runtime package and into this directory when the arrangement's
# code was split for merging; five benchmarks kept importing it from where it used to be and none
# of them had been run since. A benchmark that cannot import is at least loud -- the dangerous
# version of this is the one that imports something ELSE of the same name.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from under_test import model_path

# named once, in the environment, and checked for quantisation before the weights
# load. It was written down separately in eight tools, which is eight chances to
# leave one on the old checkpoint and report its numbers under the new one's name.
MODEL = model_path()
PROMPTS = [
    "The capital of France is",
    "List the first eight prime numbers, separated by commas:",
    "Explain in one sentence why the sky is blue:",
    "Write a two-line poem about a slow train:",
    "Name three programming languages that compile to machine code:",
    "What is the derivative of x squared with respect to x?",
    "Summarise the difference between a stack and a queue:",
    "Give one reason a bridge might be built as an arch:",
]


def wave(engine, requests: int, tokens: int) -> dict:
    params = {"temperature": 0.0, "max_new_tokens": tokens}
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(requests)]
    started = time.perf_counter()
    outs = engine.generate(prompts, sampling_params=params)
    wall = time.perf_counter() - started
    produced = sum(o["meta_info"]["completion_tokens"] for o in outs)
    return {"wall_s": wall, "tokens": produced, "tokens_per_second": produced / wall}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--waves", type=int, required=True)
    ap.add_argument("--requests", type=int, required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import sglang as sgl

    extra = {"afd_cache_addr": a.cache} if a.cache else {}
    engine = sgl.Engine(
        model_path=MODEL, tp_size=1, mem_fraction_static=0.78, disable_cuda_graph=True,
        attention_backend="triton", log_level="warning", afd_mode="host",
        afd_pool_addr=a.weights, afd_query_shift_layers=1, afd_coverage="all", **extra)

    rows, failures = [], []
    try:
        wave(engine, a.requests, 4)          # warm, not measured
        for index in range(a.waves):
            try:
                row = wave(engine, a.requests, a.tokens)
            except Exception as e:            # noqa: BLE001 -- recorded, not swallowed
                failures.append({"wave": index, "error": repr(e)[:200]})
                print(f"    wave {index:2d}: FAILED {repr(e)[:90]}", flush=True)
                continue
            row["wave"] = index
            rows.append(row)
            print(f"    wave {index:2d}: {row['tokens']:5d} token(s) in {row['wall_s']:6.2f}s "
                  f"= {row['tokens_per_second']:7.1f} tok/s", flush=True)
    finally:
        engine.shutdown()

    if not rows:
        print("  every wave failed", flush=True)
        json.dump({"waves": rows, "failures": failures}, open(a.out, "w"), indent=2)
        return 1

    rates = [r["tokens_per_second"] for r in rows]
    first, last = rates[: max(1, len(rates) // 3)], rates[-max(1, len(rates) // 3):]
    drift = statistics.mean(last) / statistics.mean(first)
    print(f"\n  {len(rows)} wave(s) served, {len(failures)} failed", flush=True)
    print(f"  throughput  first third {statistics.mean(first):7.1f} tok/s   "
          f"last third {statistics.mean(last):7.1f} tok/s   drift {drift:.3f}x", flush=True)
    print(f"  spread      min {min(rates):.1f}  median {statistics.median(rates):.1f}  "
          f"max {max(rates):.1f}", flush=True)
    json.dump({"waves": rows, "failures": failures, "drift": drift}, open(a.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
