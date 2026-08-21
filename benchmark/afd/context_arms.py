"""Decode throughput against context, for the arms that keep their cache on the host.

The budget says an arrangement's standing changes with context, because three of a layer's four
pieces are flat and the sweep is linear. At 1k every arm is slower than colocated -- the sweep is
41 us and no arrangement can pay for a round trip with it -- and the prediction is that the gap
closes as the history grows.

    python -m sglang.srt.afd.context_arms --weights HOST:PORT --contexts 1024,8192,32768

## Measuring decode and not prefill

A long prompt's prefill dwarfs its decode, so timing one generate measures the prefill. The prompt
is therefore warmed first -- which puts it in the radix cache -- and then two generates are timed
that differ only in how many tokens they produce. Their difference is decode.

Two sanity checks guard the arithmetic, and both have caught a real fault. An early version got 958
tokens a second and a NEGATIVE interval, because it warmed inside the timing rather than before it.
A later one warmed only at the short length, and the A arm's first timed run took 33.5s for 4 steps
against 8.4s for 36 -- the pool path paying to compile the long shape during the measurement. So
the warm-up now runs at the longest shape, the timed pair is discarded once, the longer run must
take longer, and the implied step time must be within an order of magnitude of what a step can be.
A harness that cannot fail its own sanity check is a harness that reported 958.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from sglang.srt.afd.under_test import model_path

# named once, in the environment, and checked for quantisation before the weights
# load. It was written down separately in eight tools, which is eight chances to
# leave one on the old checkpoint and report its numbers under the new one's name.
MODEL = model_path()
FILLER = ("The scheduler assigns each warp a slot in the issue queue, and the memory controller "
          "coalesces the resulting requests into cache lines before they reach the crossbar. ")
SHORT, LONG = 4, 36


def prompt_of(tokenizer, target: int) -> str:
    text = FILLER
    while len(tokenizer(text)["input_ids"]) < target:
        text += FILLER
    return tokenizer.decode(tokenizer(text)["input_ids"][:target])


def _timed(engine, prompt: str, requests: int, tokens: int) -> tuple:
    params = {"temperature": 0.0, "max_new_tokens": tokens}
    started = time.perf_counter()
    outputs = engine.generate([prompt] * requests, sampling_params=params)
    return (time.perf_counter() - started,
            sum(o["meta_info"]["completion_tokens"] for o in outputs))


def decode_rate(engine, prompt: str, requests: int) -> dict:
    """Two generates differing only in length; their difference is decode.

    Warmed at LONG before SHORT, and the timed pair discarded once. An earlier version warmed at
    SHORT alone, and the A arm's first timed run then took 33.5s for 4 steps against 8.4s for 36 --
    the longer generate finishing sooner, which tripped the check below. Four steps do not warm the
    kernels thirty-six steps use, and on the pool path the first use of a shape pays for compiling
    it. Warming at the largest shape covers every shape the timing then visits.

    The warm-up durations are returned rather than dropped: an arm whose warm-up is still an order
    of magnitude above its timed runs has not finished warming, and that should be readable in the
    record instead of showing up as a rate nobody can explain.
    """
    warm_long, _ = _timed(engine, prompt, requests, LONG)
    warm_short, _ = _timed(engine, prompt, requests, SHORT)
    _timed(engine, prompt, requests, SHORT)
    _timed(engine, prompt, requests, LONG)

    t_short, n_short = _timed(engine, prompt, requests, SHORT)
    t_long, n_long = _timed(engine, prompt, requests, LONG)

    steps, seconds = n_long - n_short, t_long - t_short
    if steps <= 0 or seconds <= 0:
        raise SystemExit(
            f"  the two runs produced {n_short} and {n_long} token(s) in {t_short:.2f}s and "
            f"{t_long:.2f}s. A longer generation that took less time, or produced no more tokens, "
            f"means the second run was not a longer version of the first -- most likely its "
            f"output was already cached -- and any rate from it is a rate for work that did not "
            f"happen. Warm-up took {warm_long:.2f}s at {LONG} tokens and {warm_short:.2f}s at "
            f"{SHORT}; a warm-up far above the timed runs means it never finished warming."
        )
    per_step_ms = seconds / ((LONG - SHORT)) * 1e3
    if not 1.0 < per_step_ms < 20_000.0:
        raise SystemExit(f"  {per_step_ms:.1f} ms a step is outside anything a step can be")
    return {"tokens": steps, "seconds": seconds, "tokens_per_second": steps / seconds,
            "ms_per_step": per_step_ms,
            "warmup_seconds": {"long": warm_long, "short": warm_short},
            "timed_seconds": {"long": t_long, "short": t_short}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--requests", type=int, required=True)
    ap.add_argument("--mem", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    contexts = [int(c) for c in a.contexts.split(",")]

    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = {c: prompt_of(tokenizer, c) for c in contexts}
    print(f"  prompts: {[len(tokenizer(p)['input_ids']) for p in prompts.values()]} tokens, "
          f"{a.requests} request(s) each", flush=True)

    out = {}
    for label, kw in (("stock", {}),
                      ("A ffn on pool", {"afd_mode": "host", "afd_pool_addr": a.weights,
                                         "afd_query_shift_layers": 1, "afd_coverage": "all"})):
        engine = sgl.Engine(model_path=MODEL, tp_size=1, mem_fraction_static=a.mem,
                            disable_cuda_graph=True, attention_backend="triton",
                            log_level="warning", **kw)
        try:
            for context in contexts:
                row = decode_rate(engine, prompts[context], a.requests)
                out.setdefault(label, {})[context] = row
                print(f"    {label:14s} context {context:6d}: {row['tokens_per_second']:7.1f} "
                      f"tok/s  ({row['ms_per_step']:7.1f} ms a step)", flush=True)
        finally:
            engine.shutdown()

    print(f"\n  {'context':>8} {'stock':>10} {'A':>10} {'A / stock':>10}", flush=True)
    for context in contexts:
        s = out["stock"][context]["tokens_per_second"]
        p = out["A ffn on pool"][context]["tokens_per_second"]
        print(f"  {context:8d} {s:9.1f} {p:9.1f} {p / s:9.3f}x", flush=True)
    json.dump(out, open(a.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
