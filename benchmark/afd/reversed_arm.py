"""The reversed arrangement, end to end: the feed-forward stays, only the sweep travels.

    python -m sglang.srt.afd.reversed_arm --cache HOST:PORT --contexts 1024,8192,32768 \\
        --requests 8 --mem 0.78 --out reversed.json

Every number this tree has published about arrangement E came from arithmetic over a cost model.
The table in AFD_FINDINGS section 9 ranks it first at long context, at 1.78x, with a pipeline depth
requirement of 2.4 against arrangement A's 15.7 -- and E had never been run, because the only door
into host mode required a weights pool address and installed the feed-forward router on the way
through. This is the first measurement.

## What is on each machine

    host    every weight. The query and output projections, the key/value projection, the whole
            feed-forward. It computes; it holds no history.
    pool    the histories, and the sweep over them. No weights at all -- the pool process reads
            the config for its geometry and never opens a checkpoint.

The host still computes this step's own token and merges it with what the pool returns. That join
is attention over one position, which is the value itself, so what crosses the wire is a query out
and an (output, log partition) pair back.

## The control is the SAME model, and that took a wrong measurement to learn

Three arms, because two of the three differences between "stock" and "E" are not the one being
measured:

    stock          shift 0, fused         the model as shipped
    local split    shift 1, partitioned   the read point moved; every layer still local
    E              shift 1, pooled        the read point moved AND the sweep is remote

E was first compared against stock, and its tokens differed -- "The capital of France is Paris.
The capital of France is Paris." against stock's "...Germany is Berlin. ...Italy is Rome." That
looked like a broken pool for the better part of a debugging session. It was not. Shift 1 on a
checkpoint nothing has repaired IS a different model, loudly so, and the launch warns about it.
Against the local split arm, the pooled arm is token-identical.

So the ratio that means anything here is E over LOCAL SPLIT: what moving the sweep to another
machine costs, with the read point held fixed on both sides. Stock is kept as context, because the
read point's own cost is worth seeing next to it, but it is not the denominator.

## Tokens before timings

Every arm generates the same prompts greedily first and E's output must match the local split's
exactly. A pool that refuses a connection, a layer that falls back, a prefill that never mirrors
its history -- each produces a working server whose numbers are the local numbers with extra
steps, and each also changes the tokens. Checking them is cheaper than any of the counters that
would otherwise have to catch it, and this file previously promised a counter check it did not do.
"""

from __future__ import annotations

import argparse
import json
import socket
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
# short, factual, and greedy: what is compared is whether two arms agree, so a prompt whose answer
# wanders is a prompt that fails the check for reasons that are not about the arms
AGREEMENT_PROMPTS = [
    "The capital of France is",
    "List the first eight prime numbers, separated by commas:",
    "What is the derivative of x squared with respect to x?",
]


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
    """Two generations differing only in length; their difference is decode.

    Warmed at the longest shape first and the timed pair discarded once, for the reason
    `context_arms` records: warming only at the short length let the first timed run absorb the
    compile of the long one, and the arm reported four steps taking longer than thirty-six.
    """
    warm_long, _ = _timed(engine, prompt, requests, LONG)
    _timed(engine, prompt, requests, SHORT)
    _timed(engine, prompt, requests, SHORT)
    _timed(engine, prompt, requests, LONG)

    t_short, n_short = _timed(engine, prompt, requests, SHORT)
    t_long, n_long = _timed(engine, prompt, requests, LONG)
    steps, seconds = n_long - n_short, t_long - t_short
    if steps <= 0 or seconds <= 0:
        raise SystemExit(
            f"  {n_short} and {n_long} token(s) in {t_short:.2f}s and {t_long:.2f}s. A longer "
            f"generation that took less time is not a longer version of the first, and no rate "
            f"can be read off it. Warm-up at {LONG} tokens took {warm_long:.2f}s."
        )
    per_step_ms = seconds / (LONG - SHORT) * 1e3
    if not 1.0 < per_step_ms < 20_000.0:
        raise SystemExit(f"  {per_step_ms:.1f} ms a step is outside anything a step can be")
    return {"tokens": steps, "seconds": seconds, "tokens_per_second": steps / seconds,
            "ms_per_step": per_step_ms}


def sweeps_on_the_pool(address: str) -> dict:
    """Ask the pool what it did, so a run that swept nothing cannot report a speed.

    HELLO is the only op that answers without touching a history, so it is what a caller uses to
    reach a pool it has no request in flight with.
    """
    import torch
    from sglang.srt.afd.protocol import Frame, decode, send_frame, OP_HELLO

    host, port = address.split(":")
    with socket.create_connection((host, int(port)), timeout=15) as sock:
        send_frame(sock, Frame(request_id=1, layer=0, tensors=[torch.zeros(1, 1)], op=OP_HELLO))
        reply = decode(sock)
    return {"capabilities": reply.tensors[0].flatten().tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--requests", type=int, required=True)
    ap.add_argument("--mem", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    contexts = [int(c) for c in a.contexts.split(",")]

    import sglang as sgl
    from transformers import AutoTokenizer

    print(f"  cache pool at {a.cache}: {sweeps_on_the_pool(a.cache)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = {c: prompt_of(tokenizer, c) for c in contexts}
    print(f"  prompts {[len(tokenizer(p)['input_ids']) for p in prompts.values()]}, "
          f"{a.requests} request(s) each", flush=True)

    shifted = {"afd_query_shift_layers": 1, "afd_coverage": "all", "afd_split_attention": True}
    out = {"config": vars(a), "arms": {}, "tokens": {}}
    arms = (
        ("stock", {}),
        ("local split", dict(shifted)),
        ("E sweep on pool", {"afd_mode": "host", "afd_cache_addr": a.cache, **shifted}),
    )
    for label, kw in arms:
        print(f"    {label}: loading", flush=True)
        engine = sgl.Engine(model_path=MODEL, tp_size=1, mem_fraction_static=a.mem,
                            disable_cuda_graph=True, attention_backend="triton",
                            log_level="warning", **kw)
        try:
            out["tokens"][label] = [
                o["text"] for o in engine.generate(
                    AGREEMENT_PROMPTS,
                    sampling_params={"temperature": 0.0, "max_new_tokens": 24})
            ]
            for context in contexts:
                row = decode_rate(engine, prompts[context], a.requests)
                out["arms"].setdefault(label, {})[str(context)] = row
                print(f"    {label:16} context {context:6d}: "
                      f"{row['tokens_per_second']:7.1f} tok/s "
                      f"({row['ms_per_step']:7.1f} ms a step)", flush=True)
        finally:
            engine.shutdown()

    control, pooled = out["tokens"]["local split"], out["tokens"]["E sweep on pool"]
    if control != pooled:
        for prompt, a_text, b_text in zip(AGREEMENT_PROMPTS, control, pooled):
            if a_text != b_text:
                print(f"  DIFFERENT TOKENS for {prompt!r}\n    local {a_text[:80]!r}\n"
                      f"    pool  {b_text[:80]!r}", flush=True)
        raise SystemExit(
            "  the pooled arm and the local split arm produced different tokens. They are the "
            "same model -- same read point, same coverage, same partition -- so a difference is "
            "the pool computing something else, and no timing taken alongside it means anything."
        )
    print(f"\n  tokens identical across {len(AGREEMENT_PROMPTS)} prompt(s): "
          f"local split == E", flush=True)

    print(f"\n  {'context':>8} {'stock':>9} {'local':>9} {'E':>9} "
          f"{'E / local':>11} {'local / stock':>14}", flush=True)
    for context in contexts:
        st = out["arms"]["stock"][str(context)]["tokens_per_second"]
        lo = out["arms"]["local split"][str(context)]["tokens_per_second"]
        e = out["arms"]["E sweep on pool"][str(context)]["tokens_per_second"]
        print(f"  {context:8d} {st:8.1f} {lo:8.1f} {e:8.1f} {e / lo:10.3f}x {lo / st:13.3f}x",
              flush=True)
    print("  E / local is what moving the sweep costs; local / stock is what moving the read "
          "point costs, and only the first is this measurement's subject", flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
