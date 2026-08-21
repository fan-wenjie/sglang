"""Is the model that decodes the model whose perplexity was reported?

bpb and ppl were measured by scoring input logprobs -- a prefill, colocated, with the attention
fused. Three things that a served token passes through are absent from that measurement:

    the partition   sweep and join instead of one attention call. Refused in prefill
    the schedule    the query projected a layer early, inside the pool round trip
    the pool        every dense feed-forward computed in another process on another machine

A fault in any of them would leave the reported perplexity untouched and the served text wrong.
So the same two numbers are taken again through the whole arrangement, and the same prompts are
decoded greedily on both sides.

    bpb equal    the arrangement computes the model the perplexity belongs to
    text equal   and computes it the same way token by token, which bpb cannot check because
                 scoring never takes the decode path
"""
import json
import math
import sys
import time

import numpy as np

from sglang.srt.afd.under_test import model_path

# named once, in the environment, and checked for quantisation before the weights
# load. It was written down separately in eight tools, which is eight chances to
# leave one on the old checkpoint and report its numbers under the new one's name.
MODEL = model_path()
TOKENS = "/home/user/experiment/v6/run/tokens/val_Qwen_Qwen3.8-27B.u32"
BYTES_PER_TOKEN = 4.3775
PROMPTS = ["The capital of France is",
           "Explain in one sentence why the sky is blue:",
           "List the first eight prime numbers, separated by commas:",
           "Write a two-line poem about a slow train:"]


def draw(sequences=24, seq=1024, seed=0):
    tokens = np.fromfile(TOKENS, dtype=np.uint32)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(tokens) - seq - 1, size=sequences)
    return [tokens[s: s + seq].astype(np.int64).tolist() for s in starts]


def arm(label, prompts_ids, mem, pool_addr=None):
    import sglang as sgl

    extra = {}
    if pool_addr:
        extra = {"afd_mode": "host", "afd_pool_addr": pool_addr}
    engine = sgl.Engine(
        model_path=MODEL, tp_size=1, mem_fraction_static=mem, disable_cuda_graph=True,
        attention_backend="triton", log_level="warning", afd_query_shift_layers=1,
        afd_coverage="all", afd_split_attention=True, **extra)
    try:
        outs = engine.generate(input_ids=prompts_ids,
                               sampling_params={"temperature": 0.0, "max_new_tokens": 1},
                               return_logprob=True, logprob_start_len=0)
        total, counted = 0.0, 0
        for out in outs:
            for entry in out["meta_info"]["input_token_logprobs"]:
                if entry[0] is None:
                    continue
                total += float(entry[0])
                counted += 1
        nats = -total / counted
        started = time.perf_counter()
        text = [o["text"] for o in engine.generate(
            PROMPTS, sampling_params={"temperature": 0.0, "max_new_tokens": 64})]
        elapsed = time.perf_counter() - started
        metrics = {"positions": counted, "nats_per_token": nats, "ppl": math.exp(nats),
                   "bpb": nats / (BYTES_PER_TOKEN * math.log(2)), "decode_s": elapsed}
        print(f"  {label:12s} bpb {metrics['bpb']:.6f}  ppl {metrics['ppl']:.4f}  "
              f"({counted} positions)", flush=True)
        return metrics, text
    finally:
        engine.shutdown()


def main():
    mem, pool_addr = float(sys.argv[1]), sys.argv[2]
    ids = draw()
    colocated, text_a = arm("colocated", ids, mem)
    two_sided, text_b = arm("two-sided", ids, mem, pool_addr)

    d_bpb = two_sided["bpb"] - colocated["bpb"]
    print(f"\n  bpb delta pool vs colocated: {d_bpb:+.3e} "
          f"({100 * d_bpb / colocated['bpb']:+.5f}%)", flush=True)
    same = [a == b for a, b in zip(text_a, text_b)]
    for i, ok in enumerate(same):
        print(f"    prompt {i}: {'same' if ok else 'DIFFERS'}   {text_a[i][:50]!r}", flush=True)
    print(f"  {sum(same)}/{len(same)} greedy continuations identical", flush=True)
    recorded = 0.73239909
    print(f"\n  recorded shift-1 bpb from the three-arm run: {recorded:.6f}", flush=True)
    print(f"  colocated here {colocated['bpb']:.6f}, through the pool {two_sided['bpb']:.6f}",
          flush=True)
    json.dump({"colocated": colocated, "two_sided": two_sided, "recorded_bpb": recorded,
               "identical": same, "text_colocated": text_a, "text_two_sided": text_b},
              open("/home/user/experiment/sglang/afd_decode_matches_ppl.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
