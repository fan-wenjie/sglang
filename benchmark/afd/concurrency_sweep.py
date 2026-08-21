"""Does more concurrency hide the pool call, without any new machinery?

The host's GPU idles during a pool round trip because one request stream has no other work: the
sweep it can run early is 37 us against a 900 us socket wait. Two batch overlap would supply that
work, and in this tree it means rewriting a model file, because the operations decomposition it
needs dispatches on layer class name and does not know a hybrid Qwen3.5 stack.

Concurrency supplies the same thing for free. More requests per forward means more attention work
per step against the same 64 round trips, and a wider frame per departure -- and the departure
curve says a frame of 64 tokens costs 81 us/token where a frame of 4 costs 302.

So the sweep is measured before the machinery is written: if raising concurrency closes most of
the gap to colocated, two batch overlap is worth much less than the 1.81x the single-stream
arithmetic suggests.
"""
import json
import sys
import time

from sglang.srt.afd.under_test import model_path

# named once, in the environment, and checked for quantisation before the weights
# load. It was written down separately in eight tools, which is eight chances to
# leave one on the old checkpoint and report its numbers under the new one's name.
MODEL = model_path()
PROMPT = "Write a detailed technical explanation of how a modern GPU schedules work:"
LEVELS = [4, 8, 16, 24]
TOKENS = 64


def arm(label, mem, pool_addr=None):
    import sglang as sgl

    extra = {"afd_mode": "host", "afd_pool_addr": pool_addr} if pool_addr else {}
    engine = sgl.Engine(
        model_path=MODEL, tp_size=1, mem_fraction_static=mem, disable_cuda_graph=True,
        attention_backend="triton", log_level="warning", afd_query_shift_layers=1,
        afd_coverage="all", afd_split_attention=True, **extra)
    out = {}
    try:
        engine.generate([PROMPT], sampling_params={"temperature": 0.0, "max_new_tokens": 8})
        for n in LEVELS:
            params = {"temperature": 0.0, "max_new_tokens": TOKENS}
            started = time.perf_counter()
            outs = engine.generate([PROMPT] * n, sampling_params=params)
            wall = time.perf_counter() - started
            produced = sum(o["meta_info"]["completion_tokens"] for o in outs)
            out[n] = {"tokens": produced, "wall_s": wall, "tokens_per_second": produced / wall}
            print(f"    {label:10s} {n:3d} concurrent: {produced:5d} token(s) in {wall:6.2f}s "
                  f"= {produced / wall:7.1f} tok/s", flush=True)
    finally:
        engine.shutdown()
    return out


def main():
    mem, pool_addr = float(sys.argv[1]), sys.argv[2]
    colocated = arm("colocated", mem)
    two_sided = arm("two-sided", mem, pool_addr)
    print("", flush=True)
    for n in LEVELS:
        ratio = two_sided[n]["tokens_per_second"] / colocated[n]["tokens_per_second"]
        print(f"  {n:3d} concurrent: two-sided is {ratio:.3f}x colocated", flush=True)
    json.dump({"levels": LEVELS, "colocated": colocated, "two_sided": two_sided,
               "ratio": {n: two_sided[n]["tokens_per_second"] / colocated[n]["tokens_per_second"]
                         for n in LEVELS}},
              open("/home/user/experiment/sglang/afd_concurrency.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
