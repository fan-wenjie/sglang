"""The pool now does two kinds of work, and only one of them amortises.

Moving the KV cache to the pool put a non-aggregatable job on the aggregating side. The
feed-forward is weight-shared: one read of a layer's 267 MB serves every caller in the departure,
which is why a departure of 512 tokens costs 5.6x less per token than one of 4. The sweep is not:
each request attends its OWN cache, there are no shared weights to read once, and a departure of
N requests does N independent sweeps.

    pool cost per token  =  feed-forward weight read / N   +   sweep(context)

The first term falls with N. The second does not fall at all, and it GROWS with context. So the
amortisation curve measured before this change described the pool's whole job and now describes
only half of it, and where the crossover sits decides whether a shared pool is still the point.

    python -m sglang.srt.afd.pool_cost_split
"""

import json
import sys

import torch
from sglang.kernels.ops.attention.decode_attention import (
    decode_attention_fwd,
)

HIDDEN, INTERMEDIATE = 5120, 17408
KV_HEADS, Q_HEADS, HEAD_DIM = 4, 24, 256
GROUP = Q_HEADS // KV_HEADS


def timed(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


def main():
    dev = "cuda"
    gate = torch.randn(INTERMEDIATE, HIDDEN, device=dev, dtype=torch.bfloat16)
    up = torch.randn(INTERMEDIATE, HIDDEN, device=dev, dtype=torch.bfloat16)
    down = torch.randn(HIDDEN, INTERMEDIATE, device=dev, dtype=torch.bfloat16)

    def ffn(x):
        return (torch.nn.functional.silu(x @ gate.t()) * (x @ up.t())) @ down.t()

    rows = []
    print(f"  one layer, per departure. feed-forward weights "
          f"{(gate.numel()+up.numel()+down.numel())*2/1e6:.0f} MB bf16\n", flush=True)
    for context in (512, 4096, 16384):
        print(f"  context {context}:", flush=True)
        print(f"    {'requests':>9} {'ffn':>9} {'sweeps':>9} {'ffn/token':>10} "
              f"{'sweep/token':>12}", flush=True)
        for n in (1, 4, 16, 64):
            x = torch.randn(n, HIDDEN, device=dev, dtype=torch.bfloat16)
            # sglang's own decode kernel, ragged over the batch, so the grouped-query expansion is
            # never materialised. An einsum with repeat_interleave expands 4 key heads to 24 in
            # memory -- six times the traffic -- and measures that instead of the architecture.
            k_buf = torch.randn(n * context, KV_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
            v_buf = torch.randn(n * context, KV_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
            q = torch.randn(n, Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
            o = torch.empty(n, Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
            kv_indptr = torch.arange(n + 1, device=dev, dtype=torch.int64) * context
            kv_indices = torch.arange(n * context, device=dev, dtype=torch.int64)
            SPLITS = 8
            logits = torch.empty(n, Q_HEADS, SPLITS, HEAD_DIM, device=dev, dtype=torch.float32)
            lse = torch.empty(n, Q_HEADS, SPLITS, device=dev, dtype=torch.float32)
            splits = torch.full((n,), SPLITS, device=dev, dtype=torch.int32)

            def sweeps():
                decode_attention_fwd(
                    q, k_buf, v_buf, o, kv_indptr, kv_indices, logits, lse, splits,
                    SPLITS, HEAD_DIM**-0.5, 1.0, 1.0,
                )

            t_ffn = timed(lambda: ffn(x))
            t_sweep = timed(sweeps, iters=20)
            rows.append({"context": context, "requests": n, "ffn_us": t_ffn,
                         "sweep_us": t_sweep, "ffn_per_token_us": t_ffn / n,
                         "sweep_per_token_us": t_sweep / n})
            print(f"    {n:9d} {t_ffn:8.1f}us {t_sweep:8.1f}us {t_ffn/n:9.1f}us "
                  f"{t_sweep/n:11.1f}us", flush=True)
            del k_buf, v_buf, logits
            torch.cuda.empty_cache()
        print("", flush=True)

    print("  what amortises and what does not, from 1 request to 64:", flush=True)
    for context in (512, 4096, 16384):
        one = next(r for r in rows if r["context"] == context and r["requests"] == 1)
        many = next(r for r in rows if r["context"] == context and r["requests"] == 64)
        print(f"    context {context:6d}: feed-forward "
              f"{one['ffn_per_token_us']/many['ffn_per_token_us']:5.1f}x   sweep "
              f"{one['sweep_per_token_us']/many['sweep_per_token_us']:5.2f}x   "
              f"sweep is {100*many['sweep_per_token_us']/(many['sweep_per_token_us']+many['ffn_per_token_us']):.0f}%"
              f" of the pool's per-token cost at 64 requests", flush=True)
    json.dump(rows, open("/home/user/experiment/sglang/afd_pool_cost_split.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
