"""What is left on the host when only the sweep leaves, and can it cover the round trip?

The reversed arrangement keeps every weight on the host -- W_o, the MLP, W_q, W_kv -- and sends
only the query to a remote that holds the histories. So the host's work per layer is the colocated
layer MINUS the sweep, and the question is whether a queue of that work covers a database round
trip. If it does, the arrangement costs nothing in time and buys the cache's memory; if it does
not, the difference is the price.
"""
import sys

import torch
from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

HIDDEN, INTER = 5120, 17408
Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
Q_OUT, KV_OUT = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
GROUP = Q_HEADS // KV_HEADS
ROUND_TRIP_MS = 1.0          # measured, this pair of machines, 40 KB frame


def timed(fn, iters=50):
    for _ in range(8):
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
    gate = torch.randn(INTER, HIDDEN, device=dev, dtype=torch.bfloat16)
    up = torch.randn(INTER, HIDDEN, device=dev, dtype=torch.bfloat16)
    down = torch.randn(HIDDEN, INTER, device=dev, dtype=torch.bfloat16)
    w_q = torch.randn(Q_OUT, HIDDEN, device=dev, dtype=torch.bfloat16)
    w_o = torch.randn(HIDDEN, Q_OUT, device=dev, dtype=torch.bfloat16)
    w_kv = torch.randn(KV_OUT * 2, HIDDEN, device=dev, dtype=torch.bfloat16)

    print(f"  {'batch':>6} {'MLP':>8} {'Wq+Wo':>8} {'Wkv':>7} {'host/layer':>11} "
          f"{'sweep@4k':>9} {'depth to cover 1.0 ms':>22}", flush=True)
    for n in (1, 4, 16, 24):
        x = torch.randn(n, HIDDEN, device=dev, dtype=torch.bfloat16)
        a = torch.randn(n, Q_OUT, device=dev, dtype=torch.bfloat16)
        mlp = timed(lambda: (torch.nn.functional.silu(x @ gate.t()) * (x @ up.t())) @ down.t())
        proj = timed(lambda: (x @ w_q.t(), a @ w_o.t()))
        kv = timed(lambda: x @ w_kv.t())

        ctx = 4096
        k_buf = torch.randn(n * ctx, KV_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        v_buf = torch.randn(n * ctx, KV_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        q = torch.randn(n, Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        o = torch.empty(n, Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        indptr = torch.arange(n + 1, device=dev, dtype=torch.int64) * ctx
        indices = torch.arange(n * ctx, device=dev, dtype=torch.int64)
        S = 8
        logits = torch.empty(n, Q_HEADS, S, HEAD_DIM, device=dev, dtype=torch.float32)
        lse = torch.empty(n, Q_HEADS, S, device=dev, dtype=torch.float32)
        splits = torch.full((n,), S, device=dev, dtype=torch.int32)
        sweep = timed(lambda: decode_attention_fwd(
            q, k_buf, v_buf, o, indptr, indices, logits, lse, splits, S,
            HEAD_DIM**-0.5, 1.0, 1.0))
        del k_buf, v_buf, logits
        torch.cuda.empty_cache()

        host = mlp + proj + kv
        depth = ROUND_TRIP_MS * 1e3 / host
        print(f"  {n:6d} {mlp:7.1f}us {proj:7.1f}us {kv:6.1f}us {host:10.1f}us "
              f"{sweep:8.1f}us {depth:21.1f}", flush=True)

    print(f"\n  host/layer is what stays when only the sweep leaves.", flush=True)
    print(f"  depth is how many layers' worth of that work a {ROUND_TRIP_MS} ms round trip needs",
          flush=True)
    print(f"  covering -- and the queue supplies it from OTHER requests at other layers.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
