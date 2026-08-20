"""Where each arrangement's work lands, at three context lengths.

    python -m sglang.srt.afd.context_budget

One decode layer has four pieces, and only their RATIO changes with context:

    feed-forward     flat. Reading 535 MB of weights, whatever the history is
    q and o          flat
    kv projection    flat
    the sweep        LINEAR in the context, and unbounded

An arrangement is a choice of which pieces go to another machine. Its per-layer wall clock is not
the sum of the two sides but the MAX of them, provided the round trip is covered -- and how much
work stays on the host decides how deep a pipeline has to be to cover it.

So the same four arrangements are ranked at a short context, at half of this model's window, and
at the whole of it. The ranking does not survive the trip, which is the point: an argument for one
of them that does not say at what context length is not an argument.
"""

import json
import sys

import torch
from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

HIDDEN, INTER = 5120, 17408
Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
Q_OUT, KV_OUT = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
LAYERS, FULL_ATTN_LAYERS = 64, 16
ROUND_TRIP_US = 1000.0
CONTEXTS = [1024, 131072, 262144]
NAMES = {1024: "short", 131072: "half full", 262144: "full window"}


def timed(fn, iters=20):
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


def pieces(dev, n, context, weights):
    gate, up, down, w_q, w_o, w_kv = weights
    x = torch.randn(n, HIDDEN, device=dev, dtype=torch.bfloat16)
    a = torch.randn(n, Q_OUT, device=dev, dtype=torch.bfloat16)
    ffn = timed(lambda: (torch.nn.functional.silu(x @ gate.t()) * (x @ up.t())) @ down.t())
    proj = timed(lambda: (x @ w_q.t(), a @ w_o.t()))
    kv = timed(lambda: x @ w_kv.t())

    k_buf = torch.randn(n * context, KV_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    v_buf = torch.randn(n * context, KV_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    q = torch.randn(n, Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    o = torch.empty(n, Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    indptr = torch.arange(n + 1, device=dev, dtype=torch.int64) * context
    indices = torch.arange(n * context, device=dev, dtype=torch.int64)
    S = 8
    logits = torch.empty(n, Q_HEADS, S, HEAD_DIM, device=dev, dtype=torch.float32)
    lse = torch.empty(n, Q_HEADS, S, device=dev, dtype=torch.float32)
    splits = torch.full((n,), S, device=dev, dtype=torch.int32)
    sweep = timed(lambda: decode_attention_fwd(
        q, k_buf, v_buf, o, indptr, indices, logits, lse, splits, S,
        HEAD_DIM**-0.5, 1.0, 1.0), iters=10)
    del k_buf, v_buf, logits, lse
    torch.cuda.empty_cache()
    return {"ffn": ffn, "proj": proj, "kv": kv, "sweep": sweep}


def arrangements(p):
    """host work, and the work of each remote, per layer.

    The sweep is counted on the softmax layers only -- 16 of 64 on this model -- while the
    feed-forward and the projections run on all of them. Charging the sweep to every layer would
    quadruple the one piece that grows, and every ranking below would be a ranking of that error.
    """
    share = FULL_ATTN_LAYERS / LAYERS
    ffn, proj, kv, sweep = p["ffn"], p["proj"], p["kv"], p["sweep"] * share
    return {
        "colocated": (ffn + proj + kv + sweep, []),
        "A  ffn on pool": (proj + kv + sweep, [ffn]),
        "B  ffn + cache on pool": (proj + kv, [ffn + sweep]),
        "two pools": (proj + kv, [ffn, sweep]),
        "E  reversed, sweep remote": (ffn + proj + kv, [sweep]),
    }


def main():
    dev = "cuda"
    weights = (
        torch.randn(INTER, HIDDEN, device=dev, dtype=torch.bfloat16),
        torch.randn(INTER, HIDDEN, device=dev, dtype=torch.bfloat16),
        torch.randn(HIDDEN, INTER, device=dev, dtype=torch.bfloat16),
        torch.randn(Q_OUT, HIDDEN, device=dev, dtype=torch.bfloat16),
        torch.randn(HIDDEN, Q_OUT, device=dev, dtype=torch.bfloat16),
        torch.randn(KV_OUT * 2, HIDDEN, device=dev, dtype=torch.bfloat16),
    )
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out = {}
    for context in CONTEXTS:
        p = pieces(dev, batch, context, weights)
        cache_gb = batch * context * FULL_ATTN_LAYERS * KV_HEADS * HEAD_DIM * 2 * 2 / 1e9
        print(f"\n  {NAMES[context]}, context {context}, batch {batch}: "
              f"ffn {p['ffn']:.0f}us  q+o {p['proj']:.0f}us  kv {p['kv']:.0f}us  "
              f"sweep {p['sweep']:.0f}us", flush=True)
        print(f"    the cache is {cache_gb:.1f} GB of host memory at this point", flush=True)
        arr = arrangements(p)
        base = arr["colocated"][0]
        print(f"    {'arrangement':28} {'host':>8} {'remote':>16} {'ideal':>8} "
              f"{'vs coloc':>9} {'depth':>7}", flush=True)
        rows = {}
        for name, (host, remote) in arr.items():
            ideal = max([host] + remote)
            depth = ROUND_TRIP_US / host if remote else 0.0
            rows[name] = {"host_us": host, "remote_us": remote, "ideal_us": ideal,
                          "speedup": base / ideal, "depth": depth}
            tag = "" if remote else " (no wire)"
            print(f"    {name:28} {host:7.0f}u {str([round(r) for r in remote]):>16} "
                  f"{ideal:7.0f}u {base/ideal:8.2f}x {depth:6.1f}{tag}", flush=True)
        out[context] = {"pieces": p, "cache_gb": cache_gb, "arrangements": rows}
    json.dump(out, open("/home/user/experiment/sglang/afd_context_budget.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
