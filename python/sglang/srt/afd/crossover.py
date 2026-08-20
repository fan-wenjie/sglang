"""The context at which the sweep costs what the feed-forward costs -- the switch threshold.

Two arrangements, and the algebra that separates them is short:

    A  the feed-forward is remote      ideal = max(P + s, F)
    E  the sweep is remote             ideal = max(F + P, s)

with F the feed-forward, P the query/output/key/value projections, s the sweep charged over all
layers. Below s = F the first reads F and the second reads F + P, so A wins by P. Above it the
first reads P + s and the second reads s, so E wins by P. **They cross exactly where s = F**:
whichever piece is bigger is the one that should leave, and nothing else enters it.

So the threshold is not a context length, it is a CONDITION -- and turning it into a context length
needs the sweep measured at the batch size in use. The obvious shortcut, that batch x context is
invariant, is wrong: the sweep gets more efficient as the batch grows, so the same number of
cached tokens costs less at a larger batch and A stays ahead longer.

    python -m sglang.srt.afd.crossover
"""

import json
import sys

import torch
from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

HIDDEN, INTER = 5120, 17408
Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
Q_OUT, KV_OUT = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
SHARE = 16 / 64          # the sweep runs on the softmax layers only
# probes shrink as the batch grows, because the threshold does: probing at 128k with
# batch 64 measures a regime the switch never reaches and fits the line on the wrong end
PROBES = {1: [16384, 65536, 131072], 2: [16384, 65536, 131072],
          4: [16384, 65536, 131072], 8: [16384, 49152, 98304],
          16: [8192, 24576, 49152], 32: [4096, 12288, 24576],
          64: [2048, 8192, 16384]}


def timed(fn, iters=12):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


def sweep_us(dev, batch, context):
    k = torch.randn(batch * context, KV_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    v = torch.randn(batch * context, KV_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    q = torch.randn(batch, Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    o = torch.empty(batch, Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    indptr = torch.arange(batch + 1, device=dev, dtype=torch.int64) * context
    indices = torch.arange(batch * context, device=dev, dtype=torch.int64)
    S = 8
    logits = torch.empty(batch, Q_HEADS, S, HEAD_DIM, device=dev, dtype=torch.float32)
    lse = torch.empty(batch, Q_HEADS, S, device=dev, dtype=torch.float32)
    splits = torch.full((batch,), S, device=dev, dtype=torch.int32)
    out = timed(lambda: decode_attention_fwd(q, k, v, o, indptr, indices, logits, lse,
                                             splits, S, HEAD_DIM**-0.5, 1.0, 1.0), iters=8)
    del k, v, logits, lse
    torch.cuda.empty_cache()
    return out * SHARE


def main():
    dev = "cuda"
    gate = torch.randn(INTER, HIDDEN, device=dev, dtype=torch.bfloat16)
    up = torch.randn(INTER, HIDDEN, device=dev, dtype=torch.bfloat16)
    down = torch.randn(HIDDEN, INTER, device=dev, dtype=torch.bfloat16)
    w_q = torch.randn(Q_OUT, HIDDEN, device=dev, dtype=torch.bfloat16)
    w_o = torch.randn(HIDDEN, Q_OUT, device=dev, dtype=torch.bfloat16)
    w_kv = torch.randn(KV_OUT * 2, HIDDEN, device=dev, dtype=torch.bfloat16)

    rows = []
    print(f"  {'batch':>6} {'ffn':>8} {'proj':>7} {'sweep slope':>14} "
          f"{'THRESHOLD ctx':>14} {'tokens held':>13} {'cache there':>12}", flush=True)
    wanted = [int(b) for b in sys.argv[1:]] or [1, 4, 16, 32, 64]
    for batch in wanted:
        x = torch.randn(batch, HIDDEN, device=dev, dtype=torch.bfloat16)
        a = torch.randn(batch, Q_OUT, device=dev, dtype=torch.bfloat16)
        F = timed(lambda: (torch.nn.functional.silu(x @ gate.t()) * (x @ up.t())) @ down.t())
        P = timed(lambda: (x @ w_q.t(), a @ w_o.t())) + timed(lambda: x @ w_kv.t())
        probes = [(c, sweep_us(dev, batch, c)) for c in PROBES[batch]]
        # a straight line through the two longest probes; the short one carries launch overhead
        (c1, s1), (c2, s2) = probes[-2], probes[-1]
        slope = (s2 - s1) / (c2 - c1)
        intercept = s1 - slope * c1
        threshold = (F - intercept) / slope
        rows.append({"batch": batch, "ffn_us": F, "proj_us": P, "slope_us_per_token": slope,
                     "threshold_context": threshold, "tokens_held": threshold * batch,
                     "probes": probes})
        # 16 softmax layers, 4 kv heads, head_dim 256, key and value, bfloat16
        cache_gb = batch * threshold * 16 * 4 * 256 * 2 * 2 / 1e9
        rows[-1]["cache_gb_at_threshold"] = cache_gb
        print(f"  {batch:6d} {F:7.0f}u {P:6.0f}u {slope*1e3:10.4f}u/kT "
              f"{threshold:14,.0f} {threshold*batch:13,.0f} {cache_gb:10.1f} GB", flush=True)

    print(f"\n  the rule: switch when the sweep reaches the feed-forward. Below it the "
          f"feed-forward should", flush=True)
    print(f"  be the remote piece; above it the sweep should. The threshold in CONTEXT falls as "
          f"the batch", flush=True)
    print(f"  grows and the tokens held RISES, because a bigger batch sweeps the same bytes more "
          f"cheaply --", flush=True)
    print(f"  so a scheduler cannot carry one context number; it has to carry this condition.",
          flush=True)
    json.dump(rows, open("/home/user/experiment/sglang/afd_crossover.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
