"""What one span actually costs on the pool, measured rather than added up.

The span is the pool's whole job under the group cut: from a softmax layer's attention output,
through that layer's output projection and feed-forward, through three linear layers entire, to
the hidden state the next softmax attention reads. The batch is formed once at its start and is
fixed until it ends, so the span's duration is the pool's scheduling quantum.

Its cost was first arrived at by adding up the weight reads -- 1683 us at batch 4 -- and that sum
is a lower bound on a card that never launches a kernel. A span is 4 feed-forwards and 3 linear
attentions, which is on the order of forty kernels, and forty launches at the wrong end of a
decode step are not free. Measured here before anything is built on the number.

Synthetic weights at the real shapes: every stage is a weight read, so what matters is the bytes
and the launch count, not the values.

    python benchmark/afd/span_cost.py --batches 1,4,8,16 --repeats 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch


def build(config, device, dtype, max_rows):
    H, I = config["hidden_size"], config["intermediate_size"]
    QH, KVH, HD = (config["num_attention_heads"], config["num_key_value_heads"],
                   config["head_dim"])
    KH, KD = config["linear_num_key_heads"], config["linear_key_head_dim"]
    VH, VD = config["linear_num_value_heads"], config["linear_value_head_dim"]
    randn = lambda *s: torch.randn(*s, device=device, dtype=dtype)
    return {
        "W_o": randn(QH * HD, H),
        "ffn": [(randn(H, I), randn(H, I), randn(I, H)) for _ in range(4)],
        "lin_in": [randn(H, 2 * KH * KD + 2 * VH * VD) for _ in range(3)],
        "lin_out": [randn(VH * VD, H) for _ in range(3)],
        "state": torch.zeros(max_rows, VH, VD, KD, device=device, dtype=torch.float32),
        "shape": (H, I, QH, KVH, HD, KH, KD, VH, VD),
    }


def span(w, o, rows):
    """W_o, then four feed-forwards with three linear attentions between them."""
    H, I, QH, KVH, HD, KH, KD, VH, VD = w["shape"]
    x = o @ w["W_o"]
    for i in range(4):
        gate, up, down = w["ffn"][i]
        x = x + (torch.nn.functional.silu(x @ gate) * (x @ up)) @ down
        if i == 3:
            break
        qkvz = x @ w["lin_in"][i]
        q = qkvz[:, : KH * KD].reshape(rows, KH, KD).float()
        # the recurrent read and update, at the real shapes. The delta rule's own kernel is a
        # fused triton one; this is the same memory traffic in plain torch, which is what the
        # stage costs on a card where it is a weight-and-state read
        st = w["state"][:rows]
        head = q.repeat_interleave(VH // KH, dim=1)
        out = torch.einsum("bhvk,bhk->bhv", st, head)
        st.add_(torch.einsum("bhv,bhk->bhvk", out, head))
        x = x + out.reshape(rows, VH * VD).to(x.dtype) @ w["lin_out"][i]
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", required=True)
    ap.add_argument("--repeats", type=int, required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    raw = json.load(open(a.config))
    config = raw.get("text_config", raw)
    device, dtype = torch.device("cuda"), torch.bfloat16
    rows_wanted = [int(b) for b in a.batches.split(",")]
    w = build(config, device, dtype, max(rows_wanted))
    H, I, QH, KVH, HD, KH, KD, VH, VD = w["shape"]

    weights = (QH * HD * H + 4 * 3 * H * I
               + 3 * (H * (2 * KH * KD + 2 * VH * VD) + VH * VD * H)) * 2
    print(f"  a span reads {weights / 1024 ** 2:.0f} MiB of weights; the sum of those reads at "
          f"1792 GB/s is {weights / 1792e9 * 1e6:.0f} us\n")
    print(f"    {'batch':>6} {'measured':>10} {'sum of reads':>14} {'launch overhead':>17} "
          f"{'us a row':>9}")
    out = {}
    for rows in rows_wanted:
        o = torch.randn(rows, QH * HD, device=device, dtype=dtype)
        for _ in range(20):
            span(w, o, rows)
        torch.cuda.synchronize()
        xs = []
        for _ in range(a.repeats):
            t = time.perf_counter()
            span(w, o, rows)
            torch.cuda.synchronize()
            xs.append((time.perf_counter() - t) * 1e6)
        us = statistics.median(sorted(xs))
        floor = weights / 1792e9 * 1e6
        out[rows] = us
        print(f"    {rows:6d} {us:8.0f}us {floor:12.0f}us {us - floor:15.0f}us {us / rows:7.1f}us")

    print("\n  the span is the pool's scheduling quantum: a caller that misses one waits a whole")
    print("  span for the next, and the batch riding it cannot change until it ends.")
    if a.out:
        json.dump({"weights_bytes": weights, "spans_us": out}, open(a.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
