"""What a model's shape has to be for moving its feed-forward to another machine to pay.

    python -m sglang.srt.afd.compute_comm_ratio --batch 8 --wire-gbit 4.4 --rtt-us 300

The arrangement wins when the pool finishes the feed-forward before the host finishes the work it
kept, round trip included. So the quantity that decides everything is one ratio per layer:

    compute      the FFN's weights read once out of HBM       3 * hidden * intermediate * w / BW
    communicate  the hidden state sent and the result returned    2 * batch * hidden * h / wire + RTT

## The hidden dimension cancels, which is the part that is not obvious

Widening a model widens both terms. The FFN's weight matrices grow with `hidden`, and so does the
hidden state on the wire, at the same rate. Divide:

    compute / communicate  ~  (3 * intermediate * w / BW) / (2 * batch * h / wire)

`hidden` is gone. A model twice as wide is not a better candidate for this arrangement; it is the
same candidate. What actually helps is `intermediate` in absolute terms -- a FATTER feed-forward
relative to nothing at all, not relative to the model -- and a smaller batch.

This is why "optimise for big-weight models" is the right instinct and the wrong parameter. Big
usually means wide, and wide does not move this ratio. Deep does not either: every layer pays its
own round trip. Only the feed-forward's inner dimension does.

## Mixture-of-experts moves it the wrong way

An MoE layer holds enormous weights and reads only the activated experts per token, so its compute
term is the ACTIVATED intermediate, not the total. The hidden state on the wire is unchanged. A
sparse model is therefore a worse candidate than a dense one of the same footprint, which is the
opposite of what "big weights" suggests.

## What the numbers are

`--wire-gbit` and `--rtt-us` describe the interconnect and have no defaults: the whole point is
that they decide the answer, and a default would let someone read a conclusion off a link they
never described. Measured on this pair of machines they are 4.4 Gbit/s and 300 us, over a virtual
veth with no RDMA anywhere.
"""

from __future__ import annotations

import argparse
import sys

# (name, hidden, intermediate, layers, note). Dense feed-forward widths, read from public configs.
MODELS = [
    ("Qwen3.8-27B", 5120, 17408, 64, "this study's model"),
    ("Llama-3.2-1B", 2048, 8192, 16, "the second family tested"),
    ("Llama-3.1-8B", 4096, 14336, 32, ""),
    ("Llama-3.1-70B", 8192, 28672, 80, ""),
    ("Llama-3.1-405B", 16384, 53248, 126, "the widest dense one shipped"),
    ("Qwen2.5-72B", 8192, 29568, 80, ""),
]


def ratio(hidden: int, intermediate: int, batch: int, weight_bytes: float,
          hidden_bytes: int, hbm_gbps: float, wire_gbit: float, rtt_us: float) -> dict:
    """One layer's compute time against one layer's wire time, both in microseconds."""
    ffn_bytes = 3 * hidden * intermediate * weight_bytes
    compute_us = ffn_bytes / (hbm_gbps * 1e9) * 1e6
    payload_bits = batch * hidden * hidden_bytes * 8
    wire_us = 2 * payload_bits / (wire_gbit * 1e9) * 1e6 + rtt_us
    return {"ffn_mib": ffn_bytes / 1024 ** 2, "compute_us": compute_us, "wire_us": wire_us,
            "ratio": compute_us / wire_us}


def needed_wire(hidden: int, intermediate: int, batch: int, weight_bytes: float,
                hidden_bytes: int, hbm_gbps: float, target: float, rtt_us: float) -> float:
    """The bandwidth at which this model reaches `target` compute-to-wire, at the given RTT.

    Returns infinity when the RTT alone already exceeds the budget -- which is the common case on a
    link measured in hundreds of microseconds, and is the finding rather than an edge case.
    """
    compute_us = 3 * hidden * intermediate * weight_bytes / (hbm_gbps * 1e9) * 1e6
    budget_us = compute_us / target
    if budget_us <= rtt_us:
        return float("inf")
    payload_bits = batch * hidden * hidden_bytes * 8
    return 2 * payload_bits / ((budget_us - rtt_us) * 1e-6) / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--wire-gbit", type=float, required=True)
    ap.add_argument("--rtt-us", type=float, required=True)
    ap.add_argument("--hbm-gbps", type=float, required=True)
    ap.add_argument("--weight-bytes", type=float, required=True,
                    help="1 for FP8, 2 for bf16; the FFN weights as the pool stores them")
    ap.add_argument("--hidden-bytes", type=int, required=True,
                    help="bytes per element of the hidden state on the wire")
    ap.add_argument("--target", type=float, required=True,
                    help="the compute-to-wire ratio at which the arrangement is worth running")
    a = ap.parse_args()

    print(f"  batch {a.batch}, {a.wire_gbit} Gbit/s, {a.rtt_us:.0f}us RTT, "
          f"{a.hbm_gbps:.0f} GB/s HBM, FFN weights {a.weight_bytes}B, wire {a.hidden_bytes}B\n")
    print(f"    {'model':16} {'inter':>7} {'FFN/layer':>10} {'compute':>9} {'wire':>8} "
          f"{'ratio':>7} {'Gbit/s for ' + str(a.target) + 'x':>16}")
    for name, hidden, intermediate, _layers, _note in MODELS:
        r = ratio(hidden, intermediate, a.batch, a.weight_bytes, a.hidden_bytes,
                  a.hbm_gbps, a.wire_gbit, a.rtt_us)
        need = needed_wire(hidden, intermediate, a.batch, a.weight_bytes, a.hidden_bytes,
                           a.hbm_gbps, a.target, a.rtt_us)
        need_text = "RTT alone too big" if need == float("inf") else f"{need:.0f}"
        print(f"    {name:16} {intermediate:7d} {r['ffn_mib']:8.0f}MiB {r['compute_us']:7.0f}us "
              f"{r['wire_us']:6.0f}us {r['ratio']:6.2f}x {need_text:>16}")

    print(f"\n  The 'Gbit/s' column holds the RTT fixed at {a.rtt_us:.0f}us. Where it says the RTT")
    print(f"  alone is too big, no bandwidth reaches {a.target}x: the round trip's fixed cost has")
    print(f"  already spent the whole budget, and the fix is a lower-latency fabric, not a fatter")
    print(f"  one. Widening a model does not appear in this table because it cannot: the hidden")
    print(f"  dimension divides out of the ratio, leaving the feed-forward's inner width and the")
    print(f"  batch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
