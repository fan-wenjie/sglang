"""Does a linear layer's state have to be float32? The claim, and the measurement.

`linear_state.py` and `linear_history.py` both say float32 "because the state is accumulated across
every step of a generation and a bfloat16 accumulator drifts over thousands of updates in a way a
single step never shows". That was written as a reason, not as a result, and it costs 3.00 MiB a
layer a request against 1.50 -- which is the state read, the fused kernel's traffic, and the host's
whole share of the step time, all doubled.

The claim's shape is worth doubting before its magnitude. The recurrence is

    S <- alpha S + beta (v - alpha S k) k^T          alpha in (0, 1)

which is a FORGETTING accumulator. Old contributions decay geometrically, and so does the error in
them: a perturbation introduced at step t is worth alpha^(n-t) by step n. So the error should reach
a steady state at around 1/(1 - alpha) steps rather than growing with the length of a generation.
If that is what happens, "thousands of updates" is the wrong axis entirely.

Measured against a float64 reference, so neither candidate is its own judge.

    python benchmark/afd/state_precision.py --steps 4000 --heads 8
"""

from __future__ import annotations

import argparse
import sys

import torch


def run(dtype, steps, heads, dk, dv, decay, seed, device):
    """The recurrence, `steps` times, in one precision. Returns the state and every output."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    S = torch.zeros(heads, dv, dk, device=device, dtype=dtype)
    outs = []
    for _ in range(steps):
        q = torch.randn(heads, dk, generator=g).to(device)
        k = torch.randn(heads, dk, generator=g).to(device)
        v = torch.randn(heads, dv, generator=g).to(device)
        b = torch.rand(heads, 1, generator=g).to(device)
        q = torch.nn.functional.normalize(q, dim=-1).to(dtype)
        k = torch.nn.functional.normalize(k, dim=-1).to(dtype)
        v, b = v.to(dtype), b.to(dtype)
        a = torch.full((heads, 1), decay, device=device, dtype=dtype)
        h_k = torch.einsum("hvk,hk->hv", S, k)
        u = b * (v - a * h_k)
        S = a.unsqueeze(-1) * S + u.unsqueeze(-1) * k.unsqueeze(-2)
        outs.append(torch.einsum("hvk,hk->hv", S, q).float())
    return S.float(), torch.stack(outs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  {a.steps} steps, {a.heads} heads of {a.dv}x{a.dk}, against a float64 reference\n")
    print(f"    {'decay':>7} {'memory':>9} {'fp32 err':>11} {'bf16 err':>11} {'bf16/fp32':>10} "
          f"{'out fp32':>10} {'out bf16':>10}")
    for decay in (0.90, 0.99, 0.999, 1.0):
        ref_S, ref_o = run(torch.float64, a.steps, a.heads, a.dk, a.dv, decay, 7, device)
        f32_S, f32_o = run(torch.float32, a.steps, a.heads, a.dk, a.dv, decay, 7, device)
        b16_S, b16_o = run(torch.bfloat16, a.steps, a.heads, a.dk, a.dv, decay, 7, device)
        rel = lambda x, r: ((x - r).norm() / r.norm()).item()
        memory = "unbounded" if decay >= 1.0 else f"{1 / (1 - decay):.0f} steps"
        print(f"    {decay:7.3f} {memory:>9} {rel(f32_S, ref_S):11.3e} "
              f"{rel(b16_S, ref_S):11.3e} {rel(b16_S, ref_S) / max(rel(f32_S, ref_S), 1e-12):9.0f}x "
              f"{rel(f32_o, ref_o):10.3e} {rel(b16_o, ref_o):10.3e}")

    print("\n  the error against step count, at the decay a real gate spends most of its time near:")
    decay = 0.99
    ref_S, _ = run(torch.float64, a.steps, a.heads, a.dk, a.dv, decay, 7, device)
    print(f"    {'steps':>7} {'bf16 err':>11}   (if it were accumulating, this would keep climbing)")
    for n in (10, 100, 500, 1000, 2000, a.steps):
        if n > a.steps:
            continue
        r, _ = run(torch.float64, n, a.heads, a.dk, a.dv, decay, 7, device)
        b, _ = run(torch.bfloat16, n, a.heads, a.dk, a.dv, decay, 7, device)
        print(f"    {n:7} {((b.float() - r.float()).norm() / r.float().norm()).item():11.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
