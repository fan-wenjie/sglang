"""How many times a linear layer's state has to be touched, and what each pass costs.

Splitting the read from the update turned one pass over the state into four: read with the query,
read with the key, then read and write to advance it. The state is 3 MiB a layer a request and the
whole step is memory-bound, so the pass count IS the cost.

Three ways to do the same arithmetic:

    separate    two einsums and an update          4 passes
    combined    one einsum against [q|k], update   3 passes
    fused       the kernel that was there before   1 read + 1 write, and no split

The fused one is measured as the lower bound. What the gap between `combined` and `fused` is worth
decides whether a triton kernel that holds the state tile in shared memory is worth writing:
128x128 float32 is 64 KiB, which fits.

    python benchmark/afd/state_passes.py --batches 1,4,16 --repeats 200
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch


def timed(fn, repeats):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t) * 1e6)
    return statistics.median(sorted(xs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", required=True)
    ap.add_argument("--repeats", type=int, required=True)
    ap.add_argument("--heads", type=int, default=48)
    ap.add_argument("--dim", type=int, default=128)
    a = ap.parse_args()
    dev = torch.device("cuda")
    VH, D = a.heads, a.dim

    print(f"    {'batch':>6} {'separate':>10} {'combined':>10} {'fused kernel':>13} "
          f"{'MiB of state':>13} {'combined vs fused':>18}")
    for rows in [int(b) for b in a.batches.split(",")]:
        S = torch.randn(rows, VH, D, D, device=dev, dtype=torch.float32) * 0.1
        qh = torch.randn(rows, VH, D, device=dev, dtype=torch.float32)
        kh = torch.randn(rows, VH, D, device=dev, dtype=torch.float32)
        v = torch.randn(rows, VH, D, device=dev, dtype=torch.float32)
        alpha = torch.rand(rows, VH, device=dev) * 0.5 + 0.5
        beta = torch.rand(rows, VH, device=dev)
        al, be = alpha.unsqueeze(-1), beta.unsqueeze(-1)

        def separate():
            h_q = torch.einsum("bhvk,bhk->bhv", S, qh)
            h_k = torch.einsum("bhvk,bhk->bhv", S, kh)
            u = be * (v - al * h_k)
            return h_q, al.unsqueeze(-1) * S + u.unsqueeze(-1) * kh.unsqueeze(-2)

        def combined():
            qk = torch.stack([qh, kh], dim=-1)                  # [rows, VH, D, 2]
            h = torch.einsum("bhvk,bhkt->bhvt", S, qk)          # ONE read of the state
            h_q, h_k = h[..., 0], h[..., 1]
            u = be * (v - al * h_k)
            return h_q, al.unsqueeze(-1) * S + u.unsqueeze(-1) * kh.unsqueeze(-2)

        want_q, want_S = separate()
        got_q, got_S = combined()
        torch.testing.assert_close(got_q, want_q, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(got_S, want_S, rtol=1e-4, atol=1e-5)

        sep, comb = timed(separate, a.repeats), timed(combined, a.repeats)
        fused = _fused_us(rows, VH, D, a.repeats, dev)
        mib = S.numel() * 4 / 1024 ** 2
        gap = f"{comb / fused:.2f}x" if fused else "n/a"
        print(f"    {rows:6d} {sep:8.1f}us {comb:8.1f}us "
              f"{(f'{fused:.1f}us' if fused else 'unavailable'):>13} {mib:12.2f} {gap:>18}")

    print("\n  The fused column is the kernel this split replaced, timed on the same shapes. It")
    print("  does not produce the two readings separately, so it is a floor rather than an option.")
    return 0


def _fused_us(rows, VH, D, repeats, dev):
    """The pre-split kernel on the same shapes, as a floor. None if this build has no such kernel."""
    try:
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )
    except Exception:
        return None
    KH = VH
    mixed = torch.randn(rows, 2 * KH * D + VH * D, device=dev, dtype=torch.bfloat16)
    a_ = torch.randn(rows, VH, device=dev, dtype=torch.float32)
    b_ = torch.randn(rows, VH, device=dev, dtype=torch.float32)
    A_log = torch.randn(VH, device=dev, dtype=torch.float32)
    dt = torch.randn(VH, device=dev, dtype=torch.float32)
    state = torch.zeros(rows, VH, D, D, device=dev, dtype=torch.float32)
    out = torch.empty(rows, 1, VH, D, device=dev, dtype=torch.bfloat16)
    idx = torch.arange(rows, device=dev, dtype=torch.int32)

    def run():
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed, a=a_, b=b_, A_log=A_log, dt_bias=dt, scale=D**-0.5,
            initial_state=state, out=out, ssm_state_indices=idx,
            use_qk_l2norm_in_kernel=True,
        )

    try:
        return timed(run, repeats)
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
