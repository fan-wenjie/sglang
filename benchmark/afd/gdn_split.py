"""Is a linear attention's history read separable from its state update? Checked against the kernel.

`linear_state.py` says the two cannot be separated -- that a gated delta rule's read needs the
update, so its history cannot be fetched ahead of this step's key and value the way a softmax
sweep's can. That is true of the FUSED KERNEL. Whether it is true of the RECURRENCE is a different
question, and it decides whether a linear layer can wear the same interface as a softmax one.

The claim under test, for a state `S` held from the previous step:

    h_q = S q                                   a read, with the query alone
    h_k = S k                                   a read, with the key alone
    o   = alpha h_q + beta (v - alpha h_k)(k.q) both reads, mixed with scalars

If that reproduces the kernel elementwise, then a linear layer's host side is what a softmax
layer's host side already is: hold a history, read it, return the reading. The mixing, the weights
and the gates stay with the weights.

Derived from the published gated delta rule, NOT read out of sglang's kernel -- which is compiled.
So it is checked numerically before anything is built on it.

    python benchmark/afd/gdn_split.py
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    torch.manual_seed(0)
    dev = torch.device("cuda")
    rows, KH, VH, KD, VD = 3, 2, 4, 8, 8
    slots = 4
    scale = KD ** -0.5

    q = torch.randn(rows, KH, KD, device=dev, dtype=torch.bfloat16)
    k = torch.randn(rows, KH, KD, device=dev, dtype=torch.bfloat16)
    v = torch.randn(rows, VH, VD, device=dev, dtype=torch.bfloat16)
    mixed = torch.cat([q.reshape(rows, -1), k.reshape(rows, -1), v.reshape(rows, -1)], dim=-1)
    a = torch.randn(rows, VH, device=dev, dtype=torch.float32)
    b = torch.randn(rows, VH, device=dev, dtype=torch.float32)
    A_log = torch.randn(VH, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(VH, device=dev, dtype=torch.float32)

    state = torch.randn(slots, VH, VD, KD, device=dev, dtype=torch.float32) * 0.1
    kept = state.clone()
    indices = torch.arange(rows, device=dev, dtype=torch.int32)
    out = torch.empty(rows, 1, VH, VD, device=dev, dtype=torch.bfloat16)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale,
        initial_state=state, out=out, ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    kernel_out = out.reshape(rows, VH, VD).float()
    kernel_state = state[:rows].clone()

    # the gates, as the fused sigmoid-gating form defines them
    alpha = torch.exp(-torch.exp(A_log) * torch.nn.functional.softplus(a + dt_bias))
    beta = torch.sigmoid(b)

    qn = torch.nn.functional.normalize(q.float(), dim=-1) * scale
    kn = torch.nn.functional.normalize(k.float(), dim=-1)
    rep = VH // KH
    qh = qn.repeat_interleave(rep, dim=1)          # [rows, VH, KD]
    kh = kn.repeat_interleave(rep, dim=1)

    S = kept[:rows]                                 # [rows, VH, VD, KD], the OLD state
    h_q = torch.einsum("bhvk,bhk->bhv", S, qh)      # a read, with the query alone
    h_k = torch.einsum("bhvk,bhk->bhv", S, kh)      # a read, with the key alone
    kq = (kh * qh).sum(-1)                          # [rows, VH]
    al, be = alpha.unsqueeze(-1), beta.unsqueeze(-1)
    split_out = al * h_q + be * (v.float() - al * h_k) * kq.unsqueeze(-1)
    split_state = al.unsqueeze(-1) * S + (
        be * (v.float() - al * h_k)).unsqueeze(-1) * kh.unsqueeze(-2)

    def report(name, got, want):
        err = (got - want).abs().max().item()
        rel = err / max(want.abs().max().item(), 1e-9)
        print(f"    {name:34} max abs {err:.3e}   relative {rel:.3e}   "
              f"{'MATCH' if rel < 2e-2 else 'DIFFERS'}")
        return rel < 2e-2

    print("  the expansion against the kernel, elementwise:\n")
    ok_out = report("output o_t", split_out, kernel_out)
    ok_state = report("updated state S_t", split_state, kernel_state)

    print("\n  what it means if both match: a linear layer's host side is a HISTORY READ, the")
    print("  same shape as a softmax layer's -- two readings returned, the mixing done where the")
    print("  weights are. The window then exists at every layer, not at one in four.")
    if not (ok_out and ok_state):
        print("\n  NOT MATCHED. The interface cannot be unified on this derivation; do not build")
        print("  on it. Either the gate definition or the recurrence differs from the published")
        print("  form, and the difference has to be found before the claim is used.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
