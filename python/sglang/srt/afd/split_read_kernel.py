"""One pass over a linear layer's state, emitting both readings and advancing it.

`linear_history.read` and `.update` are the split written in torch, and they are three to five
times slower than the fused kernel they replace (`benchmark/afd/state_passes.py`):

    batch   torch, separate   torch, combined   the fused kernel
        1           103.3us            94.4us             27.9us
       54           925.0us           833.5us            252.4us

That gap is not a tuning question. At batch 54 the torch split puts the host at 48 x 833us = 40 ms
a decode step against a pool at 44.4 -- the host becomes the bottleneck, and the round-trip cost
the whole arrangement was being judged on stops being the number that decides anything. With the
fused shape it is 12.1 ms.

The reason torch cannot close it: counting memory passes says four to three is a 25% saving, and
measured it saves nothing, because `stack` copies and every operator lands its intermediate in
HBM. The pass count was the wrong model.

## What this kernel does that the pre-split one does not

The kernel it replaces computes `o = S_new q` and never forms `S_old q`. This one emits BOTH
readings of the OLD state -- which is what lets the query's reading be returned one message ahead
of the key and value, and that head start is the whole reason the state was split from the weights
at all.

One program per (row, head). The state tile is 128x128 float32 = 64 KiB and is loaded once into
shared memory, so:

    h_q = S q                   from the tile
    h_k = S k                   from the same tile, no second read
    u   = beta (v - alpha h_k)
    S   = alpha S + u k^T       written back from the tile

which is one read and one write of HBM, against four passes in torch.

## What it does NOT do

Normalise, expand the heads, or compute the gates. Those belong to the side that holds the
weights: `A_log` and `dt_bias` are the layer's own parameters, and a host that computed them would
have to be told which model it is holding. They arrive as per-head scalars.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:                                   # a CPU-only build still imports this module
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _read_two_and_update(
        state_ptr, slot_ptr, q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr,
        h_q_ptr, h_k_ptr,
        stride_s_slot, stride_s_head, stride_s_v,
        stride_x_row, stride_x_head,
        HEADS: tl.constexpr, DV: tl.constexpr, DK: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        slot = tl.load(slot_ptr + row)

        v_off = tl.arange(0, DV)
        k_off = tl.arange(0, DK)
        # (value, key), the state's own layout -- the tile is read once and reused for both
        # readings and for the write-back
        tile = (state_ptr + slot * stride_s_slot + head * stride_s_head
                + v_off[:, None] * stride_s_v + k_off[None, :])
        S = tl.load(tile)

        base = row * stride_x_row + head * stride_x_head
        q = tl.load(q_ptr + base + k_off)
        k = tl.load(k_ptr + base + k_off)
        v = tl.load(v_ptr + base + v_off)
        alpha = tl.load(alpha_ptr + row * HEADS + head)
        beta = tl.load(beta_ptr + row * HEADS + head)

        h_q = tl.sum(S * q[None, :], axis=1)
        h_k = tl.sum(S * k[None, :], axis=1)
        tl.store(h_q_ptr + base + v_off, h_q)
        tl.store(h_k_ptr + base + v_off, h_k)

        u = beta * (v - alpha * h_k)
        tl.store(tile, alpha * S + u[:, None] * k[None, :])


def read_two_and_update(state: torch.Tensor, slots: torch.Tensor, *, q: torch.Tensor,
                        k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor,
                        beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Both readings of the old state, and the state advanced, in one pass.

    `state` is (slots, heads, value dim, key dim) and is written IN PLACE -- the caller's buffer is
    the history, and copying it out to write it back would be the pass this exists to remove.

    `slots` says which row of the state each row of the batch belongs to. It is not optional and
    it is not inferred: a decode batch carries one token from each of several requests, and a row
    advanced against the wrong slot folds one request's history into another with nothing in the
    output to say so.
    """
    if not HAVE_TRITON or not state.is_cuda:
        from sglang.srt.afd.linear_history import read, update

        held = state.index_select(0, slots.long())
        h_q, h_k = read(held, q, k)
        state.index_copy_(
            0, slots.long(), update(held, h_k, v=v, k=k, alpha=alpha, beta=beta))
        return h_q, h_k

    rows, heads, dv, dk = q.shape[0], q.shape[1], v.shape[2], q.shape[2]
    if state.shape[1:] != (heads, dv, dk):
        raise ValueError(
            f"state {tuple(state.shape)} against {rows}x{heads} rows of {dk}-wide query and "
            f"{dv}-wide value; the tile this kernel loads is fixed by the state's own shape and a "
            f"mismatch would read past a head."
        )
    h_q = torch.empty(rows, heads, dv, device=state.device, dtype=state.dtype)
    h_k = torch.empty_like(h_q)
    _read_two_and_update[(rows, heads)](
        state, slots.to(torch.int32), q.contiguous(), k.contiguous(), v.contiguous(),
        alpha.contiguous(), beta.contiguous(), h_q, h_k,
        state.stride(0), state.stride(1), state.stride(2),
        q.stride(0), q.stride(1),
        HEADS=heads, DV=dv, DK=dk,
    )
    return h_q, h_k
