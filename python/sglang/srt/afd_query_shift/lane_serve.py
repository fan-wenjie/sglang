"""The lane's two serves: the wire ceremony stripped, the arithmetic identical.

`contract_early` and `apply_advance` spend most of their measured cost on ceremony a lane
does not need -- frame parsing, `rows_of`, run checks, host-to-device copies -- because the
lane's receives land in device buffers and the (layer, row) were announced at issue time.
What is left is exactly the kernels: one contraction for the read, one state update and
one ring append for the advance. The arithmetic is the handlers' own, held to it by test.
"""

from __future__ import annotations

import torch


def lane_contract(service, layer_id: int, row: int, q_buf: torch.Tensor):
    """`read_one` against the coefficient, straight from the lane's device buffer."""
    from sglang.srt.afd.linear_history import _row_index
    from sglang.srt.afd.split_read_kernel import read_one

    _, value_heads, _, _ = service.dims
    device = service.cache.state.device
    q_tilde = q_buf.reshape(1, value_heads, -1)
    rows = _row_index([service.cache.slot_of(row)], device)
    reading = read_one(service.cache.state[layer_id], rows, q_tilde)
    service.reads += 1
    return reading.reshape(1, -1).to(torch.bfloat16)


def land_advance(service, layer_id: int, row: int, slab: torch.Tensor) -> None:
    """The packed advance `[k | v | alpha | beta | column]`, split and landed."""
    from sglang.srt.afd.linear_history import _row_index
    from sglang.srt.afd.split_read_kernel import update_only

    _, value_heads, head_k_dim, head_v_dim = service.dims
    device = service.cache.state.device
    k_w = value_heads * head_k_dim
    v_w = value_heads * head_v_dim
    ring = service.cache.conv[layer_id]
    ring_w = ring.shape[-2]
    want = k_w + v_w + 2 * value_heads + ring_w
    if slab.shape[1] != want:
        raise RuntimeError(
            f"a lane advance carried {slab.shape[1]} columns where {want} were expected "
            f"({k_w}+{v_w}+{2 * value_heads}+{ring_w}). The two ends disagree about this "
            f"layer's shapes."
        )
    k, v, alpha, beta, column = torch.split(
        slab, [k_w, v_w, value_heads, value_heads, ring_w], dim=-1
    )
    rows = _row_index([service.cache.slot_of(row)], device)
    update_only(
        service.cache.state[layer_id],
        rows,
        k=k.reshape(1, value_heads, -1),
        v=v.to(service.cache.conv.dtype).reshape(1, value_heads, head_v_dim),
        alpha=alpha,
        beta=beta,
    )
    column = column.to(ring.dtype)
    ring_rows = rows.long()
    held = ring.index_select(0, ring_rows)
    ring.index_copy_(
        0, ring_rows, torch.cat([held[..., 1:], column.unsqueeze(-1)], dim=-1)
    )
    service.updates += 1
