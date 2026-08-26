"""The host's half of the push arrangement: contract and answer, apply and say nothing.

The pool cooks -- see `pool_cook`, and the ladder that put every preparatory step over there --
and this side does only what the state's owner alone can do:

    OP_STATE_EARLY   contract the state with the cooked `q~` and REPLY with the reading, so the
                     answer crosses back inside the feed-forward the pool is spending
    OP_STATE_APPLY   advance the state and the ring with what the pool sends after assembly;
                     nothing is answered, and the next EARLY for the same layer is behind it on
                     the same socket, so the order is the socket's

Registered rather than written into `afd.history_service`, so that deleting this package removes
the behaviour instead of leaving a handler nothing reaches.

THE RULE, stated where it is enforced: no handler in this file may convolve, normalise, or form
a coefficient. Any of those here is preparation on the bottleneck card, which is the exact
failure the ladder priced; `test_afd_early_frame.TheHostHalfDoesNotCook` holds this file to it,
and the runtime tripwire in `span_routing` catches whatever an edit sneaks past the test.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.afd.history_service import register_op
from sglang.srt.afd.protocol import INBOUND_OPS, OP_STATE_APPLY, OP_STATE_EARLY

logger = logging.getLogger(__name__)


def contract_early(service, frame):
    """Contract the state with the pool's cooked coefficient and answer with the reading.

    One tensor down (`q~`, float32 as `normalise` leaves it, so the wire adds no rounding),
    one tensor back. The query itself never crosses any more: `s = beta (k . q)` is assembled
    on the pool, which kept the query it cooked.
    """
    from sglang.srt.afd.linear_history import _row_index
    from sglang.srt.afd.slots import _runs
    from sglang.srt.afd.split_read_kernel import read_one

    if len(frame.tensors) != 1:
        raise RuntimeError(
            f"an early read carried {len(frame.tensors)} tensor(s) where the cooked q~ alone "
            f"was expected. The far end is running an older protocol; contracting with part "
            f"of its payload would read the wrong state."
        )
    (q_tilde,) = frame.tensors
    _, value_heads, _, head_v_dim = service.dims
    device = service.cache.state.device
    q_tilde = q_tilde.to(device).float().reshape(q_tilde.shape[0], value_heads, -1)
    ids = [int(r) for r in service.rows_of(frame)]
    if any(n != 1 for _, _, n in _runs(ids)):
        raise RuntimeError(
            f"an early read for layer {frame.layer} carried a run of more than one row. A "
            f"chunk's tokens read each other's advance, so its read cannot be separated from "
            f"its update; the caller was supposed to refuse this bus."
        )
    rows = _row_index([service.cache.slot_of(r) for r in ids], device)
    reading = read_one(service.cache.state[frame.layer], rows, q_tilde)
    service.reads += 1
    # bfloat16 on the wire: half the reply's bytes on the loop whose lateness is the one
    # blocking cost left. The reading feeds the OUTPUT's assembly and never the state, so the
    # rounding is spent in one step -- the same reassociation regime as the fused cook.
    return (reading.reshape(q_tilde.shape[0], -1).to(torch.bfloat16),)


def apply_advance(service, frame):
    """Advance the state and the ring with the pool's finished step. Nothing is answered.

    The frame carries ONE tensor: `[k | v | alpha | beta | column]` packed flat in float32,
    because the serve was measured at 0.879 ms of mostly per-tensor machinery -- five
    host-to-device copies and their enqueues where one suffices. float32 round-trips every
    part exactly, so the packing changes no arithmetic. The parts: this step's key --
    convolved, normalised, expanded on the pool -- the value, the two gates, and the ring's
    next column, whose q channels are the raw query projection: the operator has one query
    and that is it. The widths are the service's own dims and the ring's channel count, so
    nothing about the layout crosses the wire.
    Applied here and now rather than parked: reordering the inbound path was tried three
    ways and each lost to this plain order (the design record has the flights); the single
    worker serving in arrival order is both the correctness argument and the fastest
    arrangement measured.
    """
    from sglang.srt.afd.linear_history import _row_index
    from sglang.srt.afd.slots import _runs
    from sglang.srt.afd.split_read_kernel import update_only

    if len(frame.tensors) != 1:
        raise RuntimeError(
            f"a state apply carried {len(frame.tensors)} tensor(s) where one packed "
            f"[k | v | alpha | beta | column] was expected. The far end runs an older "
            f"protocol; applying part of its payload would advance the wrong state."
        )
    if service._parked is not None:
        raise RuntimeError(
            f"an APPLY for layer {frame.layer} reached this history with layer "
            f"{service._parked[0]}'s advance still parked. The caller of this service is not "
            f"draining between frames, so the two advances could land out of order."
        )
    (packed,) = frame.tensors
    _, value_heads, head_k_dim, head_v_dim = service.dims
    device = service.cache.state.device
    packed = packed.to(device).float()
    rows_n = packed.shape[0]
    k_w = value_heads * head_k_dim
    v_w = value_heads * head_v_dim
    ring_w = service.cache.conv.shape[-2]
    want = k_w + v_w + 2 * value_heads + ring_w
    if packed.shape[1] != want:
        raise RuntimeError(
            f"a packed apply carried {packed.shape[1]} columns where "
            f"{want} were expected ({k_w}+{v_w}+{2 * value_heads}+{ring_w}). The two ends "
            f"disagree about this layer's shapes."
        )
    k, v, alpha, beta, column = torch.split(
        packed, [k_w, v_w, value_heads, value_heads, ring_w], dim=-1
    )
    k = k.reshape(rows_n, value_heads, -1)
    v = v.to(service.cache.conv.dtype).reshape(rows_n, value_heads, head_v_dim)
    ids = [int(r) for r in service.rows_of(frame)]
    if any(n != 1 for _, _, n in _runs(ids)):
        raise RuntimeError(
            f"a state apply for layer {frame.layer} carried a run of more than one row. A "
            f"chunk goes as OP_STATE_SCAN, which carries its own materials; the caller was "
            f"supposed to refuse this bus."
        )
    rows = _row_index([service.cache.slot_of(r) for r in ids], device)
    update_only(
        service.cache.state[frame.layer], rows, k=k, v=v, alpha=alpha, beta=beta
    )
    # The ring's advance: shift left, append. The same write `convolve_with_ring` performed
    # when the convolution ran here, against the same slots.
    ring = service.cache.conv[frame.layer]
    column = column.to(device=device, dtype=ring.dtype)
    ring_rows = rows.long()
    held = ring.index_select(0, ring_rows)
    ring.index_copy_(
        0, ring_rows, torch.cat([held[..., 1:], column.unsqueeze(-1)], dim=-1)
    )
    service.updates += 1


register_op(OP_STATE_EARLY, contract_early)
register_op(OP_STATE_APPLY, apply_advance)
# membership travels with the handlers: the router may only route what something answers
INBOUND_OPS.update({OP_STATE_EARLY, OP_STATE_APPLY})
