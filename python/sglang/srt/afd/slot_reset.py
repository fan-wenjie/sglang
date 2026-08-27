"""Forget the history a starting request inherits, on both ends.

sglang reuses `req_pool_indices`, both slot tables are keyed by that id, and their entries outlive
the request. So the second request through a slot reads the first one's recurrent state and its
convolution ring.

A KV cache survives the same reuse -- a length of zero already excludes stale positions. A
recurrent state has no length: whatever is in the buffer IS the history. The output stays fluent
and is conditioned on somebody else's prompt, which is what "the same as the same as the same as"
turned out to be.

Measured on the deployment, one moved linear layer against the model's own layer on the same
input, same call, same occasion:

    fresh pool, first request     relative 0.0026 mean, 0.012 worst, cosine 1.0000
    second request, same slot     relative 0.31, and the text degenerated

Every in-process check passed throughout, because each ran ONE request.

The signal is a prefill chunk with no cached prefix, and it is taken at the START of a request
rather than at its end: an aborted or crashed request never reaches its end, and the slot it
leaves behind is indistinguishable from one in use.
"""

from __future__ import annotations

import torch

from sglang.srt.afd.protocol import OP_RELEASE


def _tell_the_pool(client, rid: int) -> None:
    """Send the release, and survive a pool that was restarted since the last call.

    Every other call on this path degrades when the pool goes away -- the feed-forward falls back
    to running locally and the router counts it -- and this one did not: a `PoolClosed` raised
    inside a forward reaches sglang as an exception in the model, and the scheduler dies. A pool
    restart then takes the host with it, which is exactly the failure `test_afd_pool_failure.py`
    exists to say cannot happen, reached by a path that file does not cover.

    One reconnect, then a refusal that says what it is. NOT a silent skip: the release is what
    stops the next request inheriting a stale recurrent state, so a host that could not deliver it
    must fail loudly rather than serve fluent text conditioned on somebody else's prompt.
    """
    from sglang.srt.afd.pool_client import PoolClosed

    try:
        handle = client.issue_frame(rid, 0, (torch.zeros(1, 1),), OP_RELEASE)
        client.collect_frame(handle, "cpu")
        return
    except PoolClosed:
        pass
    if not client.reconnect():
        raise RuntimeError(
            f"could not tell the pool to release request {rid}: it is unreachable. Serving on "
            f"without the release would hand this request whatever the last occupant of its slot "
            f"left behind, which stays fluent and reports nothing."
        )
    handle = client.issue_frame(rid, 0, (torch.zeros(1, 1),), OP_RELEASE)
    client.collect_frame(handle, "cpu")


def forget_starting_requests(forward_batch, *, history, client) -> list[int]:
    """Release every row id that BEGINS a request. Returns the ids released.

    `history` is what this end holds and may be None -- an arm that keeps no state here still owes
    the pool the message. `client` is the raw pool client, because this speaks a frame rather than
    a span.

    Call it ONCE a forward pass, before the first routed layer runs. Per layer would clear the
    state the layer before it had just written, which is a different bug with the same name.
    """
    prefix = forward_batch.extend_prefix_lens_cpu
    if prefix is None:
        return []  # a decode step begins nothing
    released = []
    for rid, cached in zip(forward_batch.req_pool_indices, prefix):
        if int(cached) != 0:
            continue  # a later chunk of a prefill already running
        rid = int(rid)
        if history is not None:
            history.forget(rid)
        _tell_the_pool(client, rid)
        released.append(rid)
    return released
