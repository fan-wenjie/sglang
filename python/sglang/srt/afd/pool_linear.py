"""One linear-attention layer a call, served on the pool with the history left on the caller.

Standard AFD's, not the derived line's. Splitting a linear layer at the recurrence needs no moved
read point -- `linear_runner` carries the argument and the arithmetic -- so this is the pool's half
of it: take a batch of rows, install the callbacks that fetch the caller's recurrent state, run the
layer, send back the attention output alone.

With this a pool holds EVERY weight, feed-forwards and linear projections alike, and the host holds
every piece of state. That is the split by state taken to its conclusion.

The callbacks are here rather than in the group cut's module because a span uses them too and the
dependency has to run one way: the derived package imports this, never the other way round.

## Why these never depart on the connection thread

Every op served here calls BACK to the caller mid-run. A departure taken on the thread that owns
the caller's socket would then wait for a message only that thread could read. That is declared in
the same call that claims the op, and the reading when it was not was

    no state reading for request 3 layer 0 within 30.0s

which names the far end and blames it, while the far end had answered immediately.
"""

from __future__ import annotations

import logging
import time

import torch

from sglang.srt.afd.pool_server import HostDeparted, register_departure
from sglang.srt.afd.slots import _runs
from sglang.srt.afd.protocol import (
    OP_STATE_APPLY,
    OP_STATE_EARLY,
    OP_STATE_MIX,
    OP_STATE_READ,
    OP_STATE_SCAN,
    Frame,
    send_frame,
)
from sglang.srt.afd.stream_sender import streamed_send

logger = logging.getLogger(__name__)

# How long a departure waits for the caller to answer with the recurrent state it holds. Named for
# what it times rather than for the group cut, which is what it was called when the span was the
# only thing that waited. The group cut has a second wait of its own, for a different thing, and
# one constant serving both means tuning either one silently moves the other.
from sglang.srt.afd.pool_server import STATE_READING_TIMEOUT_S  # noqa: F401 -- one home


def _readings_or_zeros(departure, pending):
    """Every rider's reading, with a departed host's rider zero-filled.

    `_await_reading` answers None the moment a host is known gone. A mixed departure
    zero-fills that rider -- shaped like a living rider's reading, which always exists in a
    mixed departure -- and serves the rest on time; a departure with nobody left raises
    `HostDeparted` and is abandoned whole, because there is nobody to reply to either.
    """
    out = [_await_reading(departure, sock, rid, lid) for sock, rid, lid in pending]
    if all(r is None for r in out):
        raise HostDeparted(
            f"all {len(out)} rider(s) of this departure belong to departed host(s)"
        )
    if any(r is None for r in out):
        alive = next(r for r in out if r is not None)
        out = [torch.zeros_like(alive) if r is None else r for r in out]
    return out


def _install_history_calls(departure, riding, counts) -> None:
    """Give this span the two calls it makes back to whoever is holding the history.

    A span's riders can come from SEVERAL sockets -- that is the point of a departure -- so a
    state read is not one message but one per rider, each carrying that rider's own rows. They
    are sent together and collected together, so the wait is one round trip rather than one
    per rider.

    Set per THREAD on the runner. Several groups can be departing at once and an instance
    attribute would hand one span's sockets to another's rows.
    """

    def ask_host(layer_id, request_ids, q_tilde, step=None):
        """Ask each rider's host for its reading. One message a rider, all sent before any
        is collected, so the wait is one round trip rather than one per rider.

        A rider carrying MORE THAN ONE row is a prefill chunk: its rows are one request's own
        tokens and they are sequentially dependent, so the reading cannot be separated from
        the update -- token n reads what token n-1 advanced. Those go as OP_STATE_SCAN with
        everything a step needs. A single row is a decode row and keeps the split, which is
        the one contraction the query coefficient bought.
        """
        pending = []
        offset = 0
        for (frame, sock), n in zip(riding, counts):
            sl = slice(offset, offset + n)
            offset += n
            if n == 1 or step is None:
                body = (q_tilde[sl].reshape(n, -1),)
                op = OP_STATE_READ
            else:
                k, v, alpha, beta = step
                body = (
                    q_tilde[sl].reshape(n, -1),
                    k[sl].reshape(n, -1),
                    v[sl].reshape(n, -1),
                    alpha[sl],
                    beta[sl],
                )
                op = OP_STATE_SCAN
            send_frame(sock, Frame(frame.request_id, layer_id, body, op))
            with departure._cond:
                departure._sent_at[(id(sock), frame.request_id, layer_id)] = (
                    time.perf_counter()
                )
            pending.append((sock, frame.request_id, layer_id))
        return collect_host((pending, q_tilde.shape[:2]))

    def issue_host(layer_id, request_ids, q_tilde):
        """Send every rider's cooked coefficient and return the handles for its reply.

        The far end contracts and answers with the reading STRAIGHT AWAY, inside the
        feed-forward this caller is about to spend -- so when `collect_early` asks, the answer
        is usually already on the table and the round trip has left the critical path. That is
        the push arrangement's whole mechanism.

        Decode rows only. A rider with more than one row is a prefill chunk whose tokens read
        each other's advance; `ask_host` sends those as OP_STATE_SCAN, which carries both.
        """
        offset = 0
        pending = []
        for (frame, sock), n in zip(riding, counts):
            sl = slice(offset, offset + n)
            offset += n
            if any(run != 1 for _, _, run in _runs(request_ids[sl])):
                # a RUN of one request's rows is a chunk, whose tokens read each other's
                # advance -- that read cannot be issued before its update exists. Several
                # DISTINCT requests' decode rows in one rider are fine: each reads its own
                # state, which is exactly what the batched contraction serves. The first
                # guard here rejected by row count and killed the pool's departure the
                # first time three concurrent requests shared a decode batch.
                raise RuntimeError(
                    f"a split read for layer {layer_id} reached a rider whose ids repeat "
                    f"consecutively -- a chunk. The caller was supposed to refuse this bus."
                )
            body = (q_tilde[sl].reshape(n, -1),)
            if not streamed_send(
                sock, frame.request_id, layer_id, OP_STATE_EARLY, body
            ):
                departure.post_unawaited(
                    sock, frame.request_id, layer_id, body, OP_STATE_EARLY
                )
            with departure._cond:
                departure._sent_at[(id(sock), frame.request_id, layer_id)] = (
                    time.perf_counter()
                )
            pending.append((sock, frame.request_id, layer_id))
        return pending

    def collect_early(pending, rows: int):
        """The reading for every rider of an earlier `issue_host`, in the order they rode."""
        out = _readings_or_zeros(departure, pending)
        joined = torch.cat(out, dim=0) if len(out) > 1 else out[0]
        return joined.reshape(rows, -1).float()

    def apply_host(layer_id, request_ids, packed):
        """Send the finished step for the state and ring advance. Nothing is answered.

        `packed` is the assembly kernel's slab, `[k | v | alpha | beta | column]` in
        float32 -- one tensor because the far end pays per tensor, and packed in the
        kernel because packing on this serial path with casts and a cat gave back what it
        saved (flown, mixed-to-negative). float32 round-trips every part exactly, so the
        arithmetic is unchanged. Its deadline is the next token's read of this layer, a
        whole step away; the next read for the same layer rides the same socket behind
        it, and the far end serves in order.
        """
        offset = 0
        for (frame, sock), n in zip(riding, counts):
            sl = slice(offset, offset + n)
            offset += n
            if any(run != 1 for _, _, run in _runs(request_ids[sl])):
                raise RuntimeError(
                    f"a state apply for layer {layer_id} reached a rider with a chunk's "
                    f"run; its advance travelled with its scan and must not be applied "
                    f"twice."
                )
            body = (packed[sl],)
            if not streamed_send(
                sock, frame.request_id, layer_id, OP_STATE_APPLY, body
            ):
                departure.post_unawaited(
                    sock, frame.request_id, layer_id, body, OP_STATE_APPLY
                )

    def collect_host(handle):
        """`ask_host`'s second half: the wait, and only the wait.

        Not installed on the runner. `issue_host` is, because a caller with work to do between
        the send and the wait needs the send on its own; nothing needs the wait on its own, and
        an installed call that nobody makes is a plan rather than code.
        """
        pending, (rows, heads) = handle
        out = _readings_or_zeros(departure, pending)
        joined = torch.cat(out, dim=0) if len(out) > 1 else out[0]
        return joined.reshape(rows, heads, -1).float()

    def defer_update(layer_id, request_ids, k, v, alpha, beta):
        """Advance the state, later. Only for SINGLE-row riders.

        A multi-row rider already advanced its state inside the scan, because it had to: its
        tokens read each other's updates. Sending a deferred update for it as well would apply
        the same tokens twice.
        """
        offset = 0
        for (frame, sock), n in zip(riding, counts):
            sl = slice(offset, offset + n)
            offset += n
            if n != 1:
                continue
            departure._post_update(
                sock,
                frame.request_id,
                layer_id,
                (k[sl].reshape(n, -1), v[sl].reshape(n, -1), alpha[sl], beta[sl]),
            )

    def mix_host(layer_id, request_ids, packed, alpha, beta):
        """Hand the whole history-touching half of a layer to the caller, and take back `core`.

        The same one crossing `ask_host` uses, carrying different things: the PRE-convolution
        projection with the gates goes down, `core` comes back. The caller convolves against its
        own ring, contracts the state and advances it, so this side writes nothing per request --
        which is what lets any pool answer any call.

        One message a rider, all sent before any is collected, so the wait is one round trip and
        not one per rider. The same reason `ask_host` does it that way.
        """
        # Interleave with queued or streamed frames is the per-fd wire mutex's problem now,
        # and ORDER does not arise: this op serves only layers whose early frame never went,
        # and the far end keys every other frame by (request, layer, op).
        pending = []
        offset = 0
        for (frame, sock), n in zip(riding, counts):
            sl = slice(offset, offset + n)
            offset += n
            body = (packed[sl], alpha[sl], beta[sl])
            send_frame(sock, Frame(frame.request_id, layer_id, body, OP_STATE_MIX))
            with departure._cond:
                departure._sent_at[(id(sock), frame.request_id, layer_id)] = (
                    time.perf_counter()
                )
            pending.append((sock, frame.request_id, layer_id))
        out = _readings_or_zeros(departure, pending)
        return torch.cat(out, dim=0) if len(out) > 1 else out[0]

    departure.runner._local.ask_host = ask_host
    departure.runner._local.issue_host = issue_host
    departure.runner._local.mix_host = mix_host
    departure.runner._local.collect_early = collect_early
    departure.runner._local.apply_host = apply_host
    departure.runner._local.defer_update = defer_update


def _await_reading(departure, sock, request_id: int, layer_id: int):
    """Wait for one caller's state reading, WITHOUT reading the socket here.

    The connection thread owns the socket and is already inside `decode` on it. A second
    reader is not a race that sometimes loses -- it is a deadlock that always happens as soon
    as the departure is taken by the timer thread rather than by the connection thread: the
    connection thread swallows the reading, files it as a new request, and this waits forever.
    The symptom is a watchdog timeout with no traceback, which says nothing about any of this.

    So the connection loop files readings here and this waits on them. It is the same shape as
    `PoolClient`'s INBOUND_OPS, which was built carefully on that side and then rebuilt wrongly
    on this one.
    """
    key = (id(sock), request_id, layer_id)
    with departure._cond:
        deadline = time.perf_counter() + STATE_READING_TIMEOUT_S
        while key not in departure._readings:
            if key[0] in departure._dead:
                # the host is gone and this reading can never arrive. None, not an
                # exception: a mixed departure zero-fills this rider and serves the rest
                # on time, and only a departure with nobody left escalates.
                return None
            if not departure._cond.wait(
                timeout=max(0.0, deadline - time.perf_counter())
            ):
                raise RuntimeError(
                    f"no state reading for request {request_id} layer {layer_id} within "
                    f"{STATE_READING_TIMEOUT_S}s. The far end holds the history this span "
                    f"asked for and did not answer."
                )
        out = departure._readings.pop(key)
        filed = departure._filed_at.pop(key, None)
        sent = departure._sent_at.pop(key, None)
        if filed is not None and sent is not None:
            departure._count_wait(filed - sent, time.perf_counter() - filed)
        return out.to(departure.device)


# Importing this module is what lets a pool serve one linear layer a call. `roles` does it where it
# builds the pool, which is the composition root and the only place that knows what this process is
# for.
