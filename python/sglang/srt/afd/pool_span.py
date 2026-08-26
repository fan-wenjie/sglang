"""The group cut's departures, which are the derived line's and not standard AFD's.

A span is one bus for a whole group of layers -- W_o, then a feed-forward, then three linear
attentions each followed by a feed-forward -- and it exists because the query's read point moved.
Standard AFD serves one feed-forward a call and has no use for any of it, so none of it lives
under `srt/afd`: `pool_server` holds a table, and this file is what claims the group cut's entries
in it.

Free functions rather than methods, taking the departure they serve on. A method would have to be
attached to a class in the other package, which is the coupling the table exists to remove: the
pool does not know these exist, and deleting this directory leaves it serving feed-forwards and
refusing a SPAN frame by name.

`departure.runner` is the span runner, put there by `roles.serve(runner=...)` from what the arm
built. It is the same object these functions used to reach as `departure.span`; only standard
AFD's name for the slot became generic.
"""

from __future__ import annotations

import logging
import threading
import time

import torch

from sglang.srt.afd.pool_linear import _install_history_calls
from sglang.srt.afd.pool_server import namespace_of, register_departure
from sglang.srt.afd.layer_kinds import layer_types_of
from sglang.srt.afd.protocol import (
    OP_NAMES,
    OP_SPAN,
    OP_SPAN_LANE,
    OP_FILES,
    OP_WEIGHTS,
    OP_SPAN_ENTER,
    OP_SPAN_EXIT,
    OP_SPAN_Q,
    Frame,
    send_frame,
    unpack_positions,
)

logger = logging.getLogger(__name__)

# How long the span thread waits for the early read point to reach the wire before calling it a
# failure. It is normally set inside the last feed-forward, so any real wait here means the sending
# thread is stuck rather than slow -- generous enough never to fire on a busy pool, short enough
# that a stuck one fails instead of hanging every caller behind it.
SPAN_HANDOVER_TIMEOUT_S = 30.0

SPAN_OPS = frozenset({OP_SPAN, OP_SPAN_LANE, OP_SPAN_ENTER, OP_SPAN_EXIT})


def _depart_weights(departure, group: int, op: int, riding) -> None:
    """The residual weights a weightless host computes with, in layer order.

    Fixed order IS the schema: for each linear layer in index order, the convolution
    filter and its bias; then the final norm's weight. Both ends derive the same
    list from the same config, so nothing about it needs naming on the wire.
    """
    model = departure.runner.model
    types = layer_types_of(model)
    out = []
    for index, kind in enumerate(types):
        if kind == "full_attention":
            continue
        conv = model.model.layers[index].linear_attn.conv1d
        # a frame carries (rows, columns); the host rebuilds each tensor to the
        # shape it constructed itself, so one row of numbers is the whole story
        out.append(conv.weight.detach().reshape(1, -1))
        out.append(
            conv.bias.detach().reshape(1, -1)
            if conv.bias is not None
            else torch.zeros(1, 0, device=conv.weight.device)
        )
    out.append(model.model.norm.weight.detach().reshape(1, -1))
    for frame, sock in riding:
        with departure._wire_lock:
            send_frame(sock, Frame(frame.request_id, group, tuple(out), OP_WEIGHTS))


_SPANS = [
    0,
    0.0,
    0.0,
    0,
]  # count, window wall, wire at window start, waits at window start -- module-level
# on purpose: a span is served on the departure thread and there is one of those, so no
# lock is needed and an instance attribute would be lost every time the departure is
# rebuilt. The wire share is a WINDOW, not a lifetime: the TCP callbacks a span makes
# depend on which transport the arrangement rides -- with the lane up, a decode span
# makes none -- and a lifetime average would keep quoting the prefill's wire against
# decode walls forever (found live: a lane deployment reporting "100%% waiting on the
# host" out of numbers three phases old).


def _span_report(departure, wall_s: float) -> None:
    """Print, every 200 spans, how the window's span time divides."""
    _SPANS[0] += 1
    _SPANS[1] += wall_s
    if _SPANS[0] % 200:
        return
    wall_ms = 1e3 * _SPANS[1] / 200
    waits = departure._waits - _SPANS[3]
    wire_s = departure._wire_s - _SPANS[2]
    _SPANS[1] = 0.0
    _SPANS[2] = departure._wire_s
    _SPANS[3] = departure._waits
    if waits <= 0:
        logger.info(
            "afd pool: %s span(s) -- %.3f ms wall each; no TCP callback waited in "
            "this window (the arrangement's reads ride the lane or fold into the "
            "span's own wall)",
            _SPANS[0],
            wall_ms,
        )
        return
    wire_ms = 1e3 * wire_s / waits
    logger.info(
        "afd pool: %s span(s) -- %.3f ms wall each, and the window's %s TCP "
        "callback(s) waited %.3f ms each",
        _SPANS[0],
        wall_ms,
        waits,
        wire_ms,
    )


def _depart_span(departure, group: int, op: int, riding) -> None:
    """One bus: a whole group of layers for everybody who boarded before it left.

    The batch is fixed for the span's whole 2046 us -- four feed-forwards read 2760 MiB of
    weights once, and a rider joining halfway would need that read done again. At the far end
    the riders disperse: each host computes its own softmax attention, taking as long as its
    own context takes, and boards whichever later bus it is in time for. That is what lets one
    pool serve a 1k request and a 128k one without the short one waiting on the long one.

    There is no departure timer to tune here. The previous span IS the timer: riders accumulate
    while it runs, and the next bus leaves with whoever is waiting when the pool comes free.
    """
    if departure.runner is None:
        raise RuntimeError(
            f"a span frame for group {group} reached a pool with no span runner. This pool "
            f"serves the per-layer cut; the host is speaking the group cut. Neither end can "
            f"tell from the frames alone, which is what the HELLO exchange is for."
        )
    started = time.perf_counter()
    # four tensors when the caller carries its own convolution rings, three when it leaves
    # them here. The ring is the last per-request thing this side holds, and a caller that
    # sends it makes this pool stateless WITHOUT QUALIFICATION -- which is what lets any pool
    # answer any call, and therefore what the multi-pool routing rests on.
    departure._expect_tensors(
        riding[0][0],
        (3, 4),
        "SPAN: attention output, row ids, positions [, conv rings]",
    )
    # The fourth tensor is the span's convolution WINDOWS -- the ring's last three columns per
    # linear layer, 60 KiB a layer a row -- and it is taken. A refusal used to stand here, from
    # a ledger that weighed the traffic against the whole callback's blocking rather than
    # against the host work it removes; corrected, the windows buy the host out of every
    # convolution and normalisation on its blocking path. The refusal's one live point survives
    # as a gate at the SENDER: `_decode_windows` attaches nothing for prefill chunks, whose
    # rows would multiply the traffic by the chunk length for a path the scan already serves.
    counts = [f.tensors[0].shape[0] for f, _ in riding]
    joined = torch.cat([f.tensors[0] for f, _ in riding], dim=0).to(departure.device)
    # NAMESPACED by the rider's socket. Two hosts' schedulers hand out row ids from similar
    # counters, and every pool-side table this call feeds -- the residuals, the gates, the
    # early handles -- keys on the id. Bare ids from two hosts collide and one request
    # continues from another's residual, fluently. The namespace never travels: callbacks are
    # sliced per rider and carry the rider's own frames, so only this side's keying widens.
    ids = [
        namespace_of(sock) | int(r)
        for f, sock in riding
        for r in f.tensors[1].reshape(-1).tolist()
    ]
    # concatenated along the TOKEN axis, which is the last one -- mrope's rows are axes, not
    # riders, so joining along the first would stack one rider's height row onto another's
    # temporal row and rotate every token to somewhere nobody asked for
    positions = unpack_positions(
        torch.cat([f.tensors[2] for f, _ in riding], dim=-1)
    ).to(departure.device)
    if len(ids) != joined.shape[0]:
        raise RuntimeError(
            f"{len(ids)} row id(s) for {joined.shape[0]} row(s) in group {group}'s span. Every "
            f"row has to say whose recurrent state it advances, and a mismatch folds one "
            f"request's token into another's memory with no symptom in the output."
        )

    if op == OP_SPAN_EXIT:
        want_logits = any(len(f.tensors) > 3 for f, _ in riding)
        if want_logits:
            # the head lives with the weights: the reply carries each request's
            # last-row logits beside the hidden stream, one batched GEMM for the
            # whole departure where each host used to read 2.37 GiB alone
            out, logits = departure.runner.run_epilogue_with_logits(ids, group, joined)
            _reply_exit(departure, riding, counts, ids, group, out, logits)
        else:
            out = departure.runner.run_epilogue(ids, group, joined)
            _reply_pieces(departure, riding, counts, group, (out,), OP_SPAN_EXIT)
    else:
        # both halves of the reply go down the SAME socket, and the early one is sent from
        # another thread. Two threads inside `send_frame` on one socket interleave a header
        # with somebody else's payload, and the far end reads the remainder as the next
        # frame's header -- so the second half waits for the first to be on the wire. The
        # wait costs nothing: it is spent inside the last feed-forward either way.
        # The fourth tensor, when the caller attached one, is the span's convolution windows:
        # each linear layer's last pre-convolution columns, because the ring is the caller's and
        # the convolution is about to run here. Every rider or none -- a departure mixing hosts
        # that attach with hosts that do not would convolve some rows against nothing.
        attached = [f.tensors[3] if len(f.tensors) > 3 else None for f, _ in riding]
        if any(w is not None for w in attached):
            if any(w is None for w in attached):
                raise RuntimeError(
                    f"group {group}'s departure carries riders with and without convolution "
                    f"windows. The two protocols cook on different sides; one departure cannot "
                    f"serve both."
                )
            windows = torch.cat(attached, dim=0).to(departure.device)
        else:
            windows = None
        sent = threading.Event()
        handover = _handover(departure, riding, counts, group, sent)
        _install_history_calls(departure, riding, counts)
        span_began = time.perf_counter()
        wire_at_entry = departure._wire_s
        if op == OP_SPAN_ENTER:
            # int64 means the caller holds no embedding and sent token ids instead. Decided by the
            # dtype rather than by a flag because the frame either carries ids or it does not, and
            # a flag could disagree with the bytes on the wire.
            entering = (
                departure.runner.embed(joined)
                if joined.dtype == torch.int64
                else joined
            )
            _, k, v = departure.runner.run_prologue(
                ids, entering, positions, on_query=handover, windows=windows
            )
        else:
            use_lane = op == OP_SPAN_LANE
            if use_lane:
                from sglang.srt.afd.lane import the_lane

                lane = the_lane()
                if lane is None or not lane.ready:
                    raise RuntimeError(
                        f"group {group} was asked as span_lane and this pool's lane is "
                        f"{'not up yet' if lane else 'not configured'}. The host decides "
                        f"the op from the adopted configuration, so the two ends disagree "
                        f"about --afd-transfer-backend or the lane died; check the pool "
                        f"log for the lane's own line."
                    )
            departure.runner._local.use_lane = use_lane
            try:
                _, k, v = departure.runner.run(
                    ids, group, joined, positions, on_query=handover, windows=windows
                )
            finally:
                departure.runner._local.use_lane = False
        # How a span's wall time divides between what this pool COMPUTES and what it WAITS for.
        # The question the split answers -- when the two ends are overlapped, which one is the
        # max -- cannot be answered from either end alone: the pool sees one number and the host
        # sees another, and the design's claim is that the smaller hides inside the larger.
        # `departure._wire_s` is the callback total, accumulated by the connection thread as each
        # reading is filed, so the difference is this pool's own arithmetic.
        _span_report(departure, time.perf_counter() - span_began)
        if not sent.wait(timeout=SPAN_HANDOVER_TIMEOUT_S):
            raise RuntimeError(
                f"group {group}'s read point was not on the wire "
                f"{SPAN_HANDOVER_TIMEOUT_S}s after the span finished. The host is blocked "
                f"waiting for it and sending the second half now would interleave two frames "
                f"on one socket."
            )
        # replied under the op it was ASKED with, not the literal OP_SPAN. The reply table
        # is keyed by (request, layer, op) -- which is what stops a span's two halves being
        # confused -- so a prologue asked as OP_SPAN_ENTER and answered as OP_SPAN is filed
        # under a key nobody is waiting on. The caller then waits forever having already
        # received the early half, which is exactly what it looked like: the pool idle with
        # nothing queued, the host blocked in collect_kv.
        _reply_pieces(departure, riding, counts, group, (k, v), op)

    departure.departures.append(
        {
            "layer": group,
            "riders": len(riding),
            "tokens": int(joined.shape[0]),
            "started": started,
            "seconds": time.perf_counter() - started,
            "op": OP_NAMES.get(op, op),
        }
    )
    if departure.riders_path and len(departure.departures) % 200 == 0:
        departure._write_riders()


def _handover(departure, riding, counts, group: int, sent: threading.Event):
    """Send the shifted read point without waiting for the feed-forward behind it.

    The read point exists one feed-forward before the span's output. Copying it to the host
    with a plain `.cpu()` would synchronise the stream at exactly that moment -- BEFORE the
    last feed-forward has been issued -- so the pool would sit idle for the length of the copy
    and the send, and then start the feed-forward. That closes the window this whole two-part
    reply exists to open, and it closes it silently: the answers stay right and the overlap
    just is not there. The same mistake, in the same direction, cost this arrangement a round
    of measurements on the host side.

    So the copy is queued on the stream, an event is recorded after it, and this returns
    immediately. The caller issues the last feed-forward on top; a second thread waits on the
    event -- which fires when the COPY is done, not when the feed-forward is -- and sends. The
    GPU is busy with the feed-forward for the whole of the send.
    """

    def hand_over(read_point: torch.Tensor) -> None:
        staged = read_point.to("cpu", non_blocking=True)
        copied = torch.cuda.Event()
        copied.record()

        def when_copied() -> None:
            try:
                copied.synchronize()
                _reply_pieces(departure, riding, counts, group, (staged,), OP_SPAN_Q)
            finally:
                # set even on failure: the span thread is waiting on this before it sends the
                # second half, and a handover that died silently would hang the pool rather
                # than fail it
                sent.set()

        threading.Thread(target=when_copied, daemon=True, name="afd-span-q").start()

    return hand_over


def _reply_exit(departure, riding, counts, ids, group: int, hidden, logits) -> None:
    """The exit's two tensors, cut on their OWN axes: hidden by rows, logits by requests."""
    from sglang.srt.afd.slots import _runs

    runs = _runs(ids)
    row_offset = 0
    req_offset = 0
    for (frame, sock), n in zip(riding, counts):
        nreq = sum(1 for _, first, cnt in runs if row_offset <= first < row_offset + n)
        pieces = (
            hidden[row_offset : row_offset + n],
            logits[req_offset : req_offset + nreq],
        )
        row_offset += n
        req_offset += nreq
        try:
            with departure._wire_lock:
                send_frame(sock, Frame(frame.request_id, group, pieces, OP_SPAN_EXIT))
        except OSError:
            departure.host_departed(sock)


def _reply_pieces(departure, riding, counts, group: int, out: tuple, op: int) -> None:
    """Cut one batched answer back into the rows each caller sent."""
    offset = 0
    for (frame, sock), n in zip(riding, counts):
        pieces = tuple(t[offset : offset + n] for t in out)
        offset += n
        try:
            # the same lock the sender thread takes: the departure thread, the handover thread and
            # the update sender all write this socket, and two sendmsg calls interleaved produce a
            # byte stream neither end can parse. The two-part span reply was kept apart by its
            # `sent` barrier alone, which the updates do not pass through.
            with departure._wire_lock:
                send_frame(sock, Frame(frame.request_id, group, pieces, op))
        except OSError:
            logger.warning(
                "caller for request %s group %s went away before its %s reply",
                frame.request_id,
                group,
                OP_NAMES.get(op, op),
            )


# Claiming the table's entries is what importing this module DOES, so importing the package is what
# makes a pool able to serve a span. All four call back to the caller mid-run -- they read a
# recurrent state the caller holds -- and that is declared in the same call that claims the op,
# because the two facts living apart is what hung this pool once.
for _span_op in SPAN_OPS:
    register_departure(_span_op, _depart_span, calls_back=True)
# the residual-weights push calls nothing back: it reads the pool's own parameters
register_departure(OP_WEIGHTS, _depart_weights)


def _depart_files(departure, group: int, op: int, riding) -> None:
    """The model's papers, for a host bootstrapping from nothing but an address."""
    from sglang.srt.afd.model_files import files_reply
    from sglang.srt.runtime_context import get_model

    out = files_reply(get_model().model_path)
    for frame, sock in riding:
        with departure._wire_lock:
            send_frame(sock, Frame(frame.request_id, group, out, OP_FILES))


register_departure(OP_FILES, _depart_files)
