"""The pool: feed-forward weights, no request state, one departure at a time.

Two properties the arrangement rests on, and both are the reason this is a separate process
rather than a method call:

    stateless             every call is a complete transaction. A caller that stalls stops
                          calling and blocks nobody, and the pool need not be reserved for a
                          request between that request's calls -- there is nothing of it to
                          protect. That is what lets a long-context request give the pool up
                          while it sweeps and take it back afterwards.

    context-free latency  the pool reads the same weights whatever the caller's context length.
                          A request sweeping a million positions and one sweeping a thousand cost
                          it exactly the same, which is why they can share it and why the pool
                          must NOT inherit the attention side's batching.

Departures follow the rule the study's worked example uses: wait for `min_batch` callers, then
carry EVERY caller at the stop rather than the first two. `max_wait_s` is not optional decoration
-- with a strict minimum and no timeout the last caller of a draining workload waits for a partner
that never arrives, and the request hangs rather than fails.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable

import torch
from sglang.srt.afd.protocol import (
    OP_FFN,
    OP_APPEND,
    OP_KVPROJ,
    OP_NAMES,
    OP_RELEASE,
    OP_SWEEP,
    OP_HELLO,
    OP_LINEAR,
    OP_SWEEP_Q,
    OP_SPAN,
    OP_SPAN_ENTER,
    OP_SPAN_EXIT,
    OP_SPAN_Q,
    unpack_positions,
    Frame,
    decode,
    encode,
    send_frame,
)

logger = logging.getLogger(__name__)


# How long the span thread waits for the early read point to reach the wire before calling it a
# failure. It is normally set inside the last feed-forward, so any real wait here means the sending
# thread is stuck rather than slow -- generous enough never to fire on a busy pool, short enough
# that a stuck one fails instead of hanging every caller behind it.
SPAN_HANDOVER_TIMEOUT_S = 30.0


class Departure(threading.Thread):
    """Collects frames and hands whole batches to the feed-forward."""

    def __init__(
        self,
        forward: Callable[[torch.Tensor, int], torch.Tensor],
        min_batch: int,
        max_wait_s: float,
        device: torch.device | str,
    ):
        super().__init__(name="afd-pool-departure", daemon=True)
        self.forward = forward
        # set by serve() when the pool also holds the KV cache; None means feed-forward only
        self.attention = None
        # set when this process is the CACHE pool of a two-pool split: it answers sweeps and
        # appends and holds no weights at all
        self.cache = None
        # set alongside `cache` when sweeps may arrive before the appends they read. None keeps
        # the pre-length behaviour, where a sweep is answered against whatever is held and the
        # transport is trusted to have ordered it.
        self.parked = None
        # set when this pool serves the GROUP cut: a whole span of layers a call, rather than
        # one feed-forward a call. None means it serves the per-layer cut, and a span frame is
        # refused by name rather than half-served.
        self.span = None
        # set when this pool also holds recurrent states, which is the linear-attention half of
        # the same idea: a per-request read belongs with the data. None means it answers sweeps
        # and appends only, and a LINEAR frame is refused rather than half-served.
        self.linear = None
        self.min_batch = min_batch
        self.max_wait_s = max_wait_s
        self.device = device
        self._cond = threading.Condition()
        # per layer, because a dense stack's layer weights differ: a departure is same-layer or it
        # is not a departure. Callers at different layers wait in different queues.
        self._waiting: dict[int, list[tuple[Frame, socket.socket]]] = {}
        # when a layer's queue first became non-empty. Held here rather than in run()'s frame
        # because `offer` can now take the departure itself, and a timer that still believed the
        # popped queue was waiting would depart the NEXT caller instantly.
        self._first_seen: dict[int, float] = {}
        self._stop = False
        self.departures: list[dict] = []
        # where to write the riders histogram, or None to keep it in memory only
        self.riders_path: str | None = None

    def capabilities(self) -> int:
        """What this pool serves, as a bitmask on the wire.

        1  feed-forward
        2  a cache: sweeps and appends
        4  the key/value projection with the cache
        8  whole spans: the group cut, four layers a call

        Sent in reply to a HELLO so a host learns at STARTUP that it has reached a pool which does
        not do what it was configured to ask for. Without it the mismatch surfaces as the first
        frame the far end cannot parse, the connection thread dies, and the host reports "closed
        mid-call" -- which describes the socket and not the configuration that broke it.

        The span bit matters more than the others because the two cuts differ in what the HOST
        holds. A host built for the group cut has no feed-forward weights and no recurrent state;
        pointed at a per-layer pool it would run out of layers to ask for rather than fail.
        """
        bits = 1
        if self.cache is not None:
            bits |= 2
        if self.attention is not None:
            bits |= 2 | 4
        if self.span is not None:
            bits |= 8
        return bits

    def answer_directly(self, frame: Frame, sock: socket.socket) -> bool:
        """Ops answered on the connection thread rather than queued for a departure.

        A departure exists to make one weight read serve many callers. A sweep reads the CALLER'S
        OWN cache -- there is no shared read to amortise, and measured at 16k context it is 88% of
        what the pool costs per token even at 64 requests -- so queueing it would add the
        departure's latency to buy batching it cannot use.

        Every layout is CHECKED rather than unpacked. A tuple unpack going wrong here raises
        inside a socket thread, the connection dies, and the caller reports "closed mid-call" --
        a message about a socket that names neither the frame nor the configuration behind it.
        That cost two debugging rounds in this tree before the checks went in.

            HELLO     anything                     ->  what this pool serves
            KVPROJ    normalised x_l, positions    ->  k, v, gate
            SWEEP     normalised x_l, positions, row ids -> o, lse, score, v
            RELEASE   anything                     ->  layers dropped
        """
        if frame.op == OP_HELLO:
            send_frame(sock, Frame(frame.request_id, 0,
                                   (torch.tensor([[float(self.capabilities())]]),), OP_HELLO))
            return True
        if self.cache is not None and frame.op in (OP_SWEEP_Q, OP_APPEND, OP_RELEASE):
            return self._answer_cache(frame, sock)
        if self.attention is None or frame.op not in (OP_SWEEP, OP_RELEASE, OP_KVPROJ):
            return False

        device = self.device
        if frame.op == OP_KVPROJ:
            self._expect_tensors(frame, (2,), "KVPROJ: normalised x_l, positions")
            normed, positions = frame.tensors[0], unpack_positions(frame.tensors[1])
            k, v, gate = self.attention.project_kv(
                frame.layer, normed.to(device), positions.to(device)
            )
            send_frame(sock, Frame(frame.request_id, frame.layer, (k, v, gate), OP_KVPROJ))
            return True
        if frame.op == OP_RELEASE:
            dropped = self.attention.holder.release(frame.request_id)
            send_frame(sock, Frame(frame.request_id, frame.layer,
                                   (torch.tensor([[float(dropped)]]),), OP_RELEASE))
            return True

        self._expect_tensors(frame, (2, 3), "SWEEP: normalised x_l, positions [, row ids]")
        normed, positions = frame.tensors[0], unpack_positions(frame.tensors[1])
        ids = frame.tensors[2].view(-1) if len(frame.tensors) > 2 else frame.request_id
        o, lse, score, v = self.attention.serve_attention(
            ids, frame.layer, normed.to(device), positions.to(device),
        )
        send_frame(sock, Frame(frame.request_id, frame.layer, (o, lse, score, v), OP_SWEEP))
        return True

    def _answer_cache(self, frame: Frame, sock: socket.socket) -> bool:
        """The cache pool's three ops. None of them touches a weight or a hidden state.

        Frame layouts, checked rather than unpacked. A tuple unpack that goes wrong raises "too
        many values to unpack" from inside a socket thread, which reaches the caller as a broken
        pipe with no explanation -- the wrong end of the connection learning the wrong thing. A
        protocol mismatch should name itself.

            SWEEP_Q   q, row request ids [, expected history lengths]  ->  o, lse
            APPEND    k, v, row request ids                            ->  positions held
            RELEASE   anything                                         ->  layers dropped
        """
        device = self.device
        if frame.op == OP_SWEEP_Q:
            self._expect_tensors(frame, (2, 3), "SWEEP_Q: q, row ids [, expected lengths]")
            q, ids = frame.tensors[0], frame.tensors[1].view(-1)
            expect = frame.tensors[2].view(-1) if len(frame.tensors) > 2 else None
            return self._sweep_or_park(frame, sock, q.to(device), ids, expect)
        if frame.op == OP_LINEAR:
            self._expect_tensors(
                frame, (6,), "LINEAR: mixed_qkv, a, b, A_log, dt_bias, row ids")
            if self.linear is None:
                raise ConnectionError(
                    "a LINEAR frame reached a pool that holds no recurrent states. Start it with "
                    "--linear-slots, or the host is configured to move its linear layers here and "
                    "this pool was not."
                )
            qkv, a, b, a_log, dt, ids = frame.tensors
            out = self.linear.step(
                [int(r) for r in ids.view(-1)], frame.layer, qkv.to(device), a.to(device),
                b.to(device), a_log.view(-1).to(device), dt.view(-1).to(device))
            send_frame(sock, Frame(frame.request_id, frame.layer, (out,), OP_LINEAR))
            return True
        if frame.op == OP_APPEND:
            self._expect_tensors(frame, (3,), "APPEND: k, v, row ids")
            k, v, ids = frame.tensors[0], frame.tensors[1], frame.tensors[2].view(-1)
            held = self.cache.append_rows(ids, frame.layer, k.to(device), v.to(device))
            most = max(held.values()) if held else 0
            send_frame(sock, Frame(frame.request_id, frame.layer,
                                   (torch.tensor([[float(most)]]),), OP_APPEND))
            # An append is what makes a parked sweep readable. Released AFTER the reply above, so
            # the appending caller is never made to wait on somebody else's sweep, and outside the
            # buffer's lock, because a resume writes to a socket whose far end may be parking.
            if self.parked is not None:
                for request_id, layer in {(int(r), frame.layer) for r in ids}:
                    for entry in self.parked.release(
                            request_id=request_id, layer=layer,
                            held=self.cache.holder.positions(request_id, layer)):
                        entry.resume()
            return True
        if self.linear is not None:
            # a recurrent state has no length to reset, so the slot's buffers are zeroed on
            # release; handing one over uncleared gives the next request the previous one's
            # memory and the output stays fluent
            self.linear.states.release(frame.request_id)
        if self.parked is not None:
            # a released request's parked sweeps can never be satisfied: the history they name is
            # exactly what is being dropped. Left behind they would sit until the timeout and
            # report a lost append that was never lost.
            self.parked.drop(request_id=frame.request_id)
        dropped = self.cache.holder.release(frame.request_id)
        send_frame(sock, Frame(frame.request_id, frame.layer,
                               (torch.tensor([[float(dropped)]]),), OP_RELEASE))
        return True

    def _sweep_or_park(self, frame: Frame, sock: socket.socket, q, ids, expect) -> bool:
        """Answer this sweep, or hold it until the history it names exists.

        The length on the wire says which positions this sweep covers. When the cache holds them,
        it is answered here on the connection thread, which is the common case and stays cheap.
        When it does not, the sweep is PARKED and this thread returns to reading -- because the
        append it is waiting for travels the same socket, so blocking here would block the only
        thread that could deliver it. That deadlock has no message: the pool simply stops
        answering.

        Without a length the sweep is answered immediately against whatever is held. That is the
        pre-length behaviour and it is correct exactly while the transport keeps sweeps and appends
        in order on one connection -- which is true today and stops being true at the first shard.
        """
        if self.parked is None or expect is None:
            o, lse = self.cache.sweep_rows(ids, frame.layer, q, expect=expect)
            send_frame(sock, Frame(frame.request_id, frame.layer, (o, lse), OP_SWEEP_Q))
            return True

        wanted = {int(r): int(e) for r, e in zip(ids.tolist(), expect.tolist())}
        short = [r for r, n in wanted.items()
                 if self.cache.holder.positions(r, frame.layer) < n]
        if not short:
            o, lse = self.cache.sweep_rows(ids, frame.layer, q, expect=expect)
            send_frame(sock, Frame(frame.request_id, frame.layer, (o, lse), OP_SWEEP_Q))
            return True

        def resume() -> None:
            still = [r for r, n in wanted.items()
                     if self.cache.holder.positions(r, frame.layer) < n]
            if still:
                return               # another row is still short; a later append will call again
            o, lse = self.cache.sweep_rows(ids, frame.layer, q, expect=expect)
            send_frame(sock, Frame(frame.request_id, frame.layer, (o, lse), OP_SWEEP_Q))

        # parked once per short row: any of their appends may be the one that completes the set,
        # and `resume` re-checks all of them before answering, so extra wakeups are harmless and a
        # missed one is not
        for request_id in short:
            self.parked.park(request_id=request_id, layer=frame.layer,
                             length=wanted[request_id], resume=resume)
        return True

    def _write_riders(self) -> None:
        import json

        recent = self.departures[-2000:]
        try:
            with open(self.riders_path, "w") as f:
                json.dump(_riders_summary(recent), f, indent=2)
        except OSError:                       # a full disk must not stop the pool serving
            pass

    @staticmethod
    def _expect_tensors(frame: Frame, allowed: tuple, layout: str) -> None:
        if len(frame.tensors) not in allowed:
            raise ConnectionError(
                f"{OP_NAMES.get(frame.op, frame.op)} frame for request {frame.request_id} layer "
                f"{frame.layer} carries {len(frame.tensors)} tensor(s); this pool speaks "
                f"{sorted(allowed)}. Layout is {layout}. The two sides are running different "
                f"versions of the protocol, and unpacking anyway would answer with something "
                f"nobody asked for."
            )

    def offer(self, frame: Frame, sock: socket.socket) -> None:
        """Queue a frame, and depart it here if that completes a batch.

        The connection thread used to hand every frame to the departure thread and go back to the
        socket, which costs a condition-variable wake and a scheduler round trip per call -- on a
        single-caller pool, where the batch is complete the moment it arrives, that handoff buys
        nothing. When the queue is already full the offering thread takes the departure itself.
        The queue is popped under the lock, so exactly one thread ever carries a given set of
        riders, and the timer thread still owns the partial-batch case.
        """
        riding = None
        with self._cond:
            queue = self._waiting.setdefault(frame.layer, [])
            queue.append((frame, sock))
            if len(queue) >= self.min_batch:
                riding = self._waiting.pop(frame.layer)
                self._first_seen.pop(frame.layer, None)
            else:
                self._first_seen.setdefault(frame.layer, time.perf_counter())
                self._cond.notify()
        if riding is not None:
            self._depart(frame.layer, riding)

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def _ready_layer(self, now: float) -> int | None:
        for layer, queue in self._waiting.items():
            if not queue:
                continue
            if len(queue) >= self.min_batch:
                return layer
            if now - self._first_seen.get(layer, now) >= self.max_wait_s:
                # the timeout the strict rule needs. Without it the last caller of a draining
                # workload waits for a partner that never comes.
                return layer
        return None

    def run(self) -> None:
        """The timeout half. The full-batch half is taken by whichever thread offered the frame."""
        while True:
            with self._cond:
                while True:
                    if self._stop and not any(self._waiting.values()):
                        return
                    now = time.perf_counter()
                    for layer, queue in self._waiting.items():
                        if queue and layer not in self._first_seen:
                            self._first_seen[layer] = now
                        if not queue:
                            self._first_seen.pop(layer, None)
                    layer = self._ready_layer(now)
                    if layer is not None:
                        break
                    self._cond.wait(timeout=self.max_wait_s / 4 or 0.01)
                riding = self._waiting.pop(layer)
                self._first_seen.pop(layer, None)
            self._depart(layer, riding)

    def _depart(self, layer: int, riding: list[tuple[Frame, socket.socket]]) -> None:
        """Serve one batch. Which kind of batch is decided by the riders' opcode.

        Riders are queued by layer, so a pool serving both cuts at once could put a feed-forward
        frame and a span frame for the same layer in one queue. They are different jobs with
        different replies, and mixing them would answer one caller with the other's arithmetic --
        refused rather than dispatched on the first rider.
        """
        ops = {f.op for f, _ in riding}
        if len(ops) != 1:
            raise RuntimeError(
                f"layer {layer}'s departure mixes {sorted(OP_NAMES.get(o, o) for o in ops)}. A "
                f"departure serves one job; these callers asked for different ones and answering "
                f"them from one batch would give each the other's arithmetic."
            )
        if ops <= {OP_SPAN, OP_SPAN_ENTER, OP_SPAN_EXIT}:
            return self._depart_span(layer, ops.pop(), riding)
        return self._depart_feed_forward(layer, riding)

    def _depart_span(self, group: int, op: int, riding) -> None:
        """One bus: a whole group of layers for everybody who boarded before it left.

        The batch is fixed for the span's whole 2046 us -- four feed-forwards read 2760 MiB of
        weights once, and a rider joining halfway would need that read done again. At the far end
        the riders disperse: each host computes its own softmax attention, taking as long as its
        own context takes, and boards whichever later bus it is in time for. That is what lets one
        pool serve a 1k request and a 128k one without the short one waiting on the long one.

        There is no departure timer to tune here. The previous span IS the timer: riders accumulate
        while it runs, and the next bus leaves with whoever is waiting when the pool comes free.
        """
        if self.span is None:
            raise RuntimeError(
                f"a span frame for group {group} reached a pool with no span runner. This pool "
                f"serves the per-layer cut; the host is speaking the group cut. Neither end can "
                f"tell from the frames alone, which is what the HELLO exchange is for."
            )
        started = time.perf_counter()
        counts = [f.tensors[0].shape[0] for f, _ in riding]
        joined = torch.cat([f.tensors[0] for f, _ in riding], dim=0).to(self.device)
        ids = torch.cat([f.tensors[1] for f, _ in riding], dim=0).reshape(-1).tolist()
        if len(ids) != joined.shape[0]:
            raise RuntimeError(
                f"{len(ids)} row id(s) for {joined.shape[0]} row(s) in group {group}'s span. Every "
                f"row has to say whose recurrent state it advances, and a mismatch folds one "
                f"request's token into another's memory with no symptom in the output."
            )

        if op == OP_SPAN_EXIT:
            out = self.span.run_epilogue(ids, group, joined)
            self._reply_pieces(riding, counts, group, out, OP_SPAN_EXIT)
        else:
            # both halves of the reply go down the SAME socket, and the early one is sent from
            # another thread. Two threads inside `send_frame` on one socket interleave a header
            # with somebody else's payload, and the far end reads the remainder as the next
            # frame's header -- so the second half waits for the first to be on the wire. The
            # wait costs nothing: it is spent inside the last feed-forward either way.
            sent = threading.Event()
            handover = self._handover(riding, counts, group, sent)
            if op == OP_SPAN_ENTER:
                _, out = self.span.run_prologue(ids, joined, on_read_point=handover)
            else:
                _, out = self.span.run(ids, group, joined, on_read_point=handover)
            if not sent.wait(timeout=SPAN_HANDOVER_TIMEOUT_S):
                raise RuntimeError(
                    f"group {group}'s read point was not on the wire "
                    f"{SPAN_HANDOVER_TIMEOUT_S}s after the span finished. The host is blocked "
                    f"waiting for it and sending the second half now would interleave two frames "
                    f"on one socket."
                )
            self._reply_pieces(riding, counts, group, out, OP_SPAN)

        self.departures.append(
            {"layer": group, "riders": len(riding), "tokens": int(joined.shape[0]),
             "started": started, "seconds": time.perf_counter() - started,
             "op": OP_NAMES.get(op, op)}
        )
        if self.riders_path and len(self.departures) % 200 == 0:
            self._write_riders()

    def _handover(self, riding, counts, group: int, sent: threading.Event):
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
                    self._reply_pieces(riding, counts, group, staged, OP_SPAN_Q)
                finally:
                    # set even on failure: the span thread is waiting on this before it sends the
                    # second half, and a handover that died silently would hang the pool rather
                    # than fail it
                    sent.set()

            threading.Thread(target=when_copied, daemon=True, name="afd-span-q").start()

        return hand_over

    def _reply_pieces(self, riding, counts, group: int, out: torch.Tensor, op: int) -> None:
        """Cut one batched answer back into the rows each caller sent."""
        offset = 0
        for (frame, sock), n in zip(riding, counts):
            piece = out[offset : offset + n]
            offset += n
            try:
                send_frame(sock, Frame(frame.request_id, group, (piece,), op))
            except OSError:
                logger.warning("caller for request %s group %s went away before its %s reply",
                               frame.request_id, group, OP_NAMES.get(op, op))

    def _depart_feed_forward(self, layer: int, riding: list[tuple[Frame, socket.socket]]) -> None:
        started = time.perf_counter()
        widths = {f.tensor.shape[1] for f, _ in riding}
        if len(widths) != 1:
            raise RuntimeError(f"layer {layer} departure mixes hidden widths {widths}")
        counts = [f.tensor.shape[0] for f, _ in riding]
        # one rider is the common case on a single-caller pool, and torch.cat on a one-element
        # list still copies the whole frame
        joined = riding[0][0].tensor if len(riding) == 1 else torch.cat(
            [f.tensor for f, _ in riding], dim=0
        )
        batch = joined.to(self.device)
        out = self.forward(batch, layer)
        if out.shape != batch.shape:
            raise RuntimeError(
                f"layer {layer} feed-forward returned {tuple(out.shape)} for {tuple(batch.shape)}; "
                f"the pool returns the same tokens it was given"
            )
        offset = 0
        for (frame, sock), n in zip(riding, counts):
            piece = out[offset : offset + n]
            offset += n
            try:
                send_frame(sock, Frame.one(frame.request_id, layer, piece))
            except OSError:
                logger.warning(
                    "caller for request %s layer %s went away before its reply",
                    frame.request_id,
                    layer,
                )
        self.departures.append(
            {
                "layer": layer,
                "riders": len(riding),
                "tokens": int(batch.shape[0]),
                "started": started,
                "seconds": time.perf_counter() - started,
            }
        )
        # The riders-per-departure histogram is the pool's whole economics in one number: a pool
        # whose departures carry one rider read its weights for one caller, which is the same read
        # the caller would have done itself plus a wire. Written where an operator can see it,
        # because it is not visible from either end -- the host sees a slow layer and the pool
        # sees a busy one, and neither sees that the batch was empty.
        if self.riders_path and len(self.departures) % 200 == 0:
            self._write_riders()


def _riders_summary(departures: list[dict]) -> dict:
    """How many callers rode each departure, and what that made the weight read worth.

    `riders` is what a departure carried; `tokens` is the rows those riders brought between them.
    A pool serving N hosts should show riders clustered at N, and one showing a spike at 1 is a
    pool whose batching never fired -- which costs its callers a round trip and saves them
    nothing.
    """
    from collections import Counter

    riders = Counter(d["riders"] for d in departures)
    tokens = [d["tokens"] for d in departures]
    return {
        "departures": len(departures),
        "riders_histogram": dict(sorted(riders.items())),
        "mean_riders": sum(d["riders"] for d in departures) / max(len(departures), 1),
        "mean_tokens": sum(tokens) / max(len(tokens), 1),
        "seconds_median": sorted(d["seconds"] for d in departures)[len(departures) // 2]
        if departures else 0.0,
    }


def _expire_parked(departure: Departure) -> None:
    """Fail parked sweeps whose appends never came, and say what they were waiting for.

    A parked sweep that is never released is a lost append, a dead peer, or a plan error, and this
    thread cannot tell them apart -- so it reports rather than retries. Retrying here would turn a
    dead peer into a stall with nothing in any log.

    The connection is closed rather than answered with an error frame: the protocol has no error
    op, and inventing one that an older host would parse as a sweep result is worse than a broken
    pipe, which every caller already handles.
    """
    while True:
        time.sleep(departure.parked.timeout_s / 4.0)
        for entry in departure.parked.expired():
            logger.warning(
                "afd pool: sweep for request %s layer %s waited %.1fs for %s cached position(s) "
                "and they never arrived. Dropping the connection; the host decides whether to "
                "retry, because only the host knows whether this pool is still reachable.",
                entry.request_id, entry.layer, departure.parked.timeout_s, entry.length,
            )


def serve(
    forward: Callable[[torch.Tensor, int], torch.Tensor],
    host: str,
    port: int,
    min_batch: int,
    max_wait_s: float,
    device: torch.device | str,
    ready: threading.Event | None = None,
    attention=None,
    cache=None,
    park_timeout_s: float | None = None,
) -> Departure:
    """Run a pool until the process is killed. Returns the departure thread for inspection.

    `attention` is a SweepService when the pool also holds the KV cache. Without it the pool
    answers feed-forward frames only, which is what it did before the cache moved.
    """
    departure = Departure(forward, min_batch, max_wait_s, device)
    if cache is not None and park_timeout_s is not None:
        from sglang.srt.afd.parked_sweeps import ParkedSweeps

        departure.parked = ParkedSweeps(timeout_s=park_timeout_s)
        threading.Thread(target=_expire_parked, args=(departure,), daemon=True,
                         name="afd-pool-park-timeout").start()
    departure.attention = attention
    departure.cache = cache
    departure.start()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(64)
    bound = listener.getsockname()
    logger.info("afd pool listening on %s:%s, min_batch=%s", bound[0], bound[1], min_batch)
    if ready is not None:
        ready.port = bound[1]
        ready.set()

    def handle(sock: socket.socket) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                frame = decode(sock)
                if frame is None:
                    return
                if departure.answer_directly(frame, sock):
                    continue
                departure.offer(frame, sock)
        finally:
            sock.close()

    while True:
        conn, _ = listener.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()
