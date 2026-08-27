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

## The batch is re-formed where latency VARIES, and nowhere else

That is the whole cutting rule, and it decides both where a call boundary goes and how long a bus
may be held. Measured on this model:

    feed-forward             361 us      context-free
    linear attention         200 us      context-free
    softmax attention         19 us at 1k context -> 2397 us at 128k

Only the last one moves, and it is the one stage that stays on the HOST. So the pool's side of
every boundary is context-free, and the caller with the long context is simply LATE for the next
departure rather than slow inside one. The bus leaves on its timeout with whoever is aboard and
the late rider takes the next: riders disperse at the destination and re-form for the following
call, which is what makes one pool able to serve a 1k request and a 128k one without the short one
waiting out the long one's context.

A departure that waited for a late rider would be correct, and every aggregate would look
identical -- throughput, riders histogram, mean latency -- while every short-context caller paid
the longest caller's context length. `test_afd_async_pool.py` asserts it by order for that reason.

With ONE host the histogram is all ones and this rule buys nothing: a host sends one frame a layer
for its whole batch, so there is a single rider by construction. It becomes live with several
hosts against one pool, and with two batches staggered on one host -- see `staggered.py`.
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from collections.abc import Callable

import torch

from sglang.srt.afd.boarding import ready_to_depart
from sglang.srt.afd.meter import Meter
from sglang.srt.afd.protocol import (
    INBOUND_OPS,
    OP_FFN,
    OP_HELLO,
    OP_KVPROJ,
    OP_LINEAR,
    OP_NAMES,
    OP_RELEASE,
    OP_STATE_UPDATE,
    OP_SWEEP,
    Frame,
    decode,
    send_frame,
    unpack_positions,
)

logger = logging.getLogger(__name__)


# How long the span thread waits for the first half of a span's reply to reach the wire before
# calling it a failure. It is normally set inside the last feed-forward, so any real wait here
# means the sending thread is stuck rather than slow -- generous enough never to fire on a busy
# pool, short enough that a stuck one fails instead of hanging every caller behind it.
# Ops whose departure calls BACK to the caller. They are never departed on the thread that owns
# the caller's socket, because that thread is the only reader of it.
#
# A LAYER is one of them and was left out when it was added: a linear layer run here reads a
# recurrent state the caller holds, exactly as a span does, and asking for it from the connection
# thread makes that thread wait for a message only it can receive. The reading is "no state
# reading for request 3 layer 0 within 30.0s", which names the far end and blames it, and the far
# end answered immediately.
CALLS_BACK_OPS: set = set()

# Which departure serves which op. A table rather than a chain of `if`s, so that a cut this pool
# was not built for can add its own without editing this file -- which is the point: the group cut
# is the derived line's, and the ops it added live here today only because there was nowhere else
# to put them.
#
# A handler takes (departure, layer, op, riding) and answers every rider itself. The op is passed
# even to handlers that serve exactly one, because a handler registered for several -- the span's
# three -- needs to know which one boarded.
_DEPARTURES: dict = {}


def register_departure(op: int, handler, *, calls_back: bool = False) -> None:
    """Claim an op for a departure handler. Once, and by whoever owns the arithmetic.

    `calls_back` says the handler asks the caller for something mid-run -- a recurrent state, a
    cache read. Those are never departed on the thread that owns the caller's socket, because that
    thread is the only reader of it, and one was left out of that set once: the reading was "no
    state reading for request 3 layer 0 within 30.0s", which names the far end and blames it, and
    the far end had answered immediately. Declared here so the two facts about an op arrive
    together instead of in two places that can disagree.
    """
    if op in _DEPARTURES:
        raise ValueError(
            f"op {OP_NAMES.get(op, op)} already has a departure handler. Import order would "
            f"otherwise decide which arithmetic a caller got, and nothing in the reply would "
            f"record which."
        )
    _DEPARTURES[op] = handler
    if calls_back:
        CALLS_BACK_OPS.add(op)


# How long a departure waits for the caller to answer a state read. Defined HERE, beside
# `_await_reading`, which is the only thing that uses it -- it used to be STATE_READING_TIMEOUT_S
# and moved out with the span code in #76, leaving this file reading a name bound nowhere. Every
# path through `_await_reading` on this branch would have raised NameError instead of waiting.
STATE_READING_TIMEOUT_S = 30.0


class HostDeparted(RuntimeError):
    """Every rider of a departure belonged to a host that has disconnected.

    Raised by a collect side so a departure can be abandoned whole: there is nobody left
    to reply to, and serving it would wait on readings that can never arrive. A MIXED
    departure never raises this -- a dead rider's reading is zero-filled and its reply
    dropped, so the living riders are served on time.
    """


_NAMESPACES: dict = {}


def namespace_of(sock) -> int:
    """A stable high-bit namespace for one connection's row ids, assigned on first sight.

    Two hosts' schedulers hand out row ids from similar counters, and every pool-side table
    keyed by a bare id would let one host's request continue from another's state, fluently.
    40 bits clear the largest row id sglang hands out by orders of magnitude; the namespace
    never travels -- replies and callbacks are built per rider from the rider's own frames --
    so only this side's keying widens.
    """
    key = id(sock)
    got = _NAMESPACES.get(key)
    if got is None:
        got = _NAMESPACES[key] = (len(_NAMESPACES) % 65536) << 40
    return got


class Departure(threading.Thread):
    """Collects frames and hands whole batches to the feed-forward."""

    def __init__(
        self,
        forward: Callable[[torch.Tensor, int], torch.Tensor],
        min_batch: int,
        max_wait_s: float,
        device: torch.device | str,
        arrangement: float = 0.0,
        pushed: torch.Tensor | None = None,
    ):
        super().__init__(name="afd-pool-departure", daemon=True)
        self.forward = forward
        # broadcast at the HELLO so a caller can refuse a pool configured differently. A parameter
        # rather than a global read: reaching for the server args from a constructor has broken
        # this branch's pool tests before.
        self.arrangement = arrangement
        # the pool's pushed configuration, already encoded (pushed_config.encode_config), for the
        # same reason `arrangement` is a parameter. None -- a pool assembled outside the
        # composition root, which is every direct construction in a test -- answers the HELLO in
        # the old one-tensor shape, and a host's adoption path refuses that pool by name.
        self.pushed = pushed
        # sockets whose host has disconnected, by id. A reading owed by one is zero-filled,
        # its queued riders are dismissed at the next re-form, and its rows are released --
        # the pool serves on, which is the property the whole arrangement rests on.
        self._dead: set[int] = set()
        # set by serve() when the pool also holds the KV cache; None means feed-forward only
        # set when this process is the CACHE pool of a two-pool split: it answers sweeps and
        # appends and holds no weights at all
        # set alongside `cache` when sweeps may arrive before the appends they read. None keeps
        # the pre-length behaviour, where a sweep is answered against whatever is held and the
        # transport is trusted to have ordered it.
        # An extra runner this pool serves beyond the feed-forward, supplied by whoever asked
        # for a cut this pool was not written for. None -- always, on standard AFD -- means the
        # feed-forward is all this pool does, and any op nobody registered a departure for is
        # refused by name rather than half-served.
        self.runner = None
        # There is no field here for recurrent states, deliberately. See OP_LINEAR in the protocol:
        # a pool that holds per-request state stops being stateless and can no longer be released
        # between one request's own calls, which is the property this whole arrangement rests on.
        # The states live on the caller and are read back across the wire.
        self.meter = Meter()
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
        # state readings filed by the connection thread, keyed by (socket, request, layer)
        self._readings: dict = {}
        # when each callback went out, and when its answer was filed by the connection thread. The
        # gap between them is the round trip; the gap between filing and the waiter waking is this
        # process's own scheduling. Measured on the deployment: 3.457 ms wire against 0.052 ms
        # wakeup, which is what said the remaining work is not in this process.
        self._sent_at: dict = {}
        self._filed_at: dict = {}
        self._wire_s = 0.0
        self._wake_s = 0.0
        self._waits = 0
        self._state_read_s = 0.0
        self._state_reads = 0
        # deferred state updates on their way to the caller. Nothing waits for a reply to one --
        # the protocol files OP_STATE_UPDATE off the critical path -- but the SEND was still
        # inline, and `protocol._payload_of` copies with a blocking `.to("cpu")`, so every one of
        # them synchronised this pool's stream in the middle of a layer.
        self._outbox: queue.Queue = queue.Queue()
        # ONE lock for every thread that writes a caller's socket: the departure thread's replies
        # and the sender thread's updates. Two `sendmsg` calls interleaved on one socket produce a
        # byte stream neither end can parse.
        self._wire_lock = threading.RLock()
        self._sender = threading.Thread(
            target=self._drain_outbox, name="afd-pool-updates", daemon=True
        )
        self._sender.start()
        # where to write the riders histogram, or None to keep it in memory only
        self.riders_path: str | None = None

    def capabilities(self) -> int:
        """What this pool serves, as a bitmask on the wire.

        1  feed-forward
        2  a cache: sweeps
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
        if self.runner is not None:
            bits |= self.runner.capability_bit
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
            send_frame(
                sock,
                Frame(
                    frame.request_id,
                    0,
                    # two numbers and a dict: what this pool SERVES, the arrangement word both
                    # ends must agree on (see `arms.arrangement_word` -- this file does not know
                    # what it means and does not need to), and the pool's pushed configuration,
                    # which is the word the host adopts instead of configuring itself.
                    (
                        torch.tensor(
                            [[float(self.capabilities()), float(self.arrangement)]]
                        ),
                    )
                    + ((self.pushed,) if self.pushed is not None else ()),
                    OP_HELLO,
                ),
            )
            return True
        if frame.op == OP_LINEAR:
            # FIRST, above every other branch, because the alternative is not "unserved" -- an op
            # this function returns False for falls through to `offer` and is queued as a
            # feed-forward, then answered with something nobody asked for. A reserved op has to be
            # refused where nothing can reach past it.
            raise ConnectionError(
                "a LINEAR frame reached this pool. That op is RESERVED and not served: it runs a "
                "linear layer here with the request's recurrent state HELD here, and a pool "
                "holding per-request state can no longer be released between one request's own "
                "calls -- which is the property this arrangement is built on. A host that wants "
                "its linear layers run here sends LAYER or SPAN, which leave the state on the "
                "caller and read it back across the wire."
            )
        if self.runner is not None and frame.op == OP_RELEASE:
            # The op existed, with the comment that says exactly why -- "sglang reuses slots and
            # the next one is not this one" -- and it reached the KV cache and the attention holder
            # and never the recurrent state. A KV cache survives that: a length of zero already
            # excludes stale positions. A recurrent state has no length. Whatever is in the buffer
            # IS the history, so the second request through a slot is conditioned on the first
            # one's prompt, fluently, with nothing raising.
            # On `states`, which is the object that would hold it. `LinearRunner` has never had
            # a `release` -- the call reached for one and this branch is only entered under
            # `--afd-pool-linear`, so it raised AttributeError on the first request that ENDED,
            # long after the arrangement looked like it was working.
            #
            # Under this arrangement the answer is always zero: the recurrent state is the
            # caller's and nothing here takes a slot. The frame is still answered, because the
            # caller waits for it -- and `states` is asked rather than the answer assumed, so an
            # arrangement that does hold something here would report it.
            dropped = self.runner.states.release(
                namespace_of(sock) | int(frame.request_id)
            )
            send_frame(
                sock,
                Frame(
                    frame.request_id,
                    frame.layer,
                    (torch.tensor([[float(dropped)]]),),
                    OP_RELEASE,
                ),
            )
            return True
        return False

    def _write_riders(self) -> None:
        import json

        recent = self.departures[-2000:]
        try:
            with open(self.riders_path, "w") as f:
                json.dump(_riders_summary(recent), f, indent=2)
        except OSError:  # a full disk must not stop the pool serving
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
        # A SPAN is never departed on this thread. It calls back to the caller mid-run, and the
        # caller's answer arrives on this socket -- which this thread is the only reader of. A
        # connection thread that departs its own span becomes the thread waiting for a message
        # only it can receive, and with min_batch=1, which is what the deployment runs, every
        # offer completes a batch so it happens on the first token. It is a deadlock, not a race:
        # no traceback, no wrong value, and a watchdog timeout that names none of this.
        calls_back = frame.op in CALLS_BACK_OPS
        with self._cond:
            queue = self._waiting.setdefault(frame.layer, [])
            queue.append((frame, sock))
            # the same question `_ready_layer` asks, with no wait behind it -- a frame that has
            # just arrived cannot be overdue, and a second rule here would be a second thing to
            # keep in step with the first
            if not calls_back and ready_to_depart(
                waiting=len(queue),
                min_batch=self.min_batch,
                waited_s=0.0,
                max_wait_s=self.max_wait_s,
            ):
                riding = self._waiting.pop(frame.layer)
                self._first_seen.pop(frame.layer, None)
            else:
                self._first_seen.setdefault(frame.layer, time.perf_counter())
                self._cond.notify()  # the departure thread takes it, and reads nothing
        if riding is not None:
            riding = self._without_the_departed(riding)
            if riding:
                self._depart(frame.layer, riding)
            self._listen_again()

    def _listen_again(self) -> None:
        """Having sent, look at the queue again before going back to the socket.

        The thread that just answered a call is the thread most likely to find another one ready:
        the callers it answered are, at this instant, computing their own attentions, and whoever
        finished theirs during this departure is already at the next stop. Going straight back to
        `recv` leaves them to the timer thread, which is up to a quarter of `max_wait_s` away for
        work that is ready NOW.

        A departure that CALLS BACK is never taken here, for the reason `offer` gives: this thread
        owns the caller's socket, and a callback made from it waits for a message only it can
        receive.
        """
        while True:
            riding = None
            with self._cond:
                now = time.perf_counter()
                for layer, queue in self._waiting.items():
                    if not queue or any(f.op in CALLS_BACK_OPS for f, _ in queue):
                        continue
                    if ready_to_depart(
                        waiting=len(queue),
                        min_batch=self.min_batch,
                        waited_s=now - self._first_seen.get(layer, now),
                        max_wait_s=self.max_wait_s,
                    ):
                        riding, ready = self._waiting.pop(layer), layer
                        self._first_seen.pop(layer, None)
                        break
            if riding is not None:
                riding = self._without_the_departed(riding)
                if not riding:
                    continue
            if riding is None:
                return
            self._depart(ready, riding)

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def _ready_layer(self, now: float) -> int | None:
        """Ask the boarding question again, of every stop, on every pass. See `boarding.py`: the
        answer is not computable in advance because the cadence belongs to the callers'
        contexts and those are not uniform even inside one batch."""
        for layer, queue in self._waiting.items():
            if ready_to_depart(
                waiting=len(queue),
                min_batch=self.min_batch,
                waited_s=now - self._first_seen.get(layer, now),
                max_wait_s=self.max_wait_s,
            ):
                return layer
        return None

    def run(self) -> None:
        """The timeout half. The full-batch half is taken by whichever thread offered the frame."""
        from sglang.srt.afd.serve_clock import hold_this_thread

        hold_this_thread()
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
            try:
                self._depart(layer, riding)
            except BaseException as e:  # noqa: BLE001 -- reported, never swallowed
                # A departure that raised used to kill this thread silently, and everything after
                # it hung with no message anywhere. The riders are failed by name instead: their
                # callers see a closed socket, which is a fact they can act on, and the reason is
                # logged once where an operator will find it.
                logger.exception(
                    "afd pool: layer %s departure failed for %s rider(s); failing them rather "
                    "than losing the departure thread",
                    layer,
                    len(riding),
                )
                for _, sock in riding:
                    # SHUTDOWN, not just close. This socket's own connection thread is blocked
                    # in `decode` on it right now, so it holds a reference: `close()` drops one
                    # reference and sends NO FIN, and the caller waits forever for a reply that
                    # was already given up on. Measured both ways -- with `close()` alone the
                    # far end's `_closed` was still False two seconds later; with a shutdown
                    # first it was True immediately. That difference is the whole distance
                    # between "refused by name" and a watchdog timeout at 300 s.
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:  # already gone, which is the outcome wanted anyway
                        pass
                    try:
                        sock.close()
                    except OSError:
                        pass

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
        op = ops.pop()
        handler = _DEPARTURES.get(op)
        if handler is not None:
            try:
                return handler(self, layer, op, riding)
            except OSError as e:
                # a rider's socket died mid-serve; its host's settling happens on the
                # connection thread. The batch is lost, the pool is not.
                logger.warning(
                    "afd pool: a departure for layer %s lost its wire mid-serve (%r). "
                    "The pool serves on.",
                    layer,
                    e,
                )
                return None
            except HostDeparted as e:
                # nobody is left to reply to; the batch is abandoned whole and the pool
                # serves the next departure. Warning, not error: the pool outliving its
                # callers is the design, and this line is how an operator reads about it.
                logger.warning(
                    "afd pool: a departure for layer %s was abandoned -- %s. "
                    "The pool serves on.",
                    layer,
                    e,
                )
                return None
        if op != OP_FFN:
            # Refused rather than served. An op with no handler used to fall through to the
            # feed-forward, which does not fail -- it answers whoever asked with arithmetic they
            # did not ask for, in a reply that looks like a reply. That mattered the moment the
            # group cut's handlers moved out of this file: a pool built from standard AFD alone
            # now has no entry for a SPAN, and the choice is between saying so and running four
            # layers of feed-forward against a frame that meant something else.
            raise ConnectionError(
                f"op {OP_NAMES.get(op, op)} reached this pool and nothing here serves it. This "
                f"pool serves the feed-forward; a cut that needs more registers its own departure "
                f"through afd.pool_server.register_departure, and the package that would have "
                f"done so is not installed here."
            )
        return self._depart_feed_forward(layer, riding)

    def _count_wait(self, wire_s: float, wake_s: float) -> None:
        """The two halves of a callback's cost, reported every 500.

        `wire_s` is send to filed: the hop out, the far end's work, the hop back, and the
        connection thread reading the socket. `wake_s` is filed to waking: purely this process --
        `notify_all` to the waiter running, which is the GIL and the condition variable.

        Separated because the two have different answers. A large wake is fixable here; a large
        wire is what a cross-machine deployment costs and belongs in a document as a limit rather
        than as a defect.
        """
        self._wire_s += wire_s
        self._wake_s += wake_s
        self._waits += 1
        if self._waits % 500:
            return
        n = self._waits
        logger.info(
            "afd pool: %s callback(s) -- %.3f ms wire, %.3f ms wakeup",
            n,
            1e3 * self._wire_s / n,
            1e3 * self._wake_s / n,
        )

    def post_unawaited(
        self, sock, request_id: int, layer_id: int, tensors: tuple, op: int
    ) -> threading.Event:
        """Queue a frame nobody waits for a REPLY to, instead of sending it here.

        See `_post_update` for why queueing is worth it at all. Any op whose reply nothing blocks
        on can take this path, and the state update is one: its deadline is the next token.

        Returns an event set once the frame is on the wire, for a caller that has to order
        something else AFTER it. Waiting on that event rather than on the queue is the difference
        between "my frame has gone" and "every frame has gone" -- and the queue also carries the
        deferred updates, which were put there precisely so that nobody waits for them. Joining it
        would have dragged them back onto the critical path, MEASURED at 1.25 ms a span.
        """
        staged = tuple(t.detach().to("cpu", non_blocking=True) for t in tensors)
        copied = torch.cuda.Event()
        copied.record()
        sent = threading.Event()
        self._outbox.put((copied, sock, Frame(request_id, layer_id, staged, op), sent))
        return sent

    def _post_update(
        self, sock, request_id: int, layer_id: int, tensors: tuple
    ) -> None:
        """Queue a state update instead of sending it here.

        Nothing waits for a reply to an update, but "no reply awaited" is not "does not block".
        `send_frame` reaches `_payload_of`, whose plain `.to("cpu")` synchronises the stream, so
        the pool stopped computing, waited for the copy, wrote the socket, and only then went on
        -- once per linear-attention layer per token.

        The deadline on an update is generous: it has to land before the next token reads that
        layer, a whole step away. So the copy is staged non-blocking behind an event and one
        sender thread does the waiting. FIFO, because two updates to one slot that overtook each
        other would apply this token's key before the last one's and the state would be wrong
        with nothing raising.
        """
        self.post_unawaited(sock, request_id, layer_id, tensors, OP_STATE_UPDATE)

    def _drain_outbox(self) -> None:
        """Send queued updates once their copies land. One thread, so the order is the layer's."""
        while True:
            copied, sock, frame, sent = self._outbox.get()
            try:
                copied.synchronize()
                with self._wire_lock:
                    send_frame(sock, frame)
            except OSError:
                logger.warning(
                    "caller for request %s layer %s went away before its queued frame",
                    frame.request_id,
                    frame.layer,
                )
            except Exception:
                logger.exception("a queued frame was lost")
            finally:
                # Set whatever happened. A caller waiting to order something after this frame is
                # waiting for it to have been ATTEMPTED, not to have succeeded -- a socket that
                # went away fails the next call too, and hanging here would turn a lost caller
                # into a stuck departure.
                sent.set()

    def host_departed(self, sock) -> None:
        """A host's connection is gone: dismiss its riders, free what it held, serve on.

        Called by the connection thread on its way out. Everything a departure might still
        want from this host is settled HERE, once, so no serving thread ever waits on it:
        owed readings are dropped (the waiters zero-fill), queued riders are dismissed, and
        the rows its namespace held are released. Logged as a warning with the counts,
        because a host vanishing is an event an operator reads about -- and never an error,
        because the pool outliving its callers is the design.
        """
        sid = id(sock)
        with self._cond:
            self._dead.add(sid)
            owed = [k for k in self._readings if k[0] == sid]
            for k in owed:
                self._readings.pop(k, None)
                self._filed_at.pop(k, None)
            dismissed = 0
            for layer, queue in list(self._waiting.items()):
                kept = [(f, so) for f, so in queue if id(so) != sid]
                dismissed += len(queue) - len(kept)
                if kept:
                    self._waiting[layer] = kept
                else:
                    self._waiting.pop(layer, None)
                    self._first_seen.pop(layer, None)
            self._cond.notify_all()
        released = 0
        runner = getattr(self, "runner", None)
        states = getattr(runner, "states", None)
        release_namespace = getattr(states, "release_namespace", None)
        if release_namespace is not None:
            released = release_namespace(namespace_of(sock))
        forget = getattr(runner, "forget_namespace", None)
        if forget is not None:
            released += forget(namespace_of(sock))
        logger.warning(
            "afd pool: a host departed -- %d owed reading(s) dropped, %d queued rider(s) "
            "dismissed, %d row(s) released. The pool serves on.",
            len(owed),
            dismissed,
            released,
        )

    def _without_the_departed(self, riding):
        """The riders whose hosts are still here; the dead are dismissed with a line."""
        kept = [(f, so) for f, so in riding if id(so) not in self._dead]
        if len(kept) != len(riding):
            logger.warning(
                "afd pool: %d rider(s) dismissed at the stop; their host departed.",
                len(riding) - len(kept),
            )
        return kept

    def file_reading(self, sock, frame) -> None:
        """A reading arriving on the connection thread, for whichever span is waiting on it."""
        arrived = time.perf_counter()
        with self._cond:
            key = (id(sock), frame.request_id, frame.layer)
            self._readings[key] = frame.tensor
            self._filed_at[key] = arrived
            self._cond.notify_all()

    def _depart_feed_forward(
        self, layer: int, riding: list[tuple[Frame, socket.socket]]
    ) -> None:
        started = time.perf_counter()
        widths = {f.tensor.shape[1] for f, _ in riding}
        if len(widths) != 1:
            raise RuntimeError(f"layer {layer} departure mixes hidden widths {widths}")
        counts = [f.tensor.shape[0] for f, _ in riding]
        # one rider is the common case on a single-caller pool, and torch.cat on a one-element
        # list still copies the whole frame
        joined = (
            riding[0][0].tensor
            if len(riding) == 1
            else torch.cat([f.tensor for f, _ in riding], dim=0)
        )
        batch = joined.to(self.device)
        with self.meter.timed("work"):
            out = self.forward(batch, layer)
            # The kernels are launched asynchronously, so timing the call alone measures the
            # LAUNCH. Without this the forward's real time was charged to the reply, because
            # `_payload_of` does `.to("cpu")` and that synchronises -- and the first reading said
            # "the reply costs three times the work" when the reply was mostly waiting for the
            # work. The synchronise costs nothing that was not already going to be paid at the
            # copy; it only decides which phase pays it.
            if out.is_cuda:
                torch.cuda.current_stream().synchronize()
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
                with self.meter.timed("wire_out"):
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
        "seconds_median": (
            sorted(d["seconds"] for d in departures)[len(departures) // 2]
            if departures
            else 0.0
        ),
    }


def route_frame(departure: Departure, frame, sock) -> None:
    """Decide what an arriving frame IS, and hand it to whoever wants it.

    Three kinds arrive on one socket and nothing but the opcode separates them: a reply to a call
    this pool made, a question answerable without a batch, and a rider boarding a bus.

    A REPLY is filed rather than read by the span that wants it, because two readers on one socket
    deadlock the moment the departure is taken by the timer thread rather than by the connection
    thread. It is recognised by INBOUND_OPS -- the same set the far end answers -- and NOT by a
    literal. A literal was here, `== OP_STATE_READ`, and adding OP_STATE_SCAN to the protocol left
    a scan's reply falling through to `offer`, where it boarded as if it were a feed-forward
    request. What raised was `mat1 and mat2 shapes cannot be multiplied (122x6144 and 5120x34816)`
    six frames deep inside a MoE layer, naming nothing in this file: 6144 is 48 value heads of
    128, which is what a reading is and what a hidden state is not.

    Extracted so the tests run THIS rather than a copy of it. The copy in the test harness carried
    the same literal, so the case that should have caught the scan passed against a fake with the
    identical bug.
    """
    if frame.op in INBOUND_OPS:
        departure.file_reading(sock, frame)
        return
    if departure.answer_directly(frame, sock):
        return
    departure.offer(frame, sock)


def serve(
    forward: Callable[[torch.Tensor, int], torch.Tensor],
    host: str,
    port: int,
    min_batch: int,
    max_wait_s: float,
    device: torch.device | str,
    ready: threading.Event | None = None,
    runner=None,
    arrangement: float = 0.0,
    pushed=None,
    after_built=None,
) -> Departure:
    """Run a pool until the process is killed. Returns the departure thread for inspection."""
    departure = Departure(
        forward, min_batch, max_wait_s, device, arrangement=arrangement, pushed=pushed
    )
    departure.runner = runner
    if after_built is not None:
        # the accept loop below never returns, so anything the composition root wants
        # wired to the departure -- the lane's serving loop, the riders path -- happens
        # HERE or never. Two lines in `run_pool` after this call were dead for exactly
        # that reason, silently, until the lane's missing loop hung a host.
        after_built(departure)
    departure.start()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(64)
    bound = listener.getsockname()
    logger.info(
        "afd pool listening on %s:%s, min_batch=%s", bound[0], bound[1], min_batch
    )
    if ready is not None:
        ready.port = bound[1]
        ready.set()

    def handle(sock: socket.socket) -> None:
        from sglang.srt.afd.serve_clock import hold_this_thread

        # at min_batch 1 this thread IS the serving thread ("the full-batch half is
        # taken by whichever thread offered the frame"), so the clock hold, if the
        # operator asked for one, belongs to it
        hold_this_thread()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # a NEW connection may land on a recycled object id -- id() is an address, and a
        # dead socket's address is exactly what the allocator hands out next -- so arrival
        # clears the id from the dead set before any rider of this connection can be
        # mistaken for a departed one's. Found live: a restarted host's every frame was
        # dismissed at the stop, silently, and the host waited out its watchdog.
        with departure._cond:
            departure._dead.discard(id(sock))
        try:
            while True:
                with departure.meter.timed("wire_in"):
                    frame = decode(sock)
                if frame is None:
                    return
                route_frame(departure, frame, sock)
                departure.meter.call()
        except OSError:
            # the host's side died mid-frame; the settling below is the same either way
            return
        finally:
            departure.host_departed(sock)
            sock.close()

    while True:
        conn, _ = listener.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()
