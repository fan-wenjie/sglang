"""The host's side of the split: issue a feed-forward, keep working, collect it later.

The three calls are deliberately separate, and the separation IS the arrangement:

    issue(...)     hand the pool a layer's normalised residual and get a handle back. Returns a
                   handle, never a tensor -- a function that returned the answer would have had to
                   wait for it, which is the synchronous arrangement wearing this one's name.
    (sweep)        the caller runs its attention here, with the pool call in flight
    collect(h)     block until the reply for that handle has landed

Between `issue` and `collect` the pool is not reserved for this request. It is stateless, so
another request's call may be served in between; that is the whole reason to pool it rather than
give each request its own.

A single receiver thread owns the socket and files replies into slots keyed by (request, layer).
The key is a pair on purpose: two requests in flight at the same layer would overwrite one slot,
and the result is a model that still reads fluently from a subtly wrong distribution.
"""

from __future__ import annotations

import collections
import logging
import queue
import socket
import threading
import time
from typing import NamedTuple

import torch

from sglang.srt.afd.protocol import (
    CLOSE,
    INBOUND_OPS,
    OP_FFN,
    OP_HELLO,
    OP_NAMES,
    Frame,
    decode,
    send_frame,
)

logger = logging.getLogger(__name__)


def _claimed_lane() -> int:
    """Which lane slot this host claims, for every HELLO it sends. 0 when it cannot tell.

    Answers 0 rather than raising because a process with no published runtime -- a tool, a test
    -- is not a host claiming anything, and 0 is exactly what such a caller sent before the claim
    existed.
    """
    try:
        from sglang.srt.runtime_context import get_disagg

        return int(get_disagg().afd_host_lane or 0)
    except Exception:  # noqa: BLE001 -- no runtime, no claim
        return 0

# how many call records to keep. Enough for several report windows, bounded so a long run does
# not accumulate one dict per pool call forever.
WAIT_HISTORY = 8192


# How many inbound callbacks between reports. Counted ACROSS ops but reported per op, so a quiet
# op does not have to wait for its own five hundred to be seen.
INBOUND_REPORT_EVERY = 500


def _inbound_averages(records: dict) -> dict:
    """Turn per-op counts and sums into per-op averages, each over its own denominator.

    A free function so that the caller holding `_cond` can hand over a snapshot and format it
    outside the lock. It divides only ever within one op's record, which is the property the whole
    counter exists for -- the arithmetic is trivial and the pairing is not.

    `waited` is time in the queue, and it was invisible until it mattered: `served` brackets the
    handler alone, so a frame that sat behind another read as though the far end were fast and the
    wire were slow. Under an arrangement where one callback waits on another, that wait IS the
    cost, and no counter had it.
    """
    return {
        OP_NAMES.get(op, str(op)): {
            "calls": calls,
            "serve_ms": 1e3 * serve_s / calls,
            "reply_ms": 1e3 * reply_s / calls,
            "waited_ms": 1e3 * waited_s / calls,
        }
        for op, (calls, serve_s, reply_s, waited_s) in records.items()
        if calls
    }


class Handle(NamedTuple):
    """What `issue` returns. `issued_at` is kept so a run can prove the overlap happened.

    `op` is part of the identity, not decoration. A cache pool answers a sweep and acknowledges an
    append for the SAME (request, layer), and a slot table keyed by those two alone hands one
    caller the other's answer -- an append's one-tensor acknowledgement arriving where a sweep's
    (output, log partition) was expected. That is what "not enough values to unpack" looked like
    the first time this ran with both in flight.
    """

    request_id: int
    layer: int
    issued_at: float
    op: int = 0

    @property
    def key(self) -> tuple[int, int]:
        return (self.request_id, self.layer)

    @property
    def reply_key(self) -> tuple[int, int, int]:
        return (self.request_id, self.layer, self.op)


class PoolClosed(RuntimeError):
    """The pool went away while a call was outstanding. Never silently returns zeros for it."""


class PoolClient:
    """One connection to one pool, shared by every request on this host.

    ## A pool that dies must not take the host with it

    Without reconnection a pool restart is fatal: every call after it raises, the scheduler dies,
    and a server that was serving hundreds of requests stops because one of its two processes was
    replaced. With it, the calls that were in flight still fail -- their answers are gone and
    inventing one would be worse -- and the next call reconnects.

    That distinction is the contract: **in-flight work fails loudly, future work recovers.** A
    client that retried the in-flight frames would be guessing at whether the pool had already
    applied them, which for a stateful pool means a key appended twice.
    """

    def __init__(
        self,
        address: str,
        connect_timeout_s: float,
        reconnect: bool = True,
        max_reconnects: int = 8,
    ):
        host, _, port = address.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"--afd-pool-addr wants HOST:PORT, got {address!r}")
        self.address = address
        self._host, self._port = host, int(port)
        self._connect_timeout_s = connect_timeout_s
        self._reconnect = reconnect
        self._max_reconnects = max_reconnects
        self.reconnects = 0
        self._sock = self._open()
        self._send_lock = threading.Lock()
        self._cond = threading.Condition()
        self._slots: dict[tuple[int, int], torch.Tensor] = {}
        # multi-tensor replies live apart from the one-tensor feed-forward slots, so a reply of
        # the wrong shape cannot be handed to a caller expecting the other
        self._replies: dict[tuple[int, int], tuple] = {}
        self._hello: float | None = None
        # the pool's pushed configuration, parsed from the HELLO reply when the pool sends
        # one; None until then and forever against a pool that pushes nothing
        self.pool_config: dict | None = None
        self._arrangement: float | None = None
        # set by a host that holds state the pool has to call back for. None means this client
        # only ever receives replies, and an inbound request against it is a configuration
        # disagreement rather than a protocol error -- so it is named as one.
        self.serve = None
        self._failure: BaseException | None = None
        # bounded: an hour of decode is millions of entries, and overlap_report used to slice a
        # list that only ever grew. The report is windowed anyway.
        self._waits: collections.deque = collections.deque(maxlen=WAIT_HISTORY)
        # This side's share of the callbacks the pool blocks on, kept PER OP. A single pair of
        # running totals here is what produced -- and then had to retract -- a "2.07 ms of host
        # compute". See `_count_inbound`.
        self._inbound: dict[int, list] = {}
        self._closed = False
        self._issues = collections.deque(maxlen=4096)
        # Inbound callbacks are answered on their OWN thread, and the reader only decodes.
        # ONE thread, not a pool: a decode step must read this layer's state and then update it,
        # alternating, and two workers could answer an update before the read it follows. The
        # order a frame arrives in is the order it is answered in, exactly as when the reader
        # answered them itself.
        self._inbox: queue.Queue = queue.Queue()
        self._inbound_worker = threading.Thread(
            target=self._serve_inbound, name="afd-pool-inbound", daemon=True
        )
        self._inbound_worker.start()
        self._receiver = threading.Thread(
            target=self._receive, name="afd-pool-recv", daemon=True
        )
        self._receiver.start()

    def _open(self):
        sock = socket.create_connection(
            (self._host, self._port), timeout=self._connect_timeout_s
        )
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock

    def reconnect(self) -> bool:
        """Replace a dead connection. Returns False when reconnection is off or exhausted.

        Every call that was outstanding is failed first. Their answers went with the old process,
        and a client that retried them would be guessing at whether the pool had already applied
        them -- which for a pool holding histories means a key appended twice, a past with one
        token in it twice, and fluent output from it.
        """
        if not self._reconnect or self.reconnects >= self._max_reconnects:
            return False
        with self._cond:
            outstanding = len(self._slots) + len(self._replies)
            self._slots.clear()
            self._replies.clear()
        try:
            sock = self._open()
        except OSError as e:
            logger.warning(
                "afd: could not reconnect to the pool at %s: %s", self.address, e
            )
            return False
        with self._cond:
            old, self._sock = self._sock, sock
            self._failure, self._closed = None, False
            self.reconnects += 1
        try:
            old.close()
        except OSError:
            pass
        logger.warning(
            "afd: reconnected to the pool at %s (attempt %s). %s call(s) that were outstanding "
            "were failed rather than retried: their answers went with the old process, and "
            "resending them would risk applying an append twice.",
            self.address,
            self.reconnects,
            outstanding,
        )
        # Inbound callbacks are answered on their OWN thread, and the reader only decodes.
        # ONE thread, not a pool: a decode step must read this layer's state and then update it,
        # alternating, and two workers could answer an update before the read it follows. The
        # order a frame arrives in is the order it is answered in, exactly as when the reader
        # answered them itself.
        self._inbox: queue.Queue = queue.Queue()
        self._inbound_worker = threading.Thread(
            target=self._serve_inbound, name="afd-pool-inbound", daemon=True
        )
        self._inbound_worker.start()
        self._receiver = threading.Thread(
            target=self._receive, name="afd-pool-recv", daemon=True
        )
        self._receiver.start()
        return True

    def _serve_inbound(self) -> None:
        """Answer inbound callbacks, off the reader thread.

        Why this is not the reader's job any more, and the measurement that moved it:

            bare TCP round trip, these two machines      0.33 ms
            this side's own work per state read          0.59 ms  (0.36 served + 0.23 replying)
            what the pool measures for the same read     5.50 ms

        Everything with an owner adds to 0.92 ms. The missing 4.6 ms was this queue -- except it
        was not a queue, it was the reader thread being unavailable to DECODE while it served a
        callback. Every frame behind that callback waited, including the span replies the caller
        is blocked on and the next state read of the same span.

        `_answer_inbound` used to say the handoff was not worth a scheduler round trip because
        "the pool is blocked waiting for this and will send nothing else until it has it". The
        pool does send other things while it waits: it drains deferred state updates from a
        sender thread of its own, and an arrangement that answers part of a call before the rest
        sends that part too. They were all queueing behind a decoder that was busy answering.
        """
        while True:
            queued = self._inbox.get()
            if queued is None:
                self._inbox.task_done()
                return
            arrived, frame = queued
            waited = time.perf_counter() - arrived
            try:
                self._answer_inbound(frame, waited)
            except Exception as e:  # noqa: BLE001 -- recorded, never swallowed
                # The pool is waiting on this and has no other way to learn it failed. Closing
                # the socket is what reaches it; a silent drop is the 300 s watchdog.
                self._failure = e
                logger.error("afd host: inbound callback failed: %r", e)
                try:
                    self._sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return
            finally:
                self._inbox.task_done()

    def _answer_inbound(self, frame, waited: float = 0.0) -> None:
        """A REQUEST arriving on the reply socket, because the pool needs something this end has.

        Under the group cut the pool runs a linear layer's weights and this end holds its recurrent
        state, so the pool calls back mid-span. The call arrives interleaved with the replies this
        client is waiting for, and telling them apart is the reader's job: an inbound request
        stored in the reply table would hang the caller it was keyed as, and hand a state reading
        to whoever asked for that key.

        Answered on `_serve_inbound`'s thread, NOT the reader's. It used to be the reader's, on
        the grounds that "the pool is blocked waiting for this and will send nothing else until it
        has it, so there is no reply being delayed". That was measured false -- see
        `_serve_inbound` for the three numbers that do not add up and the 4.6 ms they left over.
        The send takes the same lock every other send takes, because a reply going out from here
        can interleave with one going out from a caller's thread.
        """
        if self.serve is None:
            raise RuntimeError(
                f"the pool sent a {OP_NAMES.get(frame.op, frame.op)} and this client has no "
                f"handler for it. The two ends disagree about who holds the recurrent state: this "
                f"one was built expecting the pool to hold it, and the pool expects this one to."
            )
        began = time.perf_counter()
        answer = self.serve(frame)
        served_at = time.perf_counter()
        if answer is None:
            self._count_inbound(frame.op, served_at - began, 0.0, waited)
            return
        with self._send_lock:
            send_frame(
                self._sock,
                Frame(frame.request_id, frame.layer, tuple(answer), frame.op),
            )
        self._count_inbound(
            frame.op, served_at - began, time.perf_counter() - served_at, waited
        )
        # After the reply, never before it. A service may park work whose result nothing in the
        # reply depends on -- the recurrent advance is the one that does -- and this is the point
        # at which the caller is already unblocked. Safe without a lock because this runs on the
        # single inbound thread, so parked work is applied before the next frame is served.
        drain = getattr(self.serve, "drain", None)
        if drain is not None:
            drain()

    def _count_inbound(
        self, op: int, serve_s: float, reply_s: float, waited_s: float = 0.0
    ) -> None:
        """This side's share of a callback the POOL is blocked on, counted PER OP.

        The pool measures 4.651 ms per state read. Its parts are this side's compute, this side's
        reply write, and the wire; only the first two are visible here, and the difference is the
        wire. That division is what the whole 44%-of-a-span figure turns on, so it has to be right.

        It was not. The first version of this kept ONE pair of running totals across every op in
        `INBOUND_OPS` -- reads, updates, scans -- and divided by the total count. Reads are the
        blocking op and there were 11,000 of them; the denominator was 30,500. The average came
        out at 2.07 ms and was reported as "host compute, 45% of the state read". Timed inside
        `_read` alone it is **0.193 ms**, the wire is 88% rather than 48%, and the optimisation
        target moved to a different machine. Nothing about the wrong number looked wrong: a mixed
        average is always plausible, and nobody asked what the denominator counted.

        So each op keeps its own count and its own sums, and every average printed here divides by
        the count of the very thing it names. An op outside `INBOUND_OPS` would be a population
        that is not a callback at all, which is why it is refused rather than folded in.
        """
        if op not in INBOUND_OPS:
            raise ValueError(
                f"{OP_NAMES.get(op, op)} is not one of the pool's callbacks and must not be "
                f"counted with them. Averaging it in is how a per-call figure stops being about "
                f"anything."
            )
        with self._cond:
            record = self._inbound.setdefault(op, [0, 0.0, 0.0, 0.0])
            record[3] += waited_s
            record[0] += 1
            record[1] += serve_s
            record[2] += reply_s
            if sum(r[0] for r in self._inbound.values()) % INBOUND_REPORT_EVERY:
                return
            records = {op: list(r) for op, r in self._inbound.items()}
        report = _inbound_averages(records)
        logger.info(
            "afd host: inbound callbacks -- %s",
            " | ".join(
                f"{name} n={r['calls']} {r['waited_ms']:.3f} ms queued, "
                f"{r['serve_ms']:.3f} ms served, {r['reply_ms']:.3f} ms replying"
                for name, r in sorted(report.items())
            ),
        )

    def inbound_report(self) -> dict:
        """What this side cost the pool, one entry an op, each with its OWN denominator.

        Public because the denominator is the point: a caller that reads `serve_ms` also reads the
        `calls` it was divided by, in the same entry, and cannot accidentally quote one op's
        average over another op's count.
        """
        with self._cond:
            records = {op: list(r) for op, r in self._inbound.items()}
        return _inbound_averages(records)

    def _receive(self) -> None:
        try:
            while True:
                frame = decode(self._sock)
                if frame is None:
                    break
                if frame.op in INBOUND_OPS:
                    self._inbox.put((time.perf_counter(), frame))
                    continue
                # A reply is filed only after every inbound frame that arrived BEFORE it has been
                # answered. The pool sends a deferred update and then the reply, and a caller that
                # collected the reply is entitled to assume the update already landed -- that is
                # the "read then update, alternating" order a decode step depends on. Moving the
                # answering to its own thread broke it, intermittently and only under load, which
                # is the shape this join exists to keep out.
                self._inbox.join()
                with self._cond:
                    if frame.op == OP_HELLO:
                        self._hello = float(frame.tensor[0, 0])
                        # second column when the pool sends one. A pool that does not is running a build
                        # from before the arrangement word existed, which is itself a disagreement worth
                        # refusing rather than defaulting past.
                        self._arrangement = (
                            float(frame.tensor[0, 1])
                            if frame.tensor.shape[1] > 1
                            else None
                        )
                        # second TENSOR when the pool pushes its configuration -- the settings
                        # this host adopts instead of configuring itself. Parsed here, judged
                        # by `pushed_config.adopt` on the install path.
                        if len(frame.tensors) > 1:
                            from sglang.srt.afd.pushed_config import decode_config

                            self.pool_config = decode_config(frame.tensors[1])
                        self._replies[(frame.request_id, frame.layer, frame.op)] = (
                            frame.tensors
                        )
                    elif len(frame.tensors) == 1 and frame.op == OP_FFN:
                        self._slots[frame.key] = frame.tensor
                    else:
                        # keyed by op as well: one (request, layer) has more than one answer in
                        # flight, and they are not interchangeable
                        self._replies[(frame.request_id, frame.layer, frame.op)] = (
                            frame.tensors
                        )
                    self._cond.notify_all()
        except (
            BaseException
        ) as e:  # noqa: BLE001 -- it is re-raised in every waiting caller
            with self._cond:
                self._failure = e
                self._cond.notify_all()
            return
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def issue(self, request_id: int, layer: int, hidden: torch.Tensor) -> Handle:
        """Send one layer's work. Returns without waiting for a reply -- but not for free.

        `issue` is timed in two parts because they have different causes and different fixes.
        Building the frame copies the hidden state to host memory, and a device-to-host copy of an
        unpinned tensor SYNCHRONISES the stream: it waits for every kernel queued before it, which
        in a decode step is this layer's whole attention. So the window this call is supposed to
        open does not start when `issue` is called; it starts when the GPU has drained.

        The measurements say a layer costs 1260 us more than the model predicts, and this is the
        first candidate. It is separated here rather than argued about.
        """
        began = time.perf_counter()
        frame = Frame.one(request_id, layer, hidden)
        built = time.perf_counter()
        with self._send_lock:
            send_frame(self._sock, frame)
        sent = time.perf_counter()
        with self._cond:
            self._issues.append({"build_s": built - began, "send_s": sent - built})
        return Handle(request_id, layer, sent)

    NEEDS_FEED_FORWARD = 1
    NEEDS_CACHE = 2
    NEEDS_KV_PROJECTION = 4
    NEEDS_SPANS = 8
    NEEDS_LINEAR_LAYERS = 16
    _NEED_NAMES = {
        1: "a feed-forward",
        2: "a cache to sweep and append",
        4: "the key and value projections",
        8: "whole spans, four layers a call",
        16: "the linear-attention layers, one a call",
    }

    def fetch_config(self, timeout_s: float = 15.0) -> dict | None:
        """One HELLO for the pool's pushed configuration. None if the pool pushes nothing.

        Separate from `require` because it runs EARLIER: which capabilities a host needs can
        depend on settings it is about to adopt, so the word has to arrive before the asking.
        """
        # The payload states which lane slot this host claims. It was a 1x1 of zeros that
        # nothing read, so a pool that predates the claim sees the same shape and a host that
        # claims nothing sends the same value it always did.
        self.collect_frame(
            self.issue_frame(0, 0, (torch.tensor([[float(_claimed_lane())]]),), OP_HELLO),
            "cpu",
        )
        return self.pool_config

    def require(
        self, needs: int, timeout_s: float = 15.0, arrangement: float | None = None
    ) -> int:
        """Ask the pool what it does, and refuse now if it is not what this host needs.

        Called once at startup. The alternative is what this cost twice: a host configured for a
        cache pool reaches one that only runs feed-forwards, sends it a frame it cannot parse, and
        reports "closed mid-call" -- a message about a socket that names neither side's
        configuration.
        """
        # Every HELLO this host sends carries the same claim: the pool records it per
        # connection, and a host whose two HELLOs disagreed would have its lane slot decided by
        # whichever arrived last. There are two of them because there are two questions -- what
        # the pool serves, and what it pushes -- and the claim belongs to the host, not to either
        # question.
        self.collect_frame(
            self.issue_frame(0, 0, (torch.tensor([[float(_claimed_lane())]]),), OP_HELLO),
            "cpu",
        )
        served = int(self._hello or 0)
        if arrangement is not None:
            self._agree_on_arrangement(arrangement)
        missing = [
            name
            for bit, name in self._NEED_NAMES.items()
            if needs & bit and not served & bit
        ]
        if missing:
            raise PoolClosed(
                f"the pool at {self.address} does not serve {', and '.join(missing)}. This host "
                f"was configured to ask it for that, so the two were started with different "
                f"roles -- most likely the pool is running without the flag that gives it a "
                f"cache. Failing here rather than at the first token."
            )
        return served

    def _agree_on_arrangement(self, mine: float) -> None:
        """Refuse a pool whose configuration disagrees with this host's, at startup.

        This file does not know what the number means -- an arm folds the settings both ends must
        share into one value and supplies the sentence when they differ. See
        `arms.arrangement_word`.

        `mine` is a PARAMETER, not read from the global server args here. Reaching for those from
        a constructor or a connection path has broken pool suites on this line repeatedly, each
        time with a message about a socket for a setting the tests are not about. The caller reads
        them once, where a command line becomes an arrangement.
        """
        from sglang.srt.afd.arms import explain_arrangement
        from sglang.srt.runtime_context import get_server_args

        theirs = self._arrangement
        if theirs is None:
            if mine == 0.0:
                return  # neither end has an arm with anything to agree on
            raise PoolClosed(
                f"the pool at {self.address} did not send an arrangement word, and this host's "
                f"is {mine}. The far end is running a build from before the two ends compared "
                f"their configurations, so a setting given here may decide nothing there."
            )
        if theirs == mine:
            return
        said = explain_arrangement(get_server_args(), mine, theirs)
        raise PoolClosed(
            f"this host and the pool at {self.address} were configured differently: "
            f"arrangement {mine} here against {theirs} there. "
            + (said or "No installed arm could say which setting differs.")
        )

    def issue_frame(self, request_id: int, layer: int, tensors, op: int) -> Handle:
        """Send a multi-tensor frame and do NOT wait. The two-pool split needs this: the sweep
        goes to the cache pool while the feed-forward is already in flight to the weights pool,
        and a synchronous call here would put them back in series."""
        with self._send_lock:
            send_frame(self._sock, Frame(request_id, layer, tuple(tensors), op))
        return Handle(request_id, layer, time.perf_counter(), op)

    def reply_waiting(self, handle: Handle) -> bool:
        """Whether the reply is already on the table, without taking it.

        This is a scheduling probe, not a collection path. A caller that has just finished its
        own work and finds the reply already waiting learned that the OTHER side finished first
        -- which is how the host detects that it, not the pool, has become the schedule's max.
        """
        with self._cond:
            return handle.reply_key in self._replies

    def collect_frame(self, handle: Handle, device):
        """Block for the reply to a frame `issue_frame` sent."""
        key = handle.reply_key
        started = time.perf_counter()
        with self._cond:
            while key not in self._replies:
                if self._failure is not None:
                    raise PoolClosed(
                        f"pool at {self.address} failed"
                    ) from self._failure
                if self._closed:
                    raise PoolClosed(f"pool at {self.address} closed mid-call")
                self._cond.wait(timeout=0.5)
            reply = self._replies.pop(key)
            now = time.perf_counter()
            self._waits.append(
                {
                    "request_id": handle.request_id,
                    "layer": handle.layer,
                    "outstanding_s": now - handle.issued_at,
                    "blocked_s": now - started,
                }
            )
        return tuple(t.to(device, non_blocking=True) for t in reply)

    def call(self, request_id: int, layer: int, tensors, op: int, device):
        """Send a multi-tensor frame and block for its multi-tensor reply.

        Synchronous on purpose. The asynchronous issue/collect pair exists so a sweep can run
        while the pool works; when the SWEEP itself is what the pool is doing, there is nothing
        left on this side to overlap it with, and pretending otherwise would add a slot table to
        buy nothing.
        """
        with self._send_lock:
            send_frame(self._sock, Frame(request_id, layer, tuple(tensors), op))
        key = (request_id, layer, op)
        with self._cond:
            while key not in self._replies:
                if self._failure is not None:
                    raise PoolClosed(
                        f"pool at {self.address} failed"
                    ) from self._failure
                if self._closed:
                    raise PoolClosed(f"pool at {self.address} closed mid-call")
                self._cond.wait(timeout=0.5)
            reply = self._replies.pop(key)
        return tuple(t.to(device, non_blocking=True) for t in reply)

    def collect(self, handle: Handle, device: torch.device | str) -> torch.Tensor:
        """Block until this handle's reply is in, and record how long the block actually was.

        The recorded wait is the evidence that the arrangement did what it claims. If every call's
        wait equals the pool's service time, nothing overlapped: the sweep did not run underneath
        it, and the async path is the synchronous path with extra machinery.
        """
        started = time.perf_counter()
        with self._cond:
            while handle.key not in self._slots:
                if self._failure is not None:
                    raise PoolClosed(
                        f"pool at {self.address} failed with an outstanding call for request "
                        f"{handle.request_id} layer {handle.layer}"
                    ) from self._failure
                if self._closed:
                    raise PoolClosed(
                        f"pool at {self.address} closed with request {handle.request_id} "
                        f"layer {handle.layer} still outstanding"
                    )
                self._cond.wait(timeout=0.5)
            out = self._slots.pop(handle.key)
            # both spans, because their DIFFERENCE is the evidence. issued->collected is how
            # long the call was outstanding; blocked is how much of that the caller actually
            # spent waiting. If they are equal, the caller did nothing in between and nothing
            # overlapped.
            now = time.perf_counter()
            self._waits.append(
                {
                    "request_id": handle.request_id,
                    "layer": handle.layer,
                    "outstanding_s": now - handle.issued_at,
                    "blocked_s": now - started,
                }
            )
        return out.to(device, non_blocking=True)

    def overlap_report(self, last: int = 0) -> dict:
        """Issue-to-collect intervals, so a test can assert the overlap rather than assume it.

        `last` keeps only the most recent N calls. A cumulative mean is the wrong statistic: the
        pool JIT-compiles its kernels on the first frames it sees, and a warm-up call two orders
        of magnitude slower than steady state still moves the mean thousands of calls later. The
        two-machine run's first 512 calls averaged 188 ms against a steady 6.7 ms.
        """
        with self._cond:
            waits = list(self._waits)
        if last:
            waits = waits[-last:]
        if not waits:
            return {"calls": 0}
        with self._cond:
            issues = list(self._issues)
        if last:
            issues = issues[-last:]
        outstanding = [w["outstanding_s"] for w in waits]
        blocked = [w["blocked_s"] for w in waits]
        hidden = [o - b for o, b in zip(outstanding, blocked)]
        return {
            "calls": len(waits),
            "mean_outstanding_s": sum(outstanding) / len(outstanding),
            "mean_blocked_s": sum(blocked) / len(blocked),
            # what the caller got done while the pool worked. Zero means the call was issued and
            # immediately waited on, which is the synchronous arrangement wearing this one's name.
            "mean_hidden_s": sum(hidden) / len(hidden),
            # what the issue itself cost. `build` is the device-to-host copy, which synchronises
            # the stream; `send` is the socket. A build time that tracks the layer's own compute
            # is a window that opens late by exactly that much.
            "mean_build_s": (
                (sum(i["build_s"] for i in issues) / len(issues)) if issues else 0.0
            ),
            "mean_send_s": (
                (sum(i["send_s"] for i in issues) / len(issues)) if issues else 0.0
            ),
        }

    def close(self) -> None:
        try:
            with self._send_lock:
                self._sock.sendall(CLOSE)
        except OSError:
            pass
        finally:
            self._sock.close()
