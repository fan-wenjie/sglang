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

# how many call records to keep. Enough for several report windows, bounded so a long run does
# not accumulate one dict per pool call forever.
WAIT_HISTORY = 8192


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

    def __init__(self, address: str, connect_timeout_s: float, reconnect: bool = True,
                 max_reconnects: int = 8):
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
        # set by a host that holds state the pool has to call back for. None means this client
        # only ever receives replies, and an inbound request against it is a configuration
        # disagreement rather than a protocol error -- so it is named as one.
        self.serve = None
        self._failure: BaseException | None = None
        # bounded: an hour of decode is millions of entries, and overlap_report used to slice a
        # list that only ever grew. The report is windowed anyway.
        self._waits: collections.deque = collections.deque(maxlen=WAIT_HISTORY)
        self._closed = False
        self._issues = collections.deque(maxlen=4096)
        self._receiver = threading.Thread(target=self._receive, name="afd-pool-recv", daemon=True)
        self._receiver.start()

    def _open(self):
        sock = socket.create_connection((self._host, self._port),
                                        timeout=self._connect_timeout_s)
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
            logger.warning("afd: could not reconnect to the pool at %s: %s", self.address, e)
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
            self.address, self.reconnects, outstanding,
        )
        self._receiver = threading.Thread(target=self._receive, name="afd-pool-recv", daemon=True)
        self._receiver.start()
        return True

    def _answer_inbound(self, frame) -> None:
        """A REQUEST arriving on the reply socket, because the pool needs something this end has.

        Under the group cut the pool runs a linear layer's weights and this end holds its recurrent
        state, so the pool calls back mid-span. The call arrives interleaved with the replies this
        client is waiting for, and telling them apart is the reader's job: an inbound request
        stored in the reply table would hang the caller it was keyed as, and hand a state reading
        to whoever asked for that key.

        Answered ON THE READER THREAD, deliberately. The pool is blocked waiting for this and will
        send nothing else until it has it, so there is no reply being delayed -- and a handoff to a
        worker would add a scheduler round trip to a call whose whole budget is one feed-forward.
        The send takes the same lock every other send takes, because a reply going out from here
        can interleave with one going out from a caller's thread.
        """
        if self.serve is None:
            raise RuntimeError(
                f"the pool sent a {OP_NAMES.get(frame.op, frame.op)} and this client has no "
                f"handler for it. The two ends disagree about who holds the recurrent state: this "
                f"one was built expecting the pool to hold it, and the pool expects this one to."
            )
        answer = self.serve(frame)
        if answer is None:
            return
        with self._send_lock:
            send_frame(self._sock, Frame(frame.request_id, frame.layer, tuple(answer), frame.op))

    def _receive(self) -> None:
        try:
            while True:
                frame = decode(self._sock)
                if frame is None:
                    break
                if frame.op in INBOUND_OPS:
                    self._answer_inbound(frame)
                    continue
                with self._cond:
                    if frame.op == OP_HELLO:
                        self._hello = float(frame.tensor[0, 0])
                        self._replies[(frame.request_id, frame.layer, frame.op)] = frame.tensors
                    elif len(frame.tensors) == 1 and frame.op == OP_FFN:
                        self._slots[frame.key] = frame.tensor
                    else:
                        # keyed by op as well: one (request, layer) has more than one answer in
                        # flight, and they are not interchangeable
                        self._replies[(frame.request_id, frame.layer, frame.op)] = frame.tensors
                    self._cond.notify_all()
        except BaseException as e:  # noqa: BLE001 -- it is re-raised in every waiting caller
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
    _NEED_NAMES = {1: "a feed-forward", 2: "a cache to sweep and append",
                   4: "the key and value projections",
                   8: "whole spans, four layers a call"}

    def require(self, needs: int, timeout_s: float = 15.0) -> int:
        """Ask the pool what it does, and refuse now if it is not what this host needs.

        Called once at startup. The alternative is what this cost twice: a host configured for a
        cache pool reaches one that only runs feed-forwards, sends it a frame it cannot parse, and
        reports "closed mid-call" -- a message about a socket that names neither side's
        configuration.
        """
        self.collect_frame(
            self.issue_frame(0, 0, (torch.zeros(1, 1),), OP_HELLO), "cpu"
        )
        served = int(self._hello or 0)
        missing = [name for bit, name in self._NEED_NAMES.items()
                   if needs & bit and not served & bit]
        if missing:
            raise PoolClosed(
                f"the pool at {self.address} does not serve {', and '.join(missing)}. This host "
                f"was configured to ask it for that, so the two were started with different "
                f"roles -- most likely the pool is running without the flag that gives it a "
                f"cache. Failing here rather than at the first token."
            )
        return served

    def issue_frame(self, request_id: int, layer: int, tensors, op: int) -> Handle:
        """Send a multi-tensor frame and do NOT wait. The two-pool split needs this: the sweep
        goes to the cache pool while the feed-forward is already in flight to the weights pool,
        and a synchronous call here would put them back in series."""
        with self._send_lock:
            send_frame(self._sock, Frame(request_id, layer, tuple(tensors), op))
        return Handle(request_id, layer, time.perf_counter(), op)

    def collect_frame(self, handle: Handle, device):
        """Block for the reply to a frame `issue_frame` sent."""
        key = handle.reply_key
        started = time.perf_counter()
        with self._cond:
            while key not in self._replies:
                if self._failure is not None:
                    raise PoolClosed(f"pool at {self.address} failed") from self._failure
                if self._closed:
                    raise PoolClosed(f"pool at {self.address} closed mid-call")
                self._cond.wait(timeout=0.5)
            reply = self._replies.pop(key)
            now = time.perf_counter()
            self._waits.append({"request_id": handle.request_id, "layer": handle.layer,
                                "outstanding_s": now - handle.issued_at,
                                "blocked_s": now - started})
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
                    raise PoolClosed(f"pool at {self.address} failed") from self._failure
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
            "mean_build_s": (sum(i["build_s"] for i in issues) / len(issues)) if issues else 0.0,
            "mean_send_s": (sum(i["send_s"] for i in issues) / len(issues)) if issues else 0.0,
        }

    def close(self) -> None:
        try:
            with self._send_lock:
                self._sock.sendall(CLOSE)
        except OSError:
            pass
        finally:
            self._sock.close()
