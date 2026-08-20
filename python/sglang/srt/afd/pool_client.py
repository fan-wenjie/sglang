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
from sglang.srt.afd.protocol import CLOSE, OP_FFN, Frame, decode, send_frame

logger = logging.getLogger(__name__)

# how many call records to keep. Enough for several report windows, bounded so a long run does
# not accumulate one dict per pool call forever.
WAIT_HISTORY = 8192


class Handle(NamedTuple):
    """What `issue` returns. `issued_at` is kept so a run can prove the overlap happened."""

    request_id: int
    layer: int
    issued_at: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.request_id, self.layer)


class PoolClosed(RuntimeError):
    """The pool went away while a call was outstanding. Never silently returns zeros for it."""


class PoolClient:
    """One connection to one pool, shared by every request on this host."""

    def __init__(self, address: str, connect_timeout_s: float):
        host, _, port = address.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"--afd-pool-addr wants HOST:PORT, got {address!r}")
        self.address = address
        self._sock = socket.create_connection((host, int(port)), timeout=connect_timeout_s)
        self._sock.settimeout(None)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._send_lock = threading.Lock()
        self._cond = threading.Condition()
        self._slots: dict[tuple[int, int], torch.Tensor] = {}
        # multi-tensor replies live apart from the one-tensor feed-forward slots, so a reply of
        # the wrong shape cannot be handed to a caller expecting the other
        self._replies: dict[tuple[int, int], tuple] = {}
        self._failure: BaseException | None = None
        # bounded: an hour of decode is millions of entries, and overlap_report used to slice a
        # list that only ever grew. The report is windowed anyway.
        self._waits: collections.deque = collections.deque(maxlen=WAIT_HISTORY)
        self._closed = False
        self._receiver = threading.Thread(target=self._receive, name="afd-pool-recv", daemon=True)
        self._receiver.start()

    def _receive(self) -> None:
        try:
            while True:
                frame = decode(self._sock)
                if frame is None:
                    break
                with self._cond:
                    if len(frame.tensors) == 1 and frame.op == OP_FFN:
                        self._slots[frame.key] = frame.tensor
                    else:
                        self._replies[frame.key] = frame.tensors
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
        """Send one layer's work. Returns immediately; the reply lands in a slot."""
        frame = Frame.one(request_id, layer, hidden)
        with self._send_lock:
            send_frame(self._sock, frame)
        return Handle(request_id, layer, time.perf_counter())

    def call(self, request_id: int, layer: int, tensors, op: int, device):
        """Send a multi-tensor frame and block for its multi-tensor reply.

        Synchronous on purpose. The asynchronous issue/collect pair exists so a sweep can run
        while the pool works; when the SWEEP itself is what the pool is doing, there is nothing
        left on this side to overlap it with, and pretending otherwise would add a slot table to
        buy nothing.
        """
        with self._send_lock:
            send_frame(self._sock, Frame(request_id, layer, tuple(tensors), op))
        key = (request_id, layer)
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
        }

    def close(self) -> None:
        try:
            with self._send_lock:
                self._sock.sendall(CLOSE)
        except OSError:
            pass
        finally:
            self._sock.close()
