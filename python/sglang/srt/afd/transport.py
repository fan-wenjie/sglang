"""The shape a transfer takes, chosen so an RDMA one can be dropped in without moving anything.

The arrangement moves hidden states between two machines. It does that over TCP today and the
measurements say the fabric is where the cost is -- a round trip is 480 us on this overlay and
about 10 us on RoCE, which is fifty times -- so the transport will be replaced. What that costs
depends entirely on the shape of the interface it is replaced through.

## Why this is verbs-shaped and not socket-shaped

sglang's own disaggregation transports already answer this. `BaseKVManager` takes `kv_data_ptrs`
and `kv_data_lens` -- raw pointers into memory registered once -- names `ib_device` in its
arguments, and reports progress through a `KVPoll` enum rather than by returning from a blocking
call. That is the verbs model, and mooncake, nixl and ascend all implement it.

A socket-shaped interface can be emulated over RDMA and the emulation throws away the two things
RDMA is for: the buffer has to be copied into a registered region on every call, and a completion
has to be turned back into a blocking read. So the interface is defined in the shape that CANNOT
be emulated cheaply, and TCP -- which can emulate anything -- implements it downward.

Three consequences, and each is a thing that would be expensive to change later:

    buffers are registered once and reused   an RDMA send from an unregistered buffer means a
                                             copy plus a registration, and registration is
                                             slower than the transfer
    completion is polled, not returned       a blocking send has nowhere to put "in flight", and
                                             in flight is the whole point of issuing early
    payloads are (pointer, length)           the transport does not own, allocate, or free the
                                             memory it moves, so the caller can keep a tensor
                                             alive across a transfer without the two disagreeing
                                             about whose it is

## What this is not

Not a replacement for `protocol.py`. Framing -- which op, which layer, which request -- stays
where it is; this is only how the bytes of a frame get from one machine to the other. A transport
that also knew what a sweep was would have to be reimplemented per fabric.
"""

from __future__ import annotations

import socket
import threading
from typing import Protocol

# What a posted transfer is doing. The names match sglang's KVPoll so a reader moving between the
# two is not learning two vocabularies for one idea.
PENDING, DONE, FAILED = "transferring", "success", "failed"


class Region:
    """Memory the transport may move without being told about it again.

    Over TCP this is a memoryview and registering is free. Over RDMA it is an `ibv_mr` and
    registering is expensive -- hundreds of microseconds, which is more than a transfer -- so a
    caller that registers per call has built something slower than the socket it replaced. That is
    why this object exists at all: to make the reuse the natural thing to write.
    """

    __slots__ = ("view", "handle", "nbytes")

    def __init__(self, view: memoryview, handle=None) -> None:
        if not isinstance(view, memoryview):
            raise TypeError(
                f"a region wraps a memoryview so the transport can address it without owning it; "
                f"got {type(view).__name__}"
            )
        if view.format != "B" or view.ndim != 1:
            # cast here rather than at each use: a typed or multidimensional view has strides an
            # RDMA scatter-gather list cannot express, and the failure would be at post time
            view = view.cast("B")
        self.view = view
        self.handle = handle          # whatever the fabric needs; None for TCP
        self.nbytes = view.nbytes

    def slice(self, offset: int, length: int) -> memoryview:
        if offset < 0 or length < 0 or offset + length > self.nbytes:
            raise ValueError(
                f"[{offset}, {offset + length}) is outside a {self.nbytes} byte region. A "
                f"transport that clamped this would send the wrong bytes and report success."
            )
        return self.view[offset : offset + length]


class Transfer:
    """One posted transfer. Ask it whether it is done; do not wait on it by default."""

    __slots__ = ("state", "error", "nbytes", "_done")

    def __init__(self, nbytes: int) -> None:
        self.state = PENDING
        self.error: BaseException | None = None
        self.nbytes = nbytes
        self._done = threading.Event()

    def finish(self, error: BaseException | None = None) -> None:
        self.error = error
        self.state = FAILED if error is not None else DONE
        self._done.set()

    def poll(self) -> str:
        return self.state

    def wait(self, timeout: float | None = None) -> str:
        """Block until done. Present because a caller sometimes has nothing else to do -- but the
        arrangement's whole schedule is built on NOT calling this between an issue and the work it
        was issued ahead of."""
        if not self._done.wait(timeout=timeout):
            return PENDING
        if self.error is not None:
            raise self.error
        return self.state


class Transport(Protocol):
    """What a fabric has to provide. TCP implements it below; RDMA implements it later."""

    def register(self, buffer) -> Region:
        """Make a buffer transferable. Called once per buffer, not once per transfer."""

    def post_send(self, peer, region: Region, offset: int, length: int) -> Transfer:
        """Start moving bytes out. Returns immediately."""

    def post_recv(self, peer, region: Region, offset: int, length: int) -> Transfer:
        """Start filling bytes in. Returns immediately."""


class SocketTransport:
    """The current fabric, behind the interface the next one wants.

    Sends are performed on the calling thread, because a socket send of a hundred kilobytes on a
    warm connection returns in tens of microseconds and a thread handoff costs more than it saves.
    Receives are posted to a reader thread per peer, because a receive genuinely has to wait, and
    waiting on the calling thread is what this interface exists to stop.
    """

    def __init__(self) -> None:
        self._readers: dict[int, threading.Thread] = {}
        self._lock = threading.Lock()

    def register(self, buffer) -> Region:
        """Free here. The object still exists so that a caller written against this interface is
        already reusing its buffers when the fabric underneath starts charging for it."""
        if isinstance(buffer, memoryview):
            return Region(buffer)
        return Region(memoryview(buffer))

    def post_send(self, peer: socket.socket, region: Region, offset: int, length: int) -> Transfer:
        transfer = Transfer(length)
        try:
            peer.sendall(region.slice(offset, length))
            transfer.finish()
        except BaseException as e:                     # a dead peer is a failed transfer, not a
            transfer.finish(e)                          # raise on the thread that posted it
        return transfer

    def post_recv(self, peer: socket.socket, region: Region, offset: int, length: int) -> Transfer:
        transfer = Transfer(length)
        target = region.slice(offset, length)

        def fill():
            got = 0
            try:
                while got < length:
                    n = peer.recv_into(target[got:], length - got)
                    if n == 0:
                        raise ConnectionError(
                            f"the peer closed with {length - got} of {length} byte(s) unread; a "
                            f"partial frame is not a short frame, it is a frame whose remainder "
                            f"would be read as the next one's header"
                        )
                    got += n
                transfer.finish()
            except BaseException as e:
                transfer.finish(e)

        thread = threading.Thread(target=fill, daemon=True, name="afd-recv")
        thread.start()
        return transfer


class BufferRing:
    """A fixed set of registered buffers, handed out and returned.

    The arrangement allocates a tensor per frame today. Over TCP that is an allocation; over RDMA
    it is an allocation AND a registration, and registration costs more than the transfer it
    enables. A ring makes the reuse structural rather than something each call site has to
    remember.

    Sized rather than grown: a ring that allocated on demand under load would register a buffer at
    exactly the moment the fabric is busiest.
    """

    def __init__(self, transport, *, count: int, nbytes: int) -> None:
        if count <= 0 or nbytes <= 0:
            raise ValueError(f"a ring of {count} x {nbytes} holds nothing")
        self.nbytes = nbytes
        self._free: list[Region] = [
            transport.register(bytearray(nbytes)) for _ in range(count)
        ]
        self._all = list(self._free)
        self._lock = threading.Lock()
        self._high_water = 0

    def take(self) -> Region:
        with self._lock:
            if not self._free:
                raise RuntimeError(
                    f"all {len(self._all)} buffer(s) are in flight. Raise the ring's size to the "
                    f"number of transfers this arrangement keeps outstanding; allocating one here "
                    f"would register memory at the busiest moment."
                )
            region = self._free.pop()
            self._high_water = max(self._high_water, len(self._all) - len(self._free))
            return region

    def give_back(self, region: Region) -> None:
        with self._lock:
            self._free.append(region)

    def report(self) -> dict:
        with self._lock:
            return {"buffers": len(self._all), "in_flight": len(self._all) - len(self._free),
                    "high_water": self._high_water, "bytes_each": self.nbytes}
