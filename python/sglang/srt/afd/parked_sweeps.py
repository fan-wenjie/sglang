"""Sweeps that arrived before the history they are meant to read.

A sweep says which range of the history it covers: positions [0, length). An append writes position
`length`. The two do not overlap, so they can be computed in the same visit and issued in either
order -- but only if a sweep that arrives first WAITS instead of failing or reading a short history.

That waiting cannot happen on the connection thread. The pool runs one thread per connection, and
today a request's sweep and its appends travel the same socket, so a thread blocked inside a sweep
is the thread that would have read the append it is waiting for. It would wait forever, and the
symptom would be a server that stops answering rather than one that reports anything.

So a sweep that cannot be satisfied is PARKED here and the connection thread goes back to reading.
Whoever appends checks this buffer and releases the sweeps whose length has become readable. This
is the two-part buffer the arrangement was specified around: neither half waits on the other, and
work leaves the buffer when both halves are present.

## What this replaces, and why the old one was not safe

`CachePool.sweep(expect=)` compared the caller's count against the pool's and raised when they
disagreed. That turned the race into a loud failure, which was the right first move -- the failure
it replaced was a model attending to a history with a hole in it, staying fluent, and saying
nothing. But raising still requires the append to have landed FIRST, and the only reason it does is
that both frames travel one TCP connection and this pool answers a connection's frames in arrival
order.

That is a property of the transport, not of the design. The measurements recommend sharding the
client across links, and the first shard would break it: two connections have no order between
them, and the sweep would start failing on traffic that is completely correct.

Parking removes the dependency. Order stops mattering because the length says what to read rather
than the cache's current fill level saying it.

## Timeouts are a report, not a retry

A parked sweep whose append never arrives is a lost append, a dead peer, or a plan error, and this
buffer cannot tell those apart. It times out and says what it was waiting for; the caller decides
whether to retry, because only the caller knows whether the peer is still alive. A buffer that
retried by itself would turn a dead peer into a stall with no message.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, NamedTuple


class Parked(NamedTuple):
    """One sweep waiting for the history it names to exist."""

    request_id: int
    layer: int
    length: int          # positions [0, length) this sweep covers
    parked_at: float
    resume: Callable     # called with no arguments once `length` positions are held


class ParkedSweeps:
    """Sweeps held until their range is readable, released by whoever makes it readable.

    Keyed by (request, layer) because that is the granularity a history is appended at. Several
    sweeps may park on one key -- a retry, or two layers of one request pipelined -- so each key
    holds a list, and release walks it rather than assuming one.
    """

    def __init__(self, *, timeout_s: float) -> None:
        if timeout_s <= 0:
            raise ValueError(
                f"a park timeout of {timeout_s} would expire every sweep before its append could "
                f"arrive, which is the failure this buffer exists to prevent"
            )
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._parked: dict[tuple[int, int], list[Parked]] = {}
        self._peak = 0
        self._parked_total = 0
        self._released_total = 0
        self._expired_total = 0

    def park(self, *, request_id: int, layer: int, length: int, resume: Callable) -> None:
        entry = Parked(request_id=request_id, layer=layer, length=length,
                       parked_at=time.monotonic(), resume=resume)
        with self._lock:
            self._parked.setdefault((request_id, layer), []).append(entry)
            self._parked_total += 1
            held = sum(len(v) for v in self._parked.values())
            self._peak = max(self._peak, held)

    def release(self, *, request_id: int, layer: int, held: int) -> list[Parked]:
        """Everything on this key that `held` positions now satisfies.

        The resume callbacks are run by the CALLER, outside this object's lock. Running them here
        would hold the lock across a socket write, and the thread on the other end of that write
        may be trying to park -- which is a lock ordering that deadlocks under exactly the load
        this buffer is for.
        """
        with self._lock:
            waiting = self._parked.get((request_id, layer))
            if not waiting:
                return []
            ready = [entry for entry in waiting if entry.length <= held]
            if not ready:
                return []
            remaining = [entry for entry in waiting if entry.length > held]
            if remaining:
                self._parked[(request_id, layer)] = remaining
            else:
                del self._parked[(request_id, layer)]
            self._released_total += len(ready)
        return ready

    def expired(self, *, now: float | None = None) -> list[Parked]:
        """Sweeps that have waited longer than the timeout, removed from the buffer.

        Removed rather than reported in place: a sweep that has timed out is going to be answered
        with an error, and leaving it here would let a late append release it as well, answering
        one frame twice.
        """
        moment = time.monotonic() if now is None else now
        out: list[Parked] = []
        with self._lock:
            for key in list(self._parked):
                keep = []
                for entry in self._parked[key]:
                    if moment - entry.parked_at >= self.timeout_s:
                        out.append(entry)
                    else:
                        keep.append(entry)
                if keep:
                    self._parked[key] = keep
                else:
                    del self._parked[key]
            self._expired_total += len(out)
        return out

    def drop(self, *, request_id: int) -> list[Parked]:
        """Everything parked for a request that has gone away, so a release cannot leak."""
        out: list[Parked] = []
        with self._lock:
            for key in [k for k in self._parked if k[0] == request_id]:
                out.extend(self._parked.pop(key))
        return out

    def held(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._parked.values())

    def report(self) -> dict:
        """What the buffer did, for a run record.

        `peak` is the number worth watching: a buffer that never holds more than one is a buffer
        whose sweeps and appends are arriving in order anyway, and the parking is costing a
        dictionary lookup to solve a problem this traffic does not have. A peak that grows with
        concurrency is the buffer doing its job.
        """
        with self._lock:
            return {
                "parked": self._parked_total,
                "released": self._released_total,
                "expired": self._expired_total,
                "peak_held": self._peak,
                "held_now": sum(len(v) for v in self._parked.values()),
                "timeout_s": self.timeout_s,
            }
