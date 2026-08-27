"""Where a pool call's time goes, accounted by the pool itself.

The pool peaks at 1403 calls/s on two connections and falls to 844 on sixteen, with the GPU at 0%.
So the cost is the serving path rather than the work -- but "the serving path" is three different
things with three different fixes, and no measurement so far separates them:

    wire in     reading a frame off the socket and materialising its tensors
    work        the forward itself, which the GPU utilisation says is small at this frame width
    wire out    serialising the reply and writing it back

py-spy would have said which, and cannot: attaching to the scheduler process needs root on this
host and there is none. So the pool accounts for its own time instead, which is better in one way
worth keeping -- it stays on in the deployment, so the reading comes from a pool doing real work
rather than from a profiling run nobody repeats.

Always on, not behind an env var. `perf_counter` is tens of nanoseconds against a call that costs
half a millisecond, and an env-gated probe that never fires is indistinguishable from a probe that
found nothing -- which has cost this tree a day before.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Every this many calls, one line. Often enough to see a change during a benchmark, rare enough
# that the logging itself is not part of what is measured.
EVERY = 2000


class Meter:
    """Nanoseconds and counts per phase, summed across connection threads.

    Summed rather than kept per thread: the question is where the pool's time goes, and with
    sixteen connections contending the per-thread split says less than the total does. The lock is
    held for an addition; it is not the contention under study.
    """

    def __init__(self, every: int = EVERY) -> None:
        self.every = every
        self._lock = threading.Lock()
        self.phases: dict[str, float] = {"wire_in": 0.0, "work": 0.0, "wire_out": 0.0}
        self.calls = 0
        self.since = time.perf_counter()

    def add(self, phase: str, seconds: float) -> None:
        with self._lock:
            self.phases[phase] += seconds

    def call(self) -> None:
        """One completed call. Reports and resets on every `every`-th."""
        with self._lock:
            self.calls += 1
            if self.calls % self.every:
                return
            elapsed = time.perf_counter() - self.since
            phases, calls = dict(self.phases), self.every
            self.phases = {k: 0.0 for k in self.phases}
            self.since = time.perf_counter()
        accounted = sum(phases.values())
        logger.info(
            "afd pool time: %s calls in %.2fs (%.0f/s) -- wire in %.3f ms, work %.3f ms, wire out "
            "%.3f ms a call; %.0f%% of the wall clock is accounted for, the rest is waiting for a "
            "caller or for the GIL",
            calls,
            elapsed,
            calls / max(elapsed, 1e-9),
            phases["wire_in"] / calls * 1e3,
            phases["work"] / calls * 1e3,
            phases["wire_out"] / calls * 1e3,
            100.0 * accounted / max(elapsed, 1e-9),
        )

    def timed(self, phase: str):
        """`with meter.timed("work"):` -- a context manager rather than a pair of calls, so an
        exception on the path cannot leave a phase permanently open."""
        return _Timed(self, phase)


class _Timed:
    def __init__(self, meter: Meter, phase: str) -> None:
        self.meter, self.phase = meter, phase

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.meter.add(self.phase, time.perf_counter() - self.started)
        return False
