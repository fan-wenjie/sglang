"""Where the two halves of a layer's attention wait for each other.

Not a queue. A queue imposes an order between different requests and there is none: what this
holds is a set of half-finished pieces of work, each keyed by (request, layer), each waiting for
whichever of its two halves has not arrived.

    the local half    k_t and v_t, computed here at the end of the stage
    the remote half   the swept history and its log partition, from the KV database

Either may arrive first, and neither is the "reply" to the other -- they are two independent
arrivals of one rendezvous. When the second lands the pair leaves the buffer, merges, and goes on
to the next layer. The merge is a lerp over a few thousand elements and can run wherever it is
convenient, including the CPU; the buffer is about arrival, not arithmetic.

## Which half arrives second is the diagnosis

`report()` counts, per side, how often it was the one being waited FOR. If the database is always
second, the database is the critical path; if the local key and value are always second, the
compute is. It is the cheapest instrument in this arrangement and it needs no benchmark: it is a
count of who was late, taken from the traffic that is already flowing.

## A slot that never completes does not fail

The failure this exists to survive is a lost remote half. Nothing raises: the request simply stops
advancing at some layer, quietly, while every other request continues. So slots carry the time
they opened, `stale()` names the ones past a deadline, and the caller probes the remote and
reissues. A reissued query whose original then arrives must not complete the slot twice, so an
attempt number travels with it and a reply from an older attempt is dropped.
"""

from __future__ import annotations

import threading
import time
from typing import NamedTuple

LOCAL = "local"
REMOTE = "remote"


class Slot(NamedTuple):
    """One half-finished piece of work."""

    key: tuple[int, int]
    opened_at: float
    attempt: int
    local: tuple | None
    remote: tuple | None

    @property
    def complete(self) -> bool:
        return self.local is not None and self.remote is not None


class Rendezvous:
    """Half-finished layers, waiting for their other half."""

    def __init__(self, now=time.perf_counter):
        self._now = now
        self._lock = threading.Lock()
        self._slots: dict[tuple[int, int], Slot] = {}
        self._waited_for = {LOCAL: 0, REMOTE: 0}
        self._completed = 0
        self._dropped_stale_replies = 0
        self._reissued = 0

    def _put(self, key, side: str, value: tuple, attempt: int | None):
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                if attempt is not None and attempt > 0:
                    # a reply to a query whose slot has already left: the request moved on, and
                    # completing anything now would merge into a layer that is finished
                    self._dropped_stale_replies += 1
                    return None
                slot = Slot(key, self._now(), 0, None, None)
            elif attempt is not None and attempt < slot.attempt:
                # a reply from a superseded attempt. Dropping it is the point of numbering them:
                # a reissued query whose original then lands would otherwise complete the slot
                # twice, and the second completion would advance the layer a second time
                self._dropped_stale_replies += 1
                return None
            slot = slot._replace(**{side: value})
            if not slot.complete:
                self._slots[key] = slot
                # whoever is here alone is the one being waited FOR
                self._waited_for[side] += 1
                return None
            self._slots.pop(key, None)
            self._completed += 1
            return slot

    def put_local(self, key: tuple[int, int], k, v):
        """This stage's own key and value. Returns the completed slot, or None to keep waiting."""
        return self._put(key, LOCAL, (k, v), None)

    def put_remote(self, key: tuple[int, int], o_swept, lse, attempt: int = 0):
        """The database's answer. Returns the completed slot, or None to keep waiting."""
        return self._put(key, REMOTE, (o_swept, lse), attempt)

    def stale(self, deadline_s: float) -> list[Slot]:
        """Slots open longer than the deadline. Nothing is dropped: the caller decides."""
        cutoff = self._now() - deadline_s
        with self._lock:
            return [s for s in self._slots.values() if s.opened_at <= cutoff]

    def reissue(self, key: tuple[int, int]) -> int | None:
        """Bump a slot's attempt so replies to the old one are ignored. Returns the new number."""
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                return None
            slot = slot._replace(attempt=slot.attempt + 1, opened_at=self._now())
            self._slots[key] = slot
            self._reissued += 1
            return slot.attempt

    def drop(self, request_id: int) -> int:
        """Abandon every slot of one request, for when it is cancelled or its remote is gone."""
        with self._lock:
            keys = [k for k in self._slots if k[0] == request_id]
            for k in keys:
                self._slots.pop(k, None)
        return len(keys)

    def outstanding(self) -> int:
        with self._lock:
            return len(self._slots)

    def report(self) -> dict:
        with self._lock:
            waited = dict(self._waited_for)
            total = waited[LOCAL] + waited[REMOTE]
            return {
                "completed": self._completed,
                "outstanding": len(self._slots),
                "waited_for_local": waited[LOCAL],
                "waited_for_remote": waited[REMOTE],
                # the critical path, read off the traffic rather than measured: the side that is
                # usually LAST is the one the other spends its time waiting on
                "remote_is_late_pct": (100.0 * waited[LOCAL] / total) if total else 0.0,
                "reissued": self._reissued,
                "dropped_stale_replies": self._dropped_stale_replies,
            }


class DepartureQueue:
    """The second layer: completed pairs, grouped by layer, waiting to ride together.

    The first layer pairs the two halves of one (request, layer) and is about CORRECTNESS -- a
    slot leaves it when it is whole. This one is about EFFICIENCY: everything the next stage does
    with a completed pair is weight-shared -- W_o, the MLP, the query and key/value projections --
    so one read of those weights should serve as many pairs as are ready. Measured on this model,
    that work is flat from batch 1 to batch 24 (409 us against 422), which is 24x per token for
    free, and it is the aggregation the whole arrangement would otherwise give up.

    Grouped BY LAYER, because two pairs at different layers share no weight and riding together
    would buy them nothing.

    ## The timeout runs from the last departure, not from arrival

    A queue that waits for a full load stalls the last few requests of a draining workload
    forever -- the same failure the pool's own departure has, and the reason it carries
    --afd-max-wait-ms. The rule here is the one asked for: at most one departure per interval, and
    at least one per interval if anything is waiting. Timing from arrival instead would let a slow
    trickle depart every item on its own the moment each landed, which is the un-batched
    arrangement wearing this one's name.
    """

    def __init__(
        self,
        min_batch: int,
        max_interval_s: float,
        now=time.perf_counter,
        pad_to: int = 0,
    ):
        if min_batch < 1:
            raise ValueError(f"a departure carries at least one, got {min_batch}")
        if max_interval_s <= 0:
            raise ValueError(
                f"max_interval_s must be positive, got {max_interval_s}. A queue with no timeout "
                f"and a minimum above one strands the last requests of a draining workload; it "
                f"does not fail, which is worse."
            )
        if pad_to and pad_to < min_batch:
            raise ValueError(
                f"pad_to={pad_to} is below min_batch={min_batch}; a load that departs full would "
                f"then be padded DOWN, which is a truncation wearing padding's name"
            )
        self.min_batch = min_batch
        self.max_interval_s = max_interval_s
        # Pad the weight-shared work up to a fixed width. Nearly free at decode, because the
        # feed-forward reads 535 MB of weights whatever the batch: measured 399 us at batch 1 and
        # 432 at batch 64, so a load of three padded to sixty-four costs about what three would.
        #
        # Only the weight-shared work. The sweep runs at the REAL count: a padded rider has no
        # history to sweep, and the sweep's cost is proportional to the batch once it is
        # bandwidth-bound, so padding it is spent on nothing.
        self.pad_to = pad_to
        self._now = now
        self._lock = threading.Lock()
        self._waiting: dict[int, list] = {}
        self._last_departure = now()
        self.departures = 0
        self.departed_full = 0
        self.departed_on_time = 0
        self.riders = 0
        self.padding = 0

    def offer(self, layer: int, item) -> list | None:
        """Add a completed pair. Returns a full load to run now, or None to keep waiting."""
        with self._lock:
            queue = self._waiting.setdefault(layer, [])
            queue.append(item)
            if len(queue) < self.min_batch:
                return None
            self.departed_full += 1
            return self._depart(layer)

    def due(self) -> list[tuple[int, list]]:
        """Everything waiting, if the interval since the last departure has run out.

        Returns (layer, riders) pairs and empties the queues. Called from a timer; a caller that
        never calls it turns min_batch into a requirement rather than a target.
        """
        with self._lock:
            if self._now() - self._last_departure < self.max_interval_s:
                return []
            if not any(self._waiting.values()):
                return []
            out = []
            for layer in list(self._waiting):
                if self._waiting[layer]:
                    self.departed_on_time += 1
                    out.append((layer, self._depart(layer)))
            return out

    def _depart(self, layer: int) -> list:
        """Caller holds the lock."""
        riders = self._waiting.pop(layer, [])
        self._last_departure = self._now()
        self.departures += 1
        self.riders += len(riders)
        if self.pad_to:
            self.padding += max(0, self.pad_to - len(riders))
        return riders

    def width_for(self, riders: list) -> int:
        """How wide to run the weight-shared work for this load.

        The sweep is not run at this width. Nothing here enforces that -- it cannot, since it
        never sees the sweep -- so the caller has to keep the two apart, and `report()` shows the
        padding it is paying for so a load that is mostly padding is visible rather than assumed
        away.
        """
        return max(len(riders), self.pad_to) if self.pad_to else len(riders)

    def waiting(self) -> int:
        with self._lock:
            return sum(len(q) for q in self._waiting.values())

    def report(self) -> dict:
        with self._lock:
            return {
                "departures": self.departures,
                "riders": self.riders,
                "mean_riders": (
                    (self.riders / self.departures) if self.departures else 0.0
                ),
                "departed_full": self.departed_full,
                "departed_on_time": self.departed_on_time,
                "waiting": sum(len(q) for q in self._waiting.values()),
                "padding": self.padding,
                "padding_pct": (
                    (100.0 * self.padding / (self.riders + self.padding))
                    if (self.riders + self.padding)
                    else 0.0
                ),
            }
