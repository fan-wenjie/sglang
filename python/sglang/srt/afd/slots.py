"""Which request owns which slot, for every kind of per-request state a pool holds.

A cache pool holds two kinds of memory and they are addressed identically: a request arrives, it
needs somewhere to keep what it remembers, and when it finishes that somewhere goes back on the
free list. The KV cache and the recurrent state differ in what they store and in how they are
read; they do not differ at all in this.

It was written twice before this file existed, once in each, which is how a fix to one of them
becomes a bug that survives in the other.

## What this deliberately does not know

Nothing about tensors, layers, or contractions. A slot table that also allocated buffers would
have to know the shape of both kinds, and the two shapes are the only real difference between
them. `report()` takes the byte count from its caller for the same reason.
"""

from __future__ import annotations

import threading

import torch


class NoFreeSlot(RuntimeError):
    """Every slot is taken.

    Raised rather than evicting, for both kinds and for different reasons. A KV cache could in
    principle be rebuilt from the prompt, but not cheaply and not without the tokens; a recurrent
    state cannot be rebuilt at all from anything the pool holds -- it IS the history, compressed,
    and a slot handed on without its owner's consent loses that history silently.
    """


class SlotTable:
    """A fixed set of slots, handed out per request and returned on release."""

    def __init__(self, slots: int, *, what: str) -> None:
        if slots <= 0:
            raise ValueError(
                f"slots={slots} for {what}: a pool with no room for a request holds nothing and "
                f"would report success while serving no history"
            )
        self.slots = slots
        self.what = what
        self._lock = threading.Lock()
        self._of: dict[int, int] = {}
        self._free = list(range(slots))
        self._high_water = 0

    def of(self, request_id: int, *, create: bool = True) -> int | None:
        with self._lock:
            slot = self._of.get(request_id)
            if slot is not None or not create:
                return slot
            if not self._free:
                raise NoFreeSlot(
                    f"all {self.slots} {self.what} slot(s) are taken and request {request_id} "
                    f"wants one. Raise the pool's slot count to the concurrency it is meant to "
                    f"serve; evicting here would serve a request whose past went missing, which "
                    f"reads as a fluent answer to a question nobody asked."
                )
            slot = self._free.pop(0)
            self._of[request_id] = slot
            self._high_water = max(self._high_water, len(self._of))
            return slot

    def give_back(self, request_id: int) -> int | None:
        """Return this request's slot to the free list. None if it never had one."""
        with self._lock:
            slot = self._of.pop(request_id, None)
            if slot is not None:
                self._free.append(slot)
            return slot

    def report(self, *, bytes_held: int = 0) -> dict:
        with self._lock:
            return {
                "what": self.what,
                "slots": self.slots,
                "slots_in_use": len(self._of),
                "high_water": self._high_water,
                "bytes": bytes_held,
            }


def row_request_ids(forward_batch) -> torch.Tensor:
    """Whose history each ROW of this forward belongs to.

    Decode carries one token per request, so the rows are the requests. Extend carries a whole
    chunk per request, so each request's id repeats for as many tokens as it brought. Both are the
    same question -- the cache is keyed by request and the frame is keyed by row -- and answering
    it from the batch size alone is right until two requests are in one forward, which is every
    forward a server actually runs.
    """
    pool_indices = forward_batch.req_pool_indices.view(-1)
    lens = forward_batch.extend_seq_lens
    if lens is None:
        return pool_indices
    return torch.repeat_interleave(pool_indices, lens.to(pool_indices.device))


def _runs(request_ids) -> list[tuple[int, int, int]]:
    """The batch as (request, first row, row count), in order.

    A decode bus has one row a request and every run is length one. A prefill bus has one row a
    TOKEN, so a request's rows are a contiguous run that has to be walked in order. A bus carrying
    both has runs of both lengths, which is why this is not a `forward_mode` question: the mode
    describes the batch and the runs describe the rows.
    """
    runs, start = [], 0
    for i, r in enumerate(request_ids):
        if i + 1 == len(request_ids) or int(request_ids[i + 1]) != int(r):
            runs.append((int(r), start, i + 1 - start))
            start = i + 1
    return runs
