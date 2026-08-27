"""Histories held as one buffer a layer, with a slot a request, so a batch is one kernel.

The cache this replaces kept a separate tensor for every (request, layer) and grew each one with
`torch.cat`. Two costs followed from that shape, and both were measured before this was written:

    appending      cat allocates and copies the WHOLE history to add one position. At 32k context
                   that is 64 MiB of keys and as much again of values, per request, per layer,
                   per step
    sweeping       separate tensors cannot be contracted together, so a batch of eight requests
                   was eight passes of about ten kernels each. Measured on the live pool: a sweep
                   answering eight requests took 3.6 ms of which 2.67 ms was this, and the cost
                   did not move when the context grew from 1k to 8k -- it is per REQUEST, not per
                   position

One buffer a layer fixes both. A slot is written in place, and every slot is contracted in one
einsum with the boundary stated as a mask.

## What a slot costs, and why the capacity is the caller's decision

A slot reserves `max_context` positions whether the request uses them or not, so the buffer is
`slots x kv_heads x max_context x head_dim` for keys and the same for values. That is the price of
a batch being one kernel, and it is the same trade every paged attention implementation makes.

Neither number gets a default here. Sized too small the pool refuses traffic it could have served;
sized too large it reserves memory the histories will never occupy, on a machine whose whole
purpose is holding histories. Only the deployment knows its own concurrency and context.

## Ordering, and what this class does NOT decide

`positions(slot)` is what a sweep reads to know how far a history goes, and the sweep's own range
comes from the caller's `length` rather than from here. This class answers
"what is held" and never "what should be read", because those differ for exactly one step and that
step is where a sweep would otherwise cover a history with a hole in it.
"""

from __future__ import annotations

import threading

import torch

from sglang.srt.afd.slots import (  # noqa: F401  (NoFreeSlot re-exported)
    NoFreeSlot,
    SlotTable,
)

# (slots, kv_heads, positions, head_dim). The position axis is named because appending along the
# wrong one is the single bug in this file that nothing downstream would catch: the shapes still
# work and the attention is over a history that has been transposed into nonsense.
SLOT_AXIS, POSITION_AXIS = 0, 2


class SlottedKV:
    """Per-layer key and value buffers, one slot per live request."""

    def __init__(
        self,
        *,
        slots: int,
        kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        max_context: int,
        device,
        dtype=torch.bfloat16,
    ) -> None:
        if slots <= 0 or max_context <= 0:
            raise ValueError(
                f"slots={slots}, max_context={max_context}: a cache pool with no room for a "
                f"request or no room for a position holds nothing and would report success"
            )
        self.slots = slots
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.max_context = max_context
        self.device = device
        self.dtype = dtype

        self._lock = threading.Lock()
        self._k: dict[int, torch.Tensor] = (
            {}
        )  # layer -> (slots, kv_heads, max_context, dim)
        self._v: dict[int, torch.Tensor] = {}
        self._held: dict[tuple[int, int], int] = (
            {}
        )  # (slot, layer) -> positions written
        # who owns which slot is the one thing this and the recurrent-state store do identically,
        # so it lives in one place; see slots.py
        self._table = SlotTable(slots, what="kv cache")

    # -- slots ------------------------------------------------------------------

    def slot_of(self, request_id: int, *, create: bool = True) -> int | None:
        return self._table.of(request_id, create=create)

    def release(self, request_id: int) -> int:
        """Free a request's slot and forget its lengths. Returns how many layers it held.

        The lengths are what make a KV slot safe to reuse without zeroing: a new occupant starts
        at zero positions and the mask excludes everything the old one wrote. The recurrent-state
        store has no such length and must clear its buffers instead.
        """
        slot = self._table.give_back(request_id)
        if slot is None:
            return 0
        with self._lock:
            layers = [key for key in self._held if key[0] == slot]
            for key in layers:
                del self._held[key]
        return len(layers)

    def positions(self, request_id: int, layer: int) -> int:
        slot = self._table.of(request_id, create=False)
        if slot is None:
            return 0
        with self._lock:
            return self._held.get((slot, layer), 0)

    # -- the buffers ------------------------------------------------------------

    def _buffers(self, layer: int) -> tuple:
        """This layer's key and value buffers, allocated on first use.

        Lazily, because a stack has many layers and a pool serving one arrangement may touch only
        some of them -- the softmax layers on a hybrid model are sixteen of sixty-four, and
        reserving the other forty-eight would be reserving three quarters of the machine for
        histories that will never exist.
        """
        with self._lock:
            k = self._k.get(layer)
            if k is not None:
                return k, self._v[layer]
            shape_k = (self.slots, self.kv_heads, self.max_context, self.head_dim)
            shape_v = (self.slots, self.kv_heads, self.max_context, self.v_head_dim)
            k = torch.zeros(shape_k, device=self.device, dtype=self.dtype)
            v = torch.zeros(shape_v, device=self.device, dtype=self.dtype)
            self._k[layer], self._v[layer] = k, v
            return k, v

    def append(self, request_ids, layer: int, k: torch.Tensor, v: torch.Tensor) -> dict:
        """Write one step's keys and values for several requests, in place.

        `k` and `v` are (rows, kv_heads, head_dim) with one row per entry of `request_ids`. Rows of
        one request must be contiguous and in order, which is how a chunk arrives.

        In place: the previous cache grew by concatenation, which allocated and copied the whole
        history to add a position. Here the write is a slice assignment into a buffer that was
        already the right size, so the cost is the new positions and not the old ones.
        """
        buf_k, buf_v = self._buffers(layer)
        held: dict[int, int] = {}
        for row, request_id in enumerate(int(r) for r in request_ids):
            slot = self.slot_of(request_id)
            with self._lock:
                at = self._held.get((slot, layer), 0)
                if at >= self.max_context:
                    raise RuntimeError(
                        f"request {request_id} layer {layer} reached {at} cached position(s), "
                        f"past the {self.max_context} this pool has room for. An append-only "
                        f"cache cannot evict; it refuses rather than overwriting."
                    )
                self._held[(slot, layer)] = at + 1
            buf_k[slot, :, at, :] = k[row]
            buf_v[slot, :, at, :] = v[row]
            held[request_id] = at + 1
        return held

    def gather(self, request_ids, layer: int) -> tuple:
        """The slot indices for these requests, and this layer's buffers.

        Returns indices rather than gathered tensors: indexing the buffer with them produces a
        (rows, kv_heads, max_context, dim) VIEW when the rows are distinct, and a sweep contracts
        against that directly. Materialising a gathered copy would put back the allocation this
        class exists to remove.
        """
        buf_k, buf_v = self._buffers(layer)
        pairs = [(r, self._table.of(int(r), create=False)) for r in request_ids]
        with self._lock:
            slots, lengths = [], []
            for request_id, slot in ((int(r), s) for r, s in pairs):
                if slot is None:
                    raise KeyError(
                        f"request {request_id} has no slot at layer {layer}: nothing was ever "
                        f"appended for it, so a sweep would read a history that does not exist"
                    )
                slots.append(slot)
                lengths.append(self._held.get((slot, layer), 0))
        return (
            torch.tensor(slots, device=self.device, dtype=torch.long),
            torch.tensor(lengths, device=self.device, dtype=torch.long),
            buf_k,
            buf_v,
        )

    def bytes_held(self) -> int:
        with self._lock:
            return sum(
                t.numel() * t.element_size()
                for t in list(self._k.values()) + list(self._v.values())
            )

    def report(self) -> dict:
        with self._lock:
            held = sum(
                t.numel() * t.element_size()
                for t in list(self._k.values()) + list(self._v.values())
            )
            layers = len(self._k)
        return {
            **self._table.report(bytes_held=held),
            "layers_allocated": layers,
            "max_context": self.max_context,
        }


def sweep_slots(
    q: torch.Tensor,
    slots: torch.Tensor,
    lengths: torch.Tensor,
    buf_k: torch.Tensor,
    buf_v: torch.Tensor,
    *,
    scaling: float,
    kv_group: int,
) -> tuple:
    """Every row's history swept in one contraction.

    `q` is (rows, heads, head_dim); `slots[i]` says which slot row i reads and `lengths[i]` how far.
    Rows belonging to different requests read different slots, and rows of one chunk read the same
    slot to different depths -- both are the same mask.

    Two things are deliberately not done here, and each replaces a measured cost:

      the grouped-query expansion is a reshape, not a repeat_interleave. Materialising it wrote a
      shared key head out `kv_group` times: 805 MiB of keys at 32k context in float32, per sweep

      the requests are one einsum, not a loop. The loop was 2.67 ms of a 3.6 ms sweep on the live
      pool, and it did not grow with context, because it was per request rather than per position

    The scores are promoted to at least float32 for the reduction over positions; the cache is not,
    because promoting it allocates a copy of the whole history and the tensor cores accumulate in
    float32 anyway.
    """
    rows, heads, head_dim = q.shape
    kv_heads = buf_k.shape[1]
    if kv_heads * kv_group != heads:
        raise RuntimeError(
            f"the sweep was given {heads} query head(s) and {kv_heads} key head(s) at a group "
            f"size of {kv_group}; the cache and the query disagree about the model"
        )
    reach = int(lengths.max().item()) if lengths.numel() else 0
    if reach == 0:
        # nothing cached for any row: an empty sum is a -inf partition and a zero output, and the
        # caller's join then takes this step's own token whole
        acc = torch.promote_types(q.dtype, torch.float32)
        return (
            torch.zeros(rows, heads, buf_v.shape[-1], device=q.device, dtype=acc),
            torch.full((rows, heads), float("-inf"), device=q.device, dtype=acc),
        )

    acc = torch.promote_types(q.dtype, torch.float32)
    k = buf_k[slots, :, :reach, :]  # (rows, kv_heads, reach, head_dim)
    v = buf_v[slots, :, :reach, :]
    qg = q.view(rows, kv_heads, kv_group, head_dim)
    if q.dtype != k.dtype:
        qg = qg.to(k.dtype)

    scores = torch.einsum("rgcd,rgjd->rgcj", qg, k).to(acc) * scaling
    j = torch.arange(reach, device=q.device).view(1, 1, 1, reach)
    scores = scores.masked_fill(j >= lengths.view(rows, 1, 1, 1), float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("rgcj,rgjd->rgcd", weights.to(v.dtype), v).to(acc)

    out = out.reshape(rows, heads, buf_v.shape[-1])
    lse = lse.reshape(rows, heads)
    empty = torch.isinf(lse) & (lse < 0)
    if empty.any():
        # softmax over an all -inf row is nan, and a nan reaching the merge poisons a whole
        # generation with nothing downstream checking for it
        out = torch.where(empty.unsqueeze(-1), torch.zeros_like(out), out)
    return out, lse
