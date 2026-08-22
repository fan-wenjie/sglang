"""The host answering the pool's calls for the history it holds.

Under the group cut the pool runs a linear layer's weights and this side holds its recurrent
state, so the pool calls back mid-span. This is what it reaches.

    OP_STATE_READ     q~ -> r = S q~          on the critical path, one contraction, no write
    OP_STATE_UPDATE   k, v, alpha, beta       deferred, no answer, advances the state

The split is the whole reason both fit: the query coefficient folds the key's correction into the
query, so the reading is ONE contraction rather than two, and the update -- which needs its own
reading of the key -- is not on the path the pool is waiting on.

## Why the decay is not applied here

`alpha` is a per-head scalar the pool computed from the layer's own parameters. Sending it across
so this side could multiply by it and send the product back would be moving a value to where it is
not, to do arithmetic that costs nothing, and moving the answer back the same size.

## Row ids

Every frame carries which request each row belongs to. Both states ARE the whole history
compressed, so a row served from the wrong slot produces fluent text conditioned on somebody
else's prompt, and there is no output symptom for it. The service refuses a frame whose row count
and id count disagree rather than broadcasting.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.afd.protocol import OP_STATE_READ, OP_STATE_SCAN, OP_STATE_UPDATE

logger = logging.getLogger(__name__)


_ORDER = {}


def _trace_order(frame) -> None:
    """The sequence of ops this side receives for one (request, layer), when SGLANG_AFD_ORDER is set.

    The pool asks for a reading and then defers the update, and the correctness of every decode
    step rests on those arriving in that order and being applied before the next read. Nothing has
    ever checked it: the comparisons built so far install a LOCAL stub for `ask_host`, so the wire
    and its ordering are outside all of them.

    A decode must read then update, alternating. A prefill chunk is one scan and no separate
    update, because a multi-row rider advances its own state inside the scan. Anything else --
    two reads with no update between them, an update before its read, a scan followed by an update
    for the same rows -- is a state one step out of place, and it stays fluent.
    """
    import os

    from sglang.srt.afd.protocol import OP_NAMES

    if not os.environ.get("SGLANG_AFD_ORDER"):
        return
    key = (frame.request_id, frame.layer)
    seq = _ORDER.setdefault(key, [])
    if len(seq) >= 6:
        return
    seq.append(OP_NAMES.get(frame.op, frame.op))
    if len(seq) == 6:
        logger.info("afd order: request %s layer %s -- %s",
                    frame.request_id, frame.layer, " ".join(seq))


class HistoryService:
    """Answers the two calls, against a `HistoryCache` this host owns."""

    def __init__(self, cache, *, rows_of):
        self.cache = cache
        # how a frame says whose rows it carries. Passed in rather than read off the frame,
        # because the host knows its own batch layout and the pool only knows the order it sent
        self.rows_of = rows_of
        self.reads = 0
        self.updates = 0

    def __call__(self, frame):
        _trace_order(frame)
        if frame.op == OP_STATE_SCAN:
            return (self._scan(frame),)
        if frame.op == OP_STATE_READ:
            return (self._read(frame),)
        if frame.op == OP_STATE_UPDATE:
            self._update(frame)
            return None
        raise RuntimeError(
            f"the pool sent op {frame.op}, which is not one this history answers. The two ends "
            f"disagree about what lives here."
        )

    def forget(self, request_id: int) -> bool:
        """Drop a request's recurrent state. Returns whether a slot was held.

        Called when a row id BEGINS a request rather than when one ends: an aborted or crashed
        request never sends its ending, and the slot it leaves behind is indistinguishable from a
        slot in use. What starts a request is knowable from the batch -- a prefill chunk with no
        cached prefix -- and it is knowable every time.
        """
        return self.cache.release(int(request_id))

    def _read(self, frame) -> torch.Tensor:
        from sglang.srt.afd.split_read_kernel import read_one

        q_tilde = self._as_heads(frame.tensor, frame)
        slots = self._slots(frame, q_tilde.shape[0])
        self.reads += 1
        return read_one(self.cache.state[frame.layer], slots, q_tilde).reshape(
            q_tilde.shape[0], -1)

    def _scan(self, frame) -> torch.Tensor:
        """A prefill chunk: read AND advance, one token at a time, in order.

        This is where the arrangement's first wrong output came from. `_read` contracts every row
        against the slot's state in one call, which is right for a decode batch -- one token from
        each of several requests, no two sharing a slot -- and wrong for a chunk, where all the
        rows are ONE request's and each reads what its predecessor wrote. Batched, the state never
        advances and the model has no memory of its own prompt: it repeats the last prompt token,
        fluently, with nothing raising.

        What it returns is the RAW reading, one a token, exactly as `_read` does -- not the mixed
        output. `linear_history.prefill_scan` computes the mix, which is the same arithmetic seen
        from the weight side, and wiring it here made the pool mix a mixed value. The two are one
        function apart and neither raises; the test below is against N separate read/update pairs
        for that reason, because that pair IS what a scan has to equal.
        """
        from sglang.srt.afd.split_read_kernel import read_one, update_only

        if len(frame.tensors) != 5:
            raise RuntimeError(
                f"a scan carried {len(frame.tensors)} tensor(s) where the coefficient, the key, "
                f"the value and the two gates were expected. A chunk cannot be advanced by a "
                f"subset of them, and reading without advancing is the bug this op exists for."
            )
        q, k, v, alpha, beta = frame.tensors
        q = self._as_heads(q, frame)
        k = self._as_heads(k, frame)
        v = self._as_heads(v, frame, value=True)
        slots = self._slots(frame, q.shape[0])
        one = int(slots[0])
        if not bool((slots == slots[0]).all()):
            raise RuntimeError(
                f"a scan for layer {frame.layer} spans more than one slot. A chunk is one "
                f"request's own tokens; rows from different requests are independent and belong "
                f"in a read, not a scan, and scanning them would thread one request's history "
                f"through another's."
            )
        device = self.cache.state.device
        alpha, beta = alpha.to(device).float(), beta.to(device).float()
        state = self.cache.state[frame.layer]
        readings = []
        for t in range(q.shape[0]):
            one_slot = slots[t : t + 1]
            readings.append(read_one(state, one_slot, q[t : t + 1]))
            update_only(state, one_slot, k=k[t : t + 1], v=v[t : t + 1],
                        alpha=alpha[t : t + 1], beta=beta[t : t + 1])
        self.reads += 1
        self.updates += 1
        return torch.cat(readings, dim=0).reshape(q.shape[0], -1)

    def _update(self, frame) -> None:
        from sglang.srt.afd.split_read_kernel import update_only

        if len(frame.tensors) != 4:
            raise RuntimeError(
                f"a state update carried {len(frame.tensors)} tensor(s) where the key, the value "
                f"and the two gates were expected. Applying three of them would advance the state "
                f"by something that is not the model."
            )
        k, v, alpha, beta = frame.tensors
        k = self._as_heads(k, frame)
        v = self._as_heads(v, frame, value=True)
        slots = self._slots(frame, k.shape[0])
        update_only(self.cache.state[frame.layer], slots, k=k, v=v,
                    alpha=alpha.to(k.device).float(), beta=beta.to(k.device).float())
        self.updates += 1

    def _as_heads(self, flat: torch.Tensor, frame, *, value: bool = False) -> torch.Tensor:
        dim = self.cache.head_v_dim if value else self.cache.head_k_dim
        return flat.to(self.cache.state.device).float().reshape(
            flat.shape[0], self.cache.value_heads, dim)

    def _slots(self, frame, rows: int) -> torch.Tensor:
        ids = self.rows_of(frame)
        if len(ids) != rows:
            raise RuntimeError(
                f"{len(ids)} row id(s) for {rows} row(s) in a {frame.op} at layer {frame.layer}. "
                f"Every row has to say whose history it reads, and a mismatch folds one request's "
                f"token into another's with nothing in the output to say so."
            )
        return torch.tensor([self.cache.slot_of(int(r)) for r in ids],
                            device=self.cache.state.device, dtype=torch.int32)

    def report(self) -> dict:
        return {"reads": self.reads, "updates": self.updates, **self.cache.report()}
