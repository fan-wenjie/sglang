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
import time

import torch

from sglang.srt.afd.protocol import (
    OP_STATE_MIX,
    OP_STATE_READ,
    OP_STATE_SCAN,
    OP_STATE_UPDATE,
)

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
        logger.info(
            "afd order: request %s layer %s -- %s",
            frame.request_id,
            frame.layer,
            " ".join(seq),
        )


def _refuse_split_runs(slots, layer: int) -> None:
    """A chunk may carry several requests, but each one's rows must be a single contiguous run.

    A scan advances each row against its OWN slot, one token at a time, in order -- so a prefill
    batch holding several new requests is served correctly as it stands, provided the rows arrive
    grouped by request. `SpanRouting._row_ids` guarantees exactly that: it expands each request's
    pool index by that request's extend length, so the rows come out grouped and in order.

    This used to refuse any chunk whose rows did not all share ONE slot, and the cost was not a
    wrong answer but a dead server: the arrangement served one request at a time and killed the
    scheduler on the second. Concurrency 1 answered, concurrency 2 raised here, inside the pool
    client's receiver thread, and the host went down with `PoolClosed`.

    What must still be refused is a slot appearing in two SEPARATE runs -- one request's tokens
    split around another's. The per-row loop would scan them as though the gap were not there,
    which threads a history through positions it never saw, and nothing downstream would say so.
    """
    seen, previous = set(), None
    for slot in (int(s) for s in slots):
        if slot == previous:
            continue
        if slot in seen:
            raise RuntimeError(
                f"a scan for layer {layer} carries slot {slot} in more than one run. A request's "
                f"tokens are contiguous within a chunk; rows split around another request's would "
                f"be scanned as though the gap were not there, threading one request's history "
                f"through another's."
            )
        seen.add(slot)
        previous = slot


# Ops this service does not answer itself. A handler is `(service, frame) -> reply or None`, and
# it may leave a contraction in `service.precontracted` for the mix that follows.
#
# A registry rather than a branch, because the arithmetic that would go in the branch belongs to
# whoever owns the op: this file is the plain arrangement, and something that is not it registers
# instead of being named here. Deleting the package that registers deletes the behaviour.
_REGISTERED: dict = {}


def register_op(op: int, handler) -> None:
    """Claim an op for a handler. Once, and by whoever owns the arithmetic."""
    if op in _REGISTERED:
        from sglang.srt.afd.protocol import OP_NAMES

        raise ValueError(
            f"op {OP_NAMES.get(op, op)} already has a handler on this history. Import order "
            f"would otherwise decide which arithmetic a caller got."
        )
    _REGISTERED[op] = handler


class HistoryService:
    """Answers the two calls, against a `HistoryCache` this host owns."""

    def __init__(self, cache, *, rows_of, conv_weight=None, dims=None):
        self.cache = cache
        # `layer -> conv1d weight`, for the arrangement where the CONVOLUTION runs here. None means
        # this host does not run it and a MIX frame is refused by name rather than convolved
        # against nothing. The weight is the model's and this side has to hold it -- 80 KiB a
        # layer, 3.76 MiB for all 48 on this model, measured from the checkpoint.
        self.conv_weight = conv_weight
        # (key_heads, value_heads, head_k_dim, head_v_dim). Needed to split `mixed_qkv`, and the
        # split width is the one thing not derivable from the frame: two key-width blocks and then
        # whatever is left.
        self.dims = dims
        # how a frame says whose rows it carries. Passed in rather than read off the frame,
        # because the host knows its own batch layout and the pool only knows the order it sent
        self.rows_of = rows_of
        self.reads = 0
        self.updates = 0
        # An advance computed by `_mix` and not yet applied. See `drain`.
        self._parked = None
        # Contractions a registered handler computed ahead of the mix that consumes them, keyed
        # by (request, layer) and cleared as they are taken. Empty unless something registers.
        self.precontracted: dict = {}
        # how many rows took a precontracted reading rather than reading here, for the report
        self.precontracted_used = 0
        # the three parts of a read, summed. See `_count_read`.
        self._read_stage_s = 0.0
        self._read_slot_s = 0.0
        self._read_kernel_s = 0.0
        # the five parts of a mix, summed. See `_count_mix`.
        self._mix_stage_s = [0.0] * 5

    def __call__(self, frame):
        _trace_order(frame)
        if frame.op == OP_STATE_SCAN:
            return (self._scan(frame),)
        if frame.op == OP_STATE_READ:
            return (self._read(frame),)
        if frame.op == OP_STATE_MIX:
            return (self._mix(frame),)
        extra = _REGISTERED.get(frame.op)
        if extra is not None:
            return extra(self, frame)
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
        """The pool blocks on this. Timed in three parts, because the whole callback is not.

        The round trip costs the pool about 4.6 ms and this side is a small matrix-vector product
        on a GPU that is idle for most of it, so which of the three parts is being paid for is a
        real question. Measured on the deployment the answer is: none of them -- 0.038 ms staging,
        0.026 slots, 0.129 launch, and the rest of the 4.6 is wire.

        Kept anyway, because the alternative is what produced the wrong answer first: an average
        over `_answer_inbound`, which mixes READ with UPDATE and SCAN, gave a host-compute figure
        ten times too large and pointed a round of work at the wrong half of the system.
        """
        from sglang.srt.afd.split_read_kernel import read_one

        began = time.perf_counter()
        q_tilde = self._as_heads(frame.tensor, frame)
        staged = time.perf_counter()
        slots = self._slots(frame, q_tilde.shape[0])
        located = time.perf_counter()
        self.reads += 1
        out = read_one(self.cache.state[frame.layer], slots, q_tilde).reshape(
            q_tilde.shape[0], -1
        )
        self._count_read(
            staged - began, located - staged, time.perf_counter() - located
        )
        return out

    def _count_read(self, stage_s: float, slot_s: float, kernel_s: float) -> None:
        """Every 500 reads, the three parts of one. See `_read`.

        `kernel_s` is the LAUNCH, not the kernel: nothing synchronises here, so the work lands on
        the stream and this returns. A large launch would itself mean the copy above had not
        finished.
        """
        self._read_stage_s += stage_s
        self._read_slot_s += slot_s
        self._read_kernel_s += kernel_s
        if self.reads % 500:
            return
        n = self.reads
        logger.info(
            "afd host: %s read(s) -- %.3f ms staging, %.3f ms slots, %.3f ms launch",
            n,
            1e3 * self._read_stage_s / n,
            1e3 * self._read_slot_s / n,
            1e3 * self._read_kernel_s / n,
        )

    def _scan(self, frame) -> torch.Tensor:
        """A prefill chunk: read AND advance, one token at a time, in order.

        This is where the arrangement's first wrong output came from. `_read` contracts every row
        against the slot's state in one call, which is right for a decode batch -- one token from
        each of several requests, no two sharing a slot -- and wrong for a chunk, where consecutive
        rows belong to ONE request and each reads what its predecessor wrote. Batched, the state
        never advances and the model has no memory of its own prompt: it repeats the last prompt
        token, fluently, with nothing raising.

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
        _refuse_split_runs(slots, frame.layer)
        device = self.cache.state.device
        alpha, beta = alpha.to(device).float(), beta.to(device).float()
        state = self.cache.state[frame.layer]
        readings = []
        for t in range(q.shape[0]):
            one_slot = slots[t : t + 1]
            readings.append(read_one(state, one_slot, q[t : t + 1]))
            update_only(
                state,
                one_slot,
                k=k[t : t + 1],
                v=v[t : t + 1],
                alpha=alpha[t : t + 1],
                beta=beta[t : t + 1],
            )
        self.reads += 1
        self.updates += 1
        return torch.cat(readings, dim=0).reshape(q.shape[0], -1)

    def _mix(self, frame) -> torch.Tensor:
        """The whole of a linear layer that touches history, for a pool that holds none of it.

        The pool sends the PRE-convolution `[q | k | v]` with the gates; this convolves against the
        ring THIS side keeps, contracts the state, advances it, and answers with `core`. The pool
        applies the z-gated norm and `out_proj`, so `z` never travels.

        This is what makes the pool unconditionally stateless: with the ring here, a request is no
        longer sticky to the pool that served its previous call, which is what multi-pool routing
        and pool replacement rest on.

        Both halves are the pool's own, called rather than reimplemented -- `convolve_with_ring`
        was lifted out of the pool's runner unchanged and `core_from_mixed` is proved bit-identical
        against the pool's inline version.
        """
        from sglang.srt.afd.linear_history import (
            _row_index,
            convolve_with_ring,
            core_from_mixed,
        )
        from sglang.srt.afd.slots import _runs
        from sglang.srt.afd.split_read_kernel import read_one

        if self.conv_weight is None or self.dims is None:
            raise RuntimeError(
                "a MIX frame reached a history that does not run the convolution. The caller "
                "moved its ring here and this end was not built to hold one, so the two were "
                "started for different arrangements. Refused rather than convolved against an "
                "empty ring, which would answer fluently and be wrong from the first token."
            )
        if len(frame.tensors) != 3:
            raise RuntimeError(
                f"a mix carried {len(frame.tensors)} tensor(s) where the pre-convolution qkv and "
                f"the two gates were expected."
            )
        if self._parked is not None:
            # The read below takes S_(t-1) for THIS layer, and an advance still parked means the
            # previous call's S_t was never applied. Answering anyway returns a core contracted
            # against a state one token stale -- fluent output, wrong from here on, and nothing
            # downstream can see it. Whoever serves this stopped calling `drain` after the reply.
            raise RuntimeError(
                f"a MIX for layer {frame.layer} reached this history with layer "
                f"{self._parked[0]}'s advance still parked. The caller of this service is not "
                f"draining between frames, so this read would contract against a stale state."
            )
        began = time.perf_counter()
        mixed_qkv, alpha, beta = frame.tensors
        key_heads, value_heads, head_k_dim, head_v_dim = self.dims
        device = self.cache.state.device
        # To the RING's dtype, not merely to its device. The ring here is bfloat16 while the
        # projection arrives in whatever the pool sent, and the pool's own ring was created with
        # the qkv's dtype -- so this is where the two arrangements would round differently if the
        # cast were left implicit. Stated because "where each side rounds" is the thing that has
        # been checked least in this arrangement.
        mixed_qkv = mixed_qkv.to(device=device, dtype=self.cache.conv.dtype)
        alpha = alpha.to(device).float()
        beta = beta.to(device).float()

        cast_at = time.perf_counter()
        ids = [int(r) for r in self.rows_of(frame)]
        runs = _runs(ids)
        slots = [self.cache.slot_of(r) for r, _, _ in runs]
        located = time.perf_counter()
        mixed = convolve_with_ring(
            self.cache.conv[frame.layer],
            mixed_qkv,
            self.conv_weight(frame.layer).to(device),
            slots=slots,
            runs=runs,
        )
        convolved = time.perf_counter()
        rows = _row_index([self.cache.slot_of(r) for r in ids], device)
        # Taken BEFORE the mix, because the mix needs both halves: the contraction, and the query
        # it was made with. `_reading` returns the pair or does the read itself and returns no
        # query, which is the arrangement without a handler registered.
        reading, early_query = self._reading(frame, ids, rows, None)
        core, (k, v) = core_from_mixed(
            mixed,
            alpha=alpha,
            beta=beta,
            key_heads=key_heads,
            value_heads=value_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            read_state=(
                (lambda _q: reading)
                if reading is not None
                else (
                    lambda q_tilde: read_one(
                        self.cache.state[frame.layer], rows, q_tilde
                    ).reshape(q_tilde.shape[0], value_heads, head_v_dim)
                )
            ),
            query=early_query,
        )
        # NOT applied here. The pool is blocked on `core` and nothing in `core` depends on the
        # advance -- the read above took S_(t-1) deliberately. So the advance is parked and the
        # inbound worker applies it AFTER the reply has gone out; it has to land before the next
        # call reads this layer, and that call cannot be served until the drain has run.
        #
        # MEASURED as 0.415 ms of a 2.731 ms callback, five stages of this method timed on the
        # deployment: cast 0.169, slots 0.009, convolve 0.297, core 0.826, update 0.415.
        mixed_at = time.perf_counter()
        self._parked = (frame.layer, rows, k, v, alpha, beta)
        self.reads += 1
        self._count_mix(
            cast_at - began,
            located - cast_at,
            convolved - located,
            mixed_at - convolved,
            time.perf_counter() - mixed_at,
        )
        return core

    def _count_mix(self, *stages: float) -> None:
        """Every 500 mixes, the five parts of one. See `_mix`.

        Launch, not kernel: nothing in `_mix` synchronises, so a large number here is CPU work --
        Python and kernel launches -- and not the card. That is what these five said the first
        time they were read, and it moved a round of work off the wire and onto this process.
        """
        for i, stage in enumerate(stages):
            self._mix_stage_s[i] += stage
        if self.reads % 500:
            return
        cast, slots, convolve, core, park = (
            x * 1e3 / self.reads for x in self._mix_stage_s
        )
        logger.info(
            "afd host: %d mix(es) -- cast %.3f, slots %.3f, convolve %.3f, core %.3f, "
            "park %.3f ms",
            self.reads,
            cast,
            slots,
            convolve,
            core,
            park,
        )

    def _reading(self, frame, ids, rows, q_tilde):
        """The contraction this mix needs: a precontracted one if a handler left it, else its own.

        `core_from_mixed` builds `q_tilde` from this step's projection and hands it here. A
        registered handler may have contracted the state already, with a coefficient of its own
        and against the state as it stood before this step; when it has, that is the reading and
        this one is discarded.

        Nothing registers by default, so by default this reads. The registry is the whole of the
        difference, and removing what registers removes the behaviour rather than disabling it.
        """

        _, value_heads, _, head_v_dim = self.dims
        kept = [self.precontracted.pop((int(r), int(frame.layer)), None) for r in ids]
        if all(k is not None for k in kept):
            self.precontracted_used += len(kept)
            return (
                torch.cat([r for r, _ in kept], dim=0),
                torch.cat([q for _, q in kept], dim=0),
            )
        if any(k is not None for k in kept):
            # Some rows on this bus were precontracted and some were not. Taking both would
            # contract one request's history one way and its neighbour's another, inside one
            # departure, with no symptom -- so it is refused rather than patched over.
            raise RuntimeError(
                f"layer {frame.layer}'s mix found a precontracted reading for some of its rows "
                f"and not others. The two ends disagree about which rows have one."
            )
        # Nothing was kept, so this side does the read -- which needs the coefficient, which
        # needs the current projection. Returning None for both says so, and `core_from_mixed`
        # then does what it did before any of this: forms the coefficient and calls back.
        return None, None

    def drain(self) -> None:
        """Apply the advance parked by `_mix`, after its reply is on the wire.

        Called by the inbound worker between sending one reply and taking the next frame, which
        is what makes the ordering safe without a lock: the worker is one thread, so a parked
        advance is always applied before the next call that could read the state it advances.
        """
        if self._parked is None:
            return
        from sglang.srt.afd.split_read_kernel import update_only

        layer, rows, k, v, alpha, beta = self._parked
        self._parked = None
        update_only(self.cache.state[layer], rows, k=k, v=v, alpha=alpha, beta=beta)
        self.updates += 1

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
        update_only(
            self.cache.state[frame.layer],
            slots,
            k=k,
            v=v,
            alpha=alpha.to(k.device).float(),
            beta=beta.to(k.device).float(),
        )
        self.updates += 1

    def _as_heads(
        self, flat: torch.Tensor, frame, *, value: bool = False
    ) -> torch.Tensor:
        dim = self.cache.head_v_dim if value else self.cache.head_k_dim
        return (
            flat.to(self.cache.state.device)
            .float()
            .reshape(flat.shape[0], self.cache.value_heads, dim)
        )

    def _slots(self, frame, rows: int) -> torch.Tensor:
        ids = self.rows_of(frame)
        if len(ids) != rows:
            raise RuntimeError(
                f"{len(ids)} row id(s) for {rows} row(s) in a {frame.op} at layer {frame.layer}. "
                f"Every row has to say whose history it reads, and a mismatch folds one request's "
                f"token into another's with nothing in the output to say so."
            )
        return torch.tensor(
            [self.cache.slot_of(int(r)) for r in ids],
            device=self.cache.state.device,
            dtype=torch.int32,
        )

    def report(self) -> dict:
        return {"reads": self.reads, "updates": self.updates, **self.cache.report()}
