"""The wire between a host that owns the KV cache and a pool that owns the static weights.

One frame in each direction. A frame carries a batch of tokens rather than one, because the
pool's whole reason for existing is that it answers many callers in one weight read: a dense
feed-forward at decode is bound by reading its weights, not by the arithmetic, so the marginal
cost of a second caller in the same departure is close to nothing.

Every frame is tagged with (request, layer). Keying replies by layer alone is the bug this format
exists to prevent -- two requests in flight at the same layer would overwrite each other's slot,
and the symptom is a model that still produces fluent text from a slightly wrong distribution,
which no assertion downstream catches.

bfloat16 crosses as raw bytes. It has no numpy dtype, so a tensor is viewed as uint8 and the bytes
are the tensor's own; going via float32 would double the traffic and make every byte figure a
fiction.

## Why a frame carries several tensors and an opcode

The first version carried one tensor and meant one thing: run this layer's feed-forward. Moving
the KV cache and the key/value projections to the pool gives it two more jobs, and both need more
than one tensor in each direction:

    FFN     h_(l-1)                 -> x_l              what it always did
    SWEEP   q, h_(l-1)              -> o, lse, x_l      the pool projects k and v ITSELF
    HEAD    the final hidden        -> logits           the last layer's head

SWEEP is the one that pays for the opcode. The pool computed the feed-forward, so it already holds
x_l = h_(l-1) + ffn(h_(l-1)); giving it W_k and W_v means the host never forms a key or a value
and never sends one. The host sends its query and the residual it already had to send anyway.

A tensor list rather than a wider header, because the alternative -- one frame per tensor, matched
by convention at the far end -- makes a lost or reordered frame into a silently mismatched pair.
"""

from __future__ import annotations

import struct
import threading
import weakref
from typing import NamedTuple

import torch

# request id, layer index, opcode, tensor count, total payload length.
HEADER = struct.Struct("!QIIII")
# per tensor: rows, columns, dtype code, byte length
PART = struct.Struct("!IIII")
CLOSE = HEADER.pack(0, 0, 0, 0, 0)

# int64 is on the list because RoPE travels with the frame: the pool projects the key, so it
# needs the positions, and sending them as floats would round past 2^24.
DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.int64)
DTYPE_CODE = {d: i for i, d in enumerate(DTYPES)}

# What the pool is being asked for. The opcode travels because the reply's SHAPE depends on it and
# a receiver that guessed from the byte count would be right until a head dimension changed.
OP_FFN = 0  # h_(l-1) -> x_l
OP_SWEEP = 1  # normalised x_l, positions, ids -> o, lse, q.k_t, v_t
# the query is projected ON THE POOL from that same input, so it never
# crosses the wire in either direction, and the feed-forward is a frame
# of its own
OP_RELEASE = (
    3  # drop one request's cache; sglang reuses slots and the next one is not this one
)
OP_KVPROJ = 4  # normalised x_l -> k, v, gate. The cache stays on the host; only the weights move
OP_SWEEP_Q = (
    5  # q -> o, lse. Sweeps the cache and appends nothing: the two-pool split, where
)
# this frame goes to the CACHE pool at the same moment the feed-forward goes
# to the WEIGHTS pool, because the query is ready a layer early and the
# feed-forward's answer is not needed to sweep positions that predate it
OP_APPEND = (
    6  # k, v -> ack. Off the critical path: this step's join uses the host's own
)
# k and v, and the cache only has to hold them by the NEXT step
OP_HELLO = (
    7  # what each side does, exchanged before the first token. A host that needs a
)
# cache pool and reaches a weights pool otherwise finds out from a frame the
# far end cannot parse, which arrives as "closed mid-call" -- a message about
# the socket that says nothing about the configuration that caused it
OP_LINEAR = 8  # RESERVED, not served, and the number is kept so nothing reuses it.
#
# It ran a linear-attention layer's decode step ON THE POOL, with that request's
# recurrent state held there. The pool server had a complete handler for it and
# nothing ever sent one -- no host built the frame, and no path constructed the
# state table it needed. It was a cheap half-road to moving the linear attention
# and it was not taken, on purpose: a pool holding per-request state stops being
# stateless, so it can no longer be released between one request's own calls,
# and that release is the property the whole arrangement is built on.
#
# OP_LAYER does the same work the other way round -- the weights are here, the
# state stays on the caller, and the pool reads it back. A frame with
# this op is refused BY NAME rather than falling through to the feed-forward
# queue, where it would be half-served and answered with something nobody asked
# for.
OP_SPAN = (
    9  # a whole group of layers: the host's attention output at a full-attention layer
)
# -> the hidden state the NEXT full attention reads. Four feed-forwards and
# three linear attentions, uninterrupted, with the batch fixed for all of it.
# `layer` names the full-attention layer the span starts at
OP_SPAN_Q = (
    10  # the first half of a span's reply: h_(l+3), the shifted read point, sent the
)
# moment it exists -- one feed-forward before the span's own output does. The
# host projects its query and sweeps its cache against it while the pool spends
# that feed-forward. Never a request; only ever a reply
OP_SPAN_ENTER = (
    11  # the layers BELOW the first full attention, fed by the embedding rather than
)
# by an output projection. It is where a request's residual on the pool begins,
# which is why it is a separate opcode: a span that silently started from zero
# would be correct arithmetic over a history the request does not have
OP_SPAN_EXIT = (
    12  # the last full attention's tail: output projection, feed-forward, final norm.
)
# Returns the normalised hidden state, not logits, so that whether the
# language-model head runs on the pool stays a separate and measurable choice
OP_STATE_READ = (
    13  # POOL TO HOST, and the only op that travels that way. The pool holds a
)
# linear layer's weights and the host holds its recurrent state, so the pool
# asks: here is one query coefficient, give me the state contracted against it.
# Carries q~ and the row ids; the reply is the raw reading, without the decay,
# because the decay is a per-head scalar the pool has and a value that crossed
# the wire to be multiplied there and back is a value that should not have gone
OP_STATE_UPDATE = (
    14  # POOL TO HOST, off the critical path. The key, the value and the gates, so
)
# the host can advance the state it holds. Deferred on purpose: the state only
# has to be right by the NEXT step, and that is what keeps the value off the
# critical path entirely
OP_STATE_SCAN = (
    15  # POOL TO HOST, for a run of MORE THAN ONE row -- a prefill chunk, whose
)
# tokens are one request's and sequentially dependent. Carries everything a
# step needs at once (the coefficient, the key, the value, both gates) because
# the read and the update cannot be separated here: token n reads the state
# token n-1 advanced, and a deferred update has not arrived yet. Splitting them
# is what OP_STATE_READ and OP_STATE_UPDATE do, and it is correct only for a
# decode row, where a request contributes exactly one token to the batch
OP_STATE_MIX = (
    17  # POOL TO HOST, and the one that makes the pool hold nothing. The pool
)
# sends the PRE-CONVOLUTION [q | k | v] with alpha and beta; the host
# convolves against its own ring, contracts the state, advances it, and
# answers with `core`. The pool then applies the z-gated norm and out_proj.
#
# It replaces READ/UPDATE/SCAN for a caller that has moved its convolution
# home, and it does not add a crossing: the callback already happened, this
# changes what rides in it. `z` deliberately does not travel -- it is
# projected on the pool and consumed there.
OP_STATE_EARLY = (
    18  # POOL TO HOST: the cooked coefficient down, the reading straight back.
)
# The pool sends the query coefficient `q~` -- convolved, normalised and formed on the pool
# against a window the host attached to the span call -- one feed-forward before it needs the
# answer. The host contracts the state with it and REPLIES with the reading immediately, so
# the answer crosses back while the pool is still inside that feed-forward and is usually on
# the table before anyone asks. That is the whole mechanism: the round trip still happens, it
# just happens where nobody is waiting for it.
#
# Finished materials, deliberately. The first build sent the raw projection and the host
# convolved, normalised and formed the coefficient itself -- 1.207 ms of preparation against a
# 0.461 ms window, so the schedule's max moved to the host and a ladder priced it at 10.69
# percentage points. The host's half of this frame is one contraction and one send.

OP_STATE_MIX_READY = (
    19  # RETIRED, never reused. It carried finished mix materials for a
)
# host that assembled `core` and replied -- a blocking round trip at the exact point the push
# arrangement (EARLY answered + APPLY unanswered) removes. The number stays burned: a pool and
# a host from either side of the change must refuse each other by op, not misread each other
# by shape.
#
# Ordering is what makes it safe, and the transport already gives it: both frames
# go down one socket in order and the far end serves them on one thread, so an
# early read is always applied before the mix that consumes it. A mix that finds
# nothing kept does its own read, which is the arrangement without this frame.

OP_STATE_APPLY = 21  # POOL TO HOST, unanswered: this step's state and ring advance.
# The push arrangement's second half. The EARLY frame is ANSWERED there -- the host contracts
# and sends the reading straight back, inside the feed-forward the pool is spending -- so by
# the time the pool needs it, it is usually already on the table and the round trip has left
# the critical path. What the host still must have is the advance: the key, the value, the two
# gates and the ring's next column, none of which the pool may keep. They ride this frame,
# unanswered, with a whole token of slack: the same socket carries the next EARLY for the same
# layer behind it, and the host serves in order, so the state is advanced before anything can
# read it.

OP_SPAN_LANE = 22  # a middle span whose read triangle -- coefficient down,
# reading up, advance down -- rides the NCCL lane instead of this wire. The HOST decides,
# at issue, one frame at a time: the lane is up, the rows are one decode row, the windows
# are attached. Everything else about the span is OP_SPAN, and a pool without a lane
# refuses the op by name rather than serving half an arrangement.

OP_LAYER = 16  # ONE linear-attention layer's own arithmetic, run on the pool while its
# recurrent state stays on the host. Carries the layer's normalised input and
# its row ids; returns that layer's attention output, nothing more. The residual
# never travels -- the host holds it and does both norms -- so this costs one
# round trip a layer and needs no per-request state on the pool.
#
# The pool answers it by calling back for the state with OP_STATE_READ or
# OP_STATE_SCAN, exactly as a span does. What differs from a span is only the
# granularity: a layer at a time rather than a group.

OP_NAMES = {
    OP_FFN: "ffn",
    OP_SWEEP: "sweep",
    OP_RELEASE: "release",
    OP_KVPROJ: "kvproj",
    OP_SWEEP_Q: "sweep_q",
    OP_APPEND: "append",
    OP_HELLO: "hello",
    OP_LINEAR: "linear",
    OP_SPAN: "span",
    OP_SPAN_Q: "span_q",
    OP_SPAN_ENTER: "span_enter",
    OP_SPAN_EXIT: "span_exit",
    OP_STATE_READ: "state_read",
    OP_STATE_UPDATE: "state_update",
    OP_STATE_SCAN: "state_scan",
    OP_STATE_MIX: "state_mix",
    OP_STATE_EARLY: "state_early",
    OP_STATE_MIX_READY: "state_mix_ready",
    OP_STATE_APPLY: "state_apply",
    OP_SPAN_LANE: "span_lane",
    OP_LAYER: "layer",
}

# What the POOL sends to the HOST, rather than the other way round. A client's reader has to tell
# these from replies: they arrive interleaved with the answers it is waiting for, on the same
# socket, and storing one in the reply table would hang the caller it belongs to and answer a
# different caller with a state reading.
# A SET, not a frozenset: this file lists the ops the base tree's own service answers,
# and a package that registers a handler for more (the early read's two) adds its ops
# here at import, beside its `register_op` -- membership and the handler arrive together
# or not at all, which is what keeps the router from routing what nothing answers.
INBOUND_OPS = {
    OP_STATE_READ,
    OP_STATE_UPDATE,
    OP_STATE_SCAN,
    OP_STATE_MIX,
}


class Frame(NamedTuple):
    """One call or one reply.

    `tensors` are each (rows, columns); what they mean is fixed by `op`, which is why the opcode
    is on the wire rather than inferred. `tensor` is the first of them, for the callers and tests
    that predate the multi-tensor form and only ever send one.
    """

    request_id: int
    layer: int
    tensors: tuple[torch.Tensor, ...]
    op: int = OP_FFN

    @property
    def key(self) -> tuple[int, int]:
        return (self.request_id, self.layer)

    @property
    def tensor(self) -> torch.Tensor:
        return self.tensors[0]

    @classmethod
    def one(
        cls, request_id: int, layer: int, tensor: torch.Tensor, op: int = OP_FFN
    ) -> Frame:
        return cls(request_id, layer, (tensor,), op)


def _payload_of(t: torch.Tensor) -> tuple[memoryview, int, int, int]:
    """One tensor's bytes and the three numbers the far end needs to rebuild it."""
    if t.dtype not in DTYPE_CODE:
        raise ValueError(f"{t.dtype} is not on the wire's dtype list {DTYPES}")
    if t.dim() != 2:
        raise ValueError(f"a frame carries (rows, columns); got {tuple(t.shape)}")
    host = t.detach().to("cpu").contiguous()
    view = memoryview(host.view(torch.uint8).numpy()).cast("B")
    return view, t.shape[0], t.shape[1], DTYPE_CODE[t.dtype]


def encode_parts(frame: Frame) -> tuple[bytes, list[memoryview]]:
    """The header block and the tensors' own bytes, without a copy to join them."""
    views, parts = [], []
    for t in frame.tensors:
        view, rows, cols, code = _payload_of(t)
        views.append(view)
        parts.append(PART.pack(rows, cols, code, len(view)))
    total = sum(len(v) for v in views)
    head = HEADER.pack(frame.request_id, frame.layer, frame.op, len(views), total)
    return head + b"".join(parts), views


def encode(frame: Frame) -> bytes:
    """The whole frame as one bytes object. Convenient, and one copy more than sending it."""
    head, views = encode_parts(frame)
    return head + b"".join(bytes(v) for v in views)


# One lock per socket, weakly held. Two threads legitimately write one socket -- a departure
# inline, its outbox sender behind it -- and their ORDER is settled by events and the queue, but
# nothing below serialises the bytes themselves: two concurrent `sendmsg` calls interleave the
# frames mid-payload and the reader finds a length field inside another frame's tensor. The
# event discipline happens to prevent overlap today at one request in flight; this makes the
# overlap safe instead of merely unexercised.
_send_locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_send_locks_guard = threading.Lock()


def _send_lock_of(sock) -> threading.Lock:
    key = id(sock)
    with _send_locks_guard:
        lock = _send_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _send_locks[key] = lock
            try:
                # tie the lock's life to the socket, so the weak table drops it with the socket
                sock._afd_send_lock_keep = lock
            except AttributeError:
                # a socket type that refuses attributes: keep the lock forever, which leaks one
                # lock per such socket and nothing else
                _send_locks_forever.append(lock)
        return lock


_send_locks_forever: list = []


# A socket whose frames also leave from a driver-thread callback (the streamed wire) registers
# its own acquire/release here, so Python writers and the callback take ONE mutex. Keyed by fd;
# registered by whichever arm built such a wire, when it first sends on a socket.
EXTERNAL_WIRE_LOCKS: dict = {}


def send_frame(sock, frame: Frame) -> None:
    """Put one frame on the wire without concatenating the payload onto the header.

    `encode` copies every tensor a second time to prepend a header. At tens of kilobytes a call
    and sixty-four calls a token that is real, so the buffers go to `sendmsg` as they are.

    Encoding happens BEFORE the wire lock, and that order is load-bearing: `_payload_of` copies
    each tensor to the CPU, which waits on its stream, and the streamed wire's callback takes
    this same lock from inside that stream. Encode under the lock and the two wait on each
    other -- the callback for the lock, the copy for the callback -- with nothing to time out.
    """
    head, views = encode_parts(frame)
    external = EXTERNAL_WIRE_LOCKS.get(sock.fileno()) if EXTERNAL_WIRE_LOCKS else None
    if external is not None:
        acquire, release = external
        acquire()
        try:
            _write_encoded(sock, head, views)
        finally:
            release()
        return
    with _send_lock_of(sock):
        _write_encoded(sock, head, views)


def _write_encoded(sock, head: bytes, views: list) -> None:
    buffers = [head] + views
    total = len(head) + sum(len(v) for v in views)
    sent = sock.sendmsg(buffers)
    while sent < total:
        # a partial write lands somewhere inside one of the buffers; drop the ones it consumed
        # whole and resume inside the one it stopped in
        remaining, offset = [], sent
        for buf in buffers:
            if offset >= len(buf):
                offset -= len(buf)
                continue
            remaining.append(memoryview(buf)[offset:] if offset else buf)
            offset = 0
        sent += sock.sendmsg(remaining)


def _recv_exactly(sock, n: int) -> bytearray | None:
    """Read n bytes or report the peer went away. A short read is not an error to paper over.

    Reads straight into one buffer. A chunk-list-and-join version copies the payload a second
    time, and `torch.frombuffer` then needs a writable buffer, which copies it a third.
    """
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        read = sock.recv_into(view[got:], n - got)
        if not read:
            return None
        got += read
    return buf


def decode(sock) -> Frame | None:
    """Read one frame off a socket. Returns None on a clean close."""
    head = _recv_exactly(sock, HEADER.size)
    if head is None:
        return None
    request_id, layer, op, count, total = HEADER.unpack(head)
    if (request_id, layer, op, count, total) == (0, 0, 0, 0, 0):
        return None
    if count == 0:
        raise ConnectionError(
            f"frame for request {request_id} layer {layer} announced no tensors; a frame with "
            f"nothing in it is a framing error, not an empty batch"
        )
    descriptors = _recv_exactly(sock, PART.size * count)
    if descriptors is None:
        raise ConnectionError(
            f"frame for request {request_id} layer {layer} closed inside its tensor table"
        )
    shapes = [PART.unpack_from(descriptors, i * PART.size) for i in range(count)]
    if sum(s[3] for s in shapes) != total:
        raise ConnectionError(
            f"frame for request {request_id} layer {layer} says {total} payload bytes and its "
            f"{count} tensor(s) account for {sum(s[3] for s in shapes)}; the two disagree and "
            f"only one of them can be used to read the socket"
        )
    payload = _recv_exactly(sock, total)
    if payload is None:
        raise ConnectionError(
            f"frame for request {request_id} layer {layer} announced {total} bytes and the "
            f"connection closed inside it; a truncated frame is not a short frame"
        )
    tensors, offset = [], 0
    for rows, cols, code, length in shapes:
        flat = torch.frombuffer(payload, dtype=torch.uint8, count=length, offset=offset)
        tensors.append(flat.view(DTYPES[code]).view(rows, cols))
        offset += length
    return Frame(request_id, layer, tuple(tensors), op)


def pack_positions(positions: torch.Tensor) -> torch.Tensor:
    """Put a positions tensor on the wire without losing its shape.

    A rotation's positions are one row per token for a text stack and SEVERAL for a multimodal
    one -- mrope carries a temporal, a height and a width row, so the tensor is (3, tokens). The
    wire takes two dimensions, which fits both, and the rule is that the ROW COUNT is the meaning:

        (tokens,)      -> (1, tokens)      one row, flattened again on arrival
        (3, tokens)    -> (3, tokens)      kept, because each row is a different axis

    Flattening the second into (3 * tokens, 1) is what the first version did. It produced a frame
    whose positions were three times as long as its hidden states, and the far end learned about
    it from a broadcast failure inside apply_rotary_emb -- which names neither the frame nor the
    axis it lost.
    """
    if positions.dim() == 1:
        return positions.reshape(1, -1).to(torch.int64)
    if positions.dim() == 2:
        return positions.to(torch.int64)
    raise ValueError(
        f"positions have {positions.dim()} dimension(s); the wire carries one row per rotation "
        f"axis and this is neither a plain sequence nor a multi-axis one"
    )


def unpack_positions(packed: torch.Tensor) -> torch.Tensor:
    """The inverse. One row means a plain sequence; more means the axes are the rows."""
    return packed.reshape(-1) if packed.shape[0] == 1 else packed
