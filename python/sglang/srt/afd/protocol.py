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
OP_FFN = 0        # h_(l-1) -> x_l
OP_SWEEP = 1      # q, h_(l-1) -> o, lse, x_l
OP_HEAD = 2       # hidden -> logits
OP_RELEASE = 3    # drop one request's cache; sglang reuses slots and the next one is not this one
OP_NAMES = {OP_FFN: "ffn", OP_SWEEP: "sweep", OP_HEAD: "head", OP_RELEASE: "release"}


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
    def one(cls, request_id: int, layer: int, tensor: torch.Tensor, op: int = OP_FFN) -> "Frame":
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


def send_frame(sock, frame: Frame) -> None:
    """Put one frame on the wire without concatenating the payload onto the header.

    `encode` copies every tensor a second time to prepend a header. At tens of kilobytes a call
    and sixty-four calls a token that is real, so the buffers go to `sendmsg` as they are.
    """
    head, views = encode_parts(frame)
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
