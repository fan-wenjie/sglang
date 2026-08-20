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
"""

from __future__ import annotations

import struct
from typing import NamedTuple

import torch

# request id, layer index, token count, hidden width, dtype code, payload length.
# The shape travels because the receiver must rebuild the tensor without inferring anything: a
# reshape inferred from a byte count is right until the day a dtype changes under it.
HEADER = struct.Struct("!QIIIII")
CLOSE = HEADER.pack(0, 0, 0, 0, 0, 0)

DTYPES = (torch.bfloat16, torch.float16, torch.float32)
DTYPE_CODE = {d: i for i, d in enumerate(DTYPES)}


class Frame(NamedTuple):
    """One call or one reply. `tensor` is (tokens, width)."""

    request_id: int
    layer: int
    tensor: torch.Tensor

    @property
    def key(self) -> tuple[int, int]:
        return (self.request_id, self.layer)


def encode(frame: Frame) -> bytes:
    """Header plus the tensor's own bytes, contiguous and on the CPU."""
    t = frame.tensor
    if t.dtype not in DTYPE_CODE:
        raise ValueError(f"{t.dtype} is not on the wire's dtype list {DTYPES}")
    if t.dim() != 2:
        raise ValueError(f"a frame carries (tokens, width); got {tuple(t.shape)}")
    payload = t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    head = HEADER.pack(
        frame.request_id, frame.layer, t.shape[0], t.shape[1], DTYPE_CODE[t.dtype], len(payload)
    )
    return head + payload


def send_frame(sock, frame: Frame) -> None:
    """Put one frame on the wire without concatenating the header onto the payload.

    `encode` builds `head + payload`, which copies the whole tensor a second time to prepend
    twelve bytes. At 40 KB a call and sixty-four calls a token that is real. `sendmsg` takes the
    two buffers as they are.
    """
    head, payload = encode_parts(frame)
    view = memoryview(payload)
    sent = sock.sendmsg([head, view])
    total = len(head) + len(view)
    while sent < total:
        # a partial write lands somewhere in one of the two buffers; resume from wherever it was
        if sent < len(head):
            sent += sock.sendmsg([memoryview(head)[sent:], view])
        else:
            sent += sock.send(view[sent - len(head):])


def encode_parts(frame: Frame) -> tuple[bytes, memoryview]:
    """The header and the tensor's own bytes, without a copy to join them."""
    t = frame.tensor
    if t.dtype not in DTYPE_CODE:
        raise ValueError(f"{t.dtype} is not on the wire's dtype list {DTYPES}")
    if t.dim() != 2:
        raise ValueError(f"a frame carries (tokens, width); got {tuple(t.shape)}")
    host = t.detach().to("cpu").contiguous()
    payload = memoryview(host.view(torch.uint8).numpy()).cast("B")
    head = HEADER.pack(
        frame.request_id, frame.layer, t.shape[0], t.shape[1], DTYPE_CODE[t.dtype], len(payload)
    )
    return head, payload


def _recv_exactly(sock, n: int) -> bytearray | None:
    """Read n bytes or report the peer went away. A short read is not an error to paper over.

    Reads straight into one buffer. The chunk-list-and-join version copied the payload a second
    time, and `torch.frombuffer` then needed a writable buffer, which copied it a third.
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
    request_id, layer, tokens, width, dtype_code, length = HEADER.unpack(head)
    if (request_id, layer, tokens, width, length) == (0, 0, 0, 0, 0):
        return None
    payload = _recv_exactly(sock, length)
    if payload is None:
        raise ConnectionError(
            f"frame for request {request_id} layer {layer} announced {length} bytes and the "
            f"connection closed inside it; a truncated frame is not a short frame"
        )
    dtype = DTYPES[dtype_code]
    flat = torch.frombuffer(payload, dtype=torch.uint8)
    return Frame(request_id, layer, flat.view(dtype).view(tokens, width))
