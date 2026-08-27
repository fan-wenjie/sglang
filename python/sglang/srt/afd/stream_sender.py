"""Load the streamed wire, build headers from metadata, and register the shared fd locks.

The C++ half is `csrc/stream_sender.cpp`; this half owes the wire two things. The header must
be byte-identical to `protocol.encode_parts`' -- it is built from the same structs against
tensor METADATA, so the payload never visits the interpreter. And a socket with a streamed
sender needs every OTHER writer serialised against the driver-thread callback, so the first
send on a socket registers acquire/release in `protocol.EXTERNAL_WIRE_LOCKS`, and
`send_frame` takes them for such sockets.

Loading is a build (nvcc, once per environment, cached). A pool that cannot build it runs the
queued Python path unchanged -- the wire format is the same bytes either way -- and says so
once at INFO rather than warning on every frame.
"""

from __future__ import annotations

import logging

from sglang.srt.afd.protocol import DTYPE_CODE, DTYPES, HEADER, PART

logger = logging.getLogger(__name__)

_EXT = None
_TRIED = False


def _load():
    global _EXT, _TRIED
    if _TRIED:
        return _EXT
    _TRIED = True
    try:
        import os
        import pathlib

        from torch.utils.cpp_extension import load

        src = pathlib.Path(__file__).parent / "csrc" / "stream_sender.cpp"
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        _EXT = load(
            name="afd_stream_sender",
            sources=[str(src)],
            extra_cflags=["-O3"],
            extra_include_paths=[f"{cuda_home}/include"],
            extra_ldflags=[f"-L{cuda_home}/lib64", f"-L{cuda_home}/lib", "-lcudart"],
            verbose=False,
        )
        logger.info("afd: the streamed wire is up; frames stage on the stream")
    except (
        Exception
    ):  # noqa: BLE001 -- a missing toolchain is a configuration, not a bug
        logger.info(
            "afd: the streamed wire did not build; frames take the queued Python path",
            exc_info=True,
        )
        _EXT = None
    return _EXT


def streamed_send(sock, request_id: int, layer: int, op: int, tensors) -> bool:
    """Send one frame with the payload staged on the current stream. False = not available."""
    ext = _load()
    if ext is None:
        return False
    parts, total = [], 0
    for t in tensors:
        if t.dtype not in DTYPE_CODE:
            raise ValueError(f"{t.dtype} is not on the wire's dtype list {DTYPES}")
        if t.dim() != 2:
            raise ValueError(f"a frame carries (rows, columns); got {tuple(t.shape)}")
        if not t.is_cuda:
            return False
        parts.append(PART.pack(t.shape[0], t.shape[1], DTYPE_CODE[t.dtype], t.nbytes))
        total += t.nbytes
    head = HEADER.pack(request_id, layer, op, len(tensors), total)
    fd = sock.fileno()
    ext.send_frame_streamed(
        fd, head + b"".join(parts), [t.contiguous() for t in tensors]
    )
    _register_lock(fd)
    return True


def _register_lock(fd: int) -> None:
    from sglang.srt.afd import protocol

    if fd in protocol.EXTERNAL_WIRE_LOCKS:
        return
    ext = _EXT
    protocol.EXTERNAL_WIRE_LOCKS[fd] = (
        lambda: ext.acquire_fd(fd),
        lambda: ext.release_fd(fd),
    )
