"""The streamed wire's frames are the Python wire's frames, byte for byte, in order.

The C++ sender stages the payload on the stream and writes from a driver-thread callback;
what could go wrong is exactly what these hold shut: a header that drifts from
`encode_parts`' bytes, payloads landing out of order, a Python `send_frame` interleaving
mid-frame with the callback, and a frame read back as something other than what was sent.
"""

import socket
import threading
import unittest

import torch

from sglang.srt.afd.protocol import OP_STATE_UPDATE, Frame, decode, send_frame
from sglang.test.test_utils import CustomTestCase

CUDA = torch.cuda.is_available()


def _loaded():
    from sglang.srt.afd.stream_sender import _load

    return _load() is not None


@unittest.skipUnless(
    CUDA and _loaded(), "the streamed wire needs a GPU and a toolchain"
)
class TestTheStreamedWireIsTheWire(CustomTestCase):
    def _pair(self):
        a, b = socket.socketpair()
        a.setblocking(True)
        b.setblocking(True)
        return a, b

    def test_one_frame_round_trips(self):
        from sglang.srt.afd.stream_sender import streamed_send

        a, b = self._pair()
        try:
            q = torch.randn(2, 128, device="cuda")
            k = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
            self.assertTrue(streamed_send(a, 7, 3, OP_STATE_UPDATE, (q, k)))
            torch.cuda.synchronize()
            got = decode(b)
            self.assertEqual(
                (got.request_id, got.layer, got.op), (7, 3, OP_STATE_UPDATE)
            )
            torch.testing.assert_close(got.tensors[0], q.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(got.tensors[1], k.cpu(), rtol=0, atol=0)
        finally:
            a.close()
            b.close()

    def test_frames_arrive_in_stream_order(self):
        from sglang.srt.afd.stream_sender import streamed_send

        a, b = self._pair()
        try:
            sent = [torch.full((1, 32), float(i), device="cuda") for i in range(20)]
            for i, t in enumerate(sent):
                self.assertTrue(streamed_send(a, i, 0, OP_STATE_UPDATE, (t,)))
            torch.cuda.synchronize()
            for i in range(20):
                got = decode(b)
                self.assertEqual(got.request_id, i)
                torch.testing.assert_close(
                    got.tensors[0], sent[i].cpu(), rtol=0, atol=0
                )
        finally:
            a.close()
            b.close()

    def test_a_python_writer_does_not_interleave(self):
        """`send_frame` on a streamed socket takes the C side's mutex, so a frame written from
        Python lands whole between callback frames rather than inside one."""
        from sglang.srt.afd.stream_sender import streamed_send

        a, b = self._pair()
        try:
            for i in range(10):
                gpu = torch.full((1, 4096), float(i), device="cuda")
                self.assertTrue(streamed_send(a, 100 + i, 0, OP_STATE_UPDATE, (gpu,)))
                send_frame(
                    a,
                    Frame(200 + i, 0, (torch.full((1, 8), float(i)),), OP_STATE_UPDATE),
                )
            torch.cuda.synchronize()
            seen = sorted(decode(b).request_id for _ in range(20))
            self.assertEqual(
                seen, sorted(list(range(100, 110)) + list(range(200, 210)))
            )
        finally:
            a.close()
            b.close()

    def test_a_python_writer_does_not_hold_the_lock_across_the_stream(self):
        """A `send_frame` racing a still-queued callback finishes rather than deadlocking.

        The callback takes the wire mutex from inside the stream; `send_frame`'s encoding
        waits on that same stream. Held in the wrong order -- lock first, then encode --
        the two wait on each other forever, which is how a whole deployment once hung on
        its first span. The stream is plugged with a sleep so the callback cannot have run
        yet, exactly the live window."""
        from sglang.srt.afd.stream_sender import streamed_send

        a, b = self._pair()
        try:
            gpu = torch.full((1, 4096), 7.0, device="cuda")
            small = torch.ones(1, 8, device="cuda")
            torch.cuda.synchronize()
            torch.cuda._sleep(
                3 * 10**9
            )  # plug the stream; the callback queues behind it
            self.assertTrue(streamed_send(a, 1, 0, OP_STATE_UPDATE, (gpu,)))
            done = []
            writer = threading.Thread(
                target=lambda: (
                    send_frame(a, Frame(2, 0, (small,), OP_STATE_UPDATE)),
                    done.append(True),
                ),
                daemon=True,
            )
            writer.start()
            writer.join(timeout=30)
            self.assertTrue(done, "send_frame deadlocked against the streamed callback")
            seen = sorted(decode(b).request_id for _ in range(2))
            self.assertEqual(seen, [1, 2])
        finally:
            a.close()
            b.close()


class TestTheEncodeComesBeforeTheLock(CustomTestCase):
    """The payload copies wait on the GPU stream and the streamed wire's callback waits on
    the lock, so `send_frame` must finish encoding before it takes the lock -- pinned here
    without a GPU, by recording the order."""

    def test_send_frame_encodes_before_taking_an_external_lock(self):
        from sglang.srt.afd import protocol

        order = []
        real = protocol.encode_parts
        a, b = socket.socketpair()
        fd = a.fileno()
        protocol.encode_parts = lambda f: (order.append("encode"), real(f))[1]
        protocol.EXTERNAL_WIRE_LOCKS[fd] = (
            lambda: order.append("acquire"),
            lambda: order.append("release"),
        )
        try:
            send_frame(a, Frame(3, 0, (torch.ones(1, 4),), OP_STATE_UPDATE))
            self.assertEqual(order, ["encode", "acquire", "release"])
            self.assertEqual(decode(b).request_id, 3)
        finally:
            protocol.encode_parts = real
            protocol.EXTERNAL_WIRE_LOCKS.pop(fd, None)
            a.close()
            b.close()


if __name__ == "__main__":
    unittest.main()
