"""The transport contract, pinned in the shape an RDMA implementation has to satisfy.

This interface exists to be replaced. The measurements say a round trip is 480 us on this overlay
and about 10 us on RoCE, so the fabric will change; what these cases protect is that changing it
does not require moving anything above it.

Each case here is a property an RDMA transport also has to have. A TCP implementation that quietly
broke one -- by owning a buffer, by blocking on a post, by clamping a bad range -- would let
callers grow a dependency the next fabric cannot honour.
"""

import socket
import threading
import time
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.transport import (
    DONE,
    FAILED,
    PENDING,
    BufferRing,
    Region,
    SocketTransport,
)
from sglang.test.test_utils import CustomTestCase


class TestARegionDoesNotOwnItsMemory(CustomTestCase):
    """The caller keeps a tensor alive across a transfer; the transport only addresses it.

    A transport that copied on register would be doing the copy RDMA registration exists to avoid,
    and a caller written against it would be reusing buffers that are not the ones being sent.
    """

    def test_writing_through_the_region_is_visible_to_the_owner(self):
        owned = bytearray(64)
        region = SocketTransport().register(owned)
        region.slice(0, 4)[:] = b"abcd"
        self.assertEqual(bytes(owned[:4]), b"abcd")

    def test_a_typed_view_is_cast_rather_than_refused(self):
        """An RDMA scatter-gather entry is bytes; a typed view has strides it cannot express."""
        import array

        typed = memoryview(array.array("i", [1, 2, 3, 4]))
        region = Region(typed)
        self.assertEqual(region.view.format, "B")
        self.assertEqual(region.nbytes, 16)


class TestARangeOutsideARegionIsRefused(CustomTestCase):
    """Clamping would send the wrong bytes and report success.

    The failure this guards is the quiet one: a length computed from a stale shape reads past the
    end, a transport that clamps sends fewer bytes than the header promised, and the far end reads
    the remainder as the next frame's header.
    """

    def test_past_the_end(self):
        region = SocketTransport().register(bytearray(16))
        with self.assertRaises(ValueError):
            region.slice(8, 16)

    def test_negative(self):
        region = SocketTransport().register(bytearray(16))
        with self.assertRaises(ValueError):
            region.slice(-1, 4)


class TestPostingDoesNotBlock(CustomTestCase):
    """A receive posted before its data exists must return immediately, still pending.

    This is the property the whole schedule rests on: the arrangement issues a transfer and then
    does other work. A post that waited would make every measurement of the overlap a measurement
    of the wait.
    """

    def test_a_receive_with_no_sender_is_pending(self):
        transport = SocketTransport()
        a, b = socket.socketpair()
        try:
            region = transport.register(bytearray(1024))
            transfer = transport.post_recv(b, region, 0, 1024)
            self.assertEqual(transfer.poll(), PENDING)
        finally:
            a.close()
            b.close()

    def test_the_bytes_arrive_and_match(self):
        transport = SocketTransport()
        a, b = socket.socketpair()
        try:
            payload = bytes(range(256)) * 8
            src = transport.register(bytearray(payload))
            dst = transport.register(bytearray(len(payload)))
            recv = transport.post_recv(b, dst, 0, len(payload))
            transport.post_send(a, src, 0, len(payload))
            self.assertEqual(recv.wait(timeout=20), DONE)
            self.assertEqual(bytes(dst.slice(0, len(payload))), payload)
        finally:
            a.close()
            b.close()


class TestAFailureIsAState(CustomTestCase):
    """A dead peer marks the transfer failed rather than raising on the posting thread.

    An exception on the posting thread lands in whatever was being computed at the time, which in
    this arrangement is somebody else's layer. The state belongs to the transfer.
    """

    def test_sending_to_a_closed_peer(self):
        transport = SocketTransport()
        a, b = socket.socketpair()
        b.close()
        try:
            region = transport.register(bytearray(1 << 16))
            transfer = transport.post_send(a, region, 0, 1 << 16)
            self.assertEqual(transfer.poll(), FAILED)
            self.assertIsNotNone(transfer.error)
        finally:
            a.close()

    def test_a_peer_that_closes_mid_frame(self):
        """A short read is not a short frame: the remainder would be read as the next header.

        Polled rather than waited on, because `wait` re-raises the transfer's error and what is
        being asserted here is that the error became a STATE. A test that caught the exception
        would pass against a transport that raised on the posting thread, which is the behaviour
        the state exists to replace.
        """
        transport = SocketTransport()
        a, b = socket.socketpair()
        try:
            region = transport.register(bytearray(4096))
            transfer = transport.post_recv(b, region, 0, 4096)
            a.sendall(b"\0" * 100)
            a.close()
            deadline = time.monotonic() + 20
            while transfer.poll() == PENDING and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(transfer.poll(), FAILED)
            self.assertIsNotNone(transfer.error)
        finally:
            b.close()


class TestTheBufferRing(CustomTestCase):
    """Buffers are registered once and reused, because registration is what RDMA charges for.

    Registering an ibv_mr takes longer than transferring the bytes it describes, so a caller that
    registers per call has built something slower than the socket it replaced. The ring makes
    reuse structural instead of leaving it to each call site.
    """

    def test_it_hands_out_and_takes_back(self):
        ring = BufferRing(SocketTransport(), count=2, nbytes=128)
        first, second = ring.take(), ring.take()
        self.assertEqual(ring.report()["in_flight"], 2)
        ring.give_back(first)
        self.assertEqual(ring.report()["in_flight"], 1)
        ring.give_back(second)
        self.assertEqual(ring.report()["in_flight"], 0)

    def test_exhaustion_is_refused_rather_than_grown(self):
        """Growing here would register memory at the moment the fabric is busiest."""
        ring = BufferRing(SocketTransport(), count=1, nbytes=64)
        ring.take()
        with self.assertRaises(RuntimeError) as caught:
            ring.take()
        self.assertIn("in flight", str(caught.exception))

    def test_the_high_water_mark_is_kept(self):
        """It is how an operator sizes the ring without guessing."""
        ring = BufferRing(SocketTransport(), count=4, nbytes=64)
        held = [ring.take() for _ in range(3)]
        for region in held:
            ring.give_back(region)
        self.assertEqual(ring.report()["high_water"], 3)

    def test_an_empty_ring_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            BufferRing(SocketTransport(), count=0, nbytes=64)


if __name__ == "__main__":
    unittest.main()
