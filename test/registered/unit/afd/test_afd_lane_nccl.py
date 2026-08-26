"""The fake pool's NCCL half: a real lane between two processes, no model.

The TCP fake pool (test_afd_fake_pool) proves the frame wire; this proves the
LANE -- rendezvous through the TCPStore, the standalone NCCL pair, and the
fixed serving order -- with a subprocess playing the host end. Both ranks sit
on one device over the socket transport, which is the pair's own fallback
(IB/P2P/SHM are disabled by the lane itself), so what is exercised is exactly
the arrangement's protocol and not the machine's fabric.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

import socket
import subprocess
import sys
import time

import torch

from sglang.test.test_utils import CustomTestCase

_HOST_END = """
import torch
from sglang.srt.afd.lane import Lane

def echo(lane):
    # the fixed order: one (1, 8) ride down, its reading (+1) back up
    buf = lane.recv((1, 8), torch.float32)
    lane.send(buf + 1)

lane = Lane("host", "127.0.0.1", {port}, torch.device("cuda:{dev}"))
lane.serve_loop(echo)
import time
for _ in range(1200):
    if lane.ready or lane.failed:
        break
    time.sleep(0.1)
assert lane.ready, "host lane never came up"
time.sleep(30)  # outlive the exchange; the parent kills us when it has asserted
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@unittest.skipUnless(torch.cuda.is_available(), "the lane is a pair of devices")
class TestTheLaneAgainstAFakePool(CustomTestCase):
    @unittest.skipUnless(
        torch.cuda.device_count() >= 2, "NCCL refuses two ranks on one device"
    )
    def test_a_ride_and_its_reading_cross_the_lane(self):
        from sglang.srt.afd.lane import Lane

        port = _free_port()
        child = subprocess.Popen(
            [sys.executable, "-c", _HOST_END.format(port=port, dev=1)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            lane = Lane("pool", "127.0.0.1", port, torch.device("cuda:0"))
            deadline = time.time() + 120
            while not (lane.ready or lane.failed) and time.time() < deadline:
                time.sleep(0.1)
            self.assertTrue(lane.ready, "pool lane never came up")

            ride = torch.arange(8, dtype=torch.float32, device="cuda:0").view(1, 8)
            lane.send(ride)
            reading = lane.recv((1, 8), torch.float32)
            torch.cuda.synchronize()
            torch.testing.assert_close(reading, ride + 1)
            self.assertFalse(lane.failed)
        finally:
            child.kill()
            child.wait()

    @unittest.skipUnless(
        torch.cuda.device_count() == 1, "the decline path needs the pairing to fail"
    )
    def test_a_lane_that_cannot_pair_declines_loudly(self):
        # two ranks on one device is exactly what a misprovisioned machine
        # offers; the lane must END (failed, not ready) rather than hang, and
        # the refusal must leave the process serving -- everything stays on
        # the TCP wire
        from sglang.srt.afd.lane import Lane

        port = _free_port()
        child = subprocess.Popen(
            [sys.executable, "-c", _HOST_END.format(port=port, dev=0)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            lane = Lane("pool", "127.0.0.1", port, torch.device("cuda:0"))
            deadline = time.time() + 120
            while not (lane.ready or lane.failed) and time.time() < deadline:
                time.sleep(0.1)
            self.assertTrue(lane.failed, "the lane neither came up nor declined")
            self.assertFalse(lane.ready)
        finally:
            child.kill()
            child.wait()


if __name__ == "__main__":
    unittest.main()
