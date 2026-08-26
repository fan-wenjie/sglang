"""The query-shift triangle on the lane: this arrangement's fixed order.

The lane itself -- the standalone NCCL pair, the rendezvous, the primitives -- is the
shared half's (`afd.lane`); what lives here is the span cut's own protocol on it. Per
cooked layer, in span order: the pool sends q~, the host answers the reading, the pool
sends the packed advance, and nothing else ever rides. No metadata crosses at all -- the
HOST already knows which (layer, row) comes next, because it issued the span and a middle
span at rung 2 cooks every linear layer it holds, in span order: `expect` is called at
issue time, and a span that rides the lane says so ON THE FRAME (OP_SPAN_LANE), which is
what makes readiness race-free. The host decides once, at issue, and the pool obeys the
op or refuses it by name.

Single host, decode rows only, one row a rider; everything else stays on the TCP wire.
"""

from __future__ import annotations

import collections
import threading

import torch

from sglang.srt.afd.lane import the_lane  # noqa: F401 -- the callers' one entry point


class Triangle:
    """The host's half: the expectation queue and the serving cycle."""

    def __init__(self, lane, service):
        self.lane = lane
        self.service = service
        self.expected: collections.deque = collections.deque()
        self._cv = threading.Condition()
        lane.serve_loop(self._loop)

    def expect(self, triples) -> None:
        """The routing announces, at ISSUE time, what the next lane triples are for."""
        with self._cv:
            self.expected.extend(triples)
            self._cv.notify()

    def _loop(self, lane) -> None:
        """recv q~ -> contract -> send reading -> recv slab -> land. Forever, in order."""
        from sglang.srt.afd_query_shift.lane_serve import land_advance, lane_contract

        while True:
            with self._cv:
                while not self.expected:
                    self._cv.wait()
                layer_id, row, q_width, slab_width = self.expected.popleft()
            q_buf = lane.recv((1, q_width), torch.float32)
            reading = lane_contract(self.service, layer_id, row, q_buf)
            lane.send(reading)
            slab = lane.recv((1, slab_width), torch.float32)
            land_advance(self.service, layer_id, row, slab)


_TRIANGLE: list = [None]


def serve_triangle(lane, service) -> None:
    """Install the host's triangle loop on the lane. Once."""
    if _TRIANGLE[0] is None:
        _TRIANGLE[0] = Triangle(lane, service)


def the_triangle():
    return _TRIANGLE[0]


# ---------------- pool side ----------------


def send_early(lane, q_tilde: torch.Tensor, reading_width: int):
    """Send the coefficient, pre-post the reading's receive. Returns the handle."""
    lane.send(q_tilde.float())
    buf = torch.empty(1, reading_width, device=lane.device, dtype=torch.bfloat16)
    return (lane.recv_into(buf), buf)


def collect_reading(handle) -> torch.Tensor:
    req, buf = handle
    req.wait()
    return buf


def send_apply(lane, slab: torch.Tensor) -> None:
    lane.send(slab)
