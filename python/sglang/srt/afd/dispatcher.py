"""The stop: one queue in front of the pool, and one connection to it.

Measured on the pool itself, 4-token frames: 1663 calls/s on one connection, 2224 on two, and
1197 on eight -- with every phase inflating about tenfold at eight, including a matmul on an idle
GPU. A phase that takes ten times longer while the work is unchanged is a thread waiting its turn.
So a pool reached by sixteen hosts is slower than the same pool reached by one, and the fix is not
inside the pool.

This sits on the node with the hosts and gives the pool a single caller. What it does for them:

    merges     same-layer feed-forward frames from several hosts into ONE upstream call, and
               splits the reply back by each host's own row count. The pool then reads the layer's
               weights once for all of them
    relays     everything else, one for one. An op whose departure CALLS BACK to its caller -- a
               span, a layer -- carries per-request state on the return path, and merging two of
               those would send one host's callback to another. Only rows that are independent may
               ride together, and the feed-forward's are

What it deliberately does NOT do is decide anything the pool used to decide differently: the
boarding rule is `boarding.ready_to_depart`, the same function the pool asks, because two copies
of "when does a bus leave" would drift.

## Why the merge is not the point

Co-batching was measured to be a pessimisation at every frame width: at 512 tokens a call, one
caller alone does 85641 tokens/s and two callers forced into one departure do 37837. The merge
here is worth having because the pool's per-call floor is a weight read (0.394 ms for one layer,
however few rows ride), so rows that arrive together should ride together -- but the REASON this
component exists is the connection count, not the batching.

## What it is not

Not a data-plane hop across the network. It belongs on the same node as the hosts it serves.

## What the hop costs, measured rather than assumed

From the host's own node, 4-token calls: 1.06 ms straight to the pool, 1.40 ms through a stop on
loopback. **0.34 ms**, and the docstring here first guessed "tens of microseconds" -- wrong by an
order of magnitude, because the hop is not a kernel copy but a full decode, queue, re-encode,
decode and re-encode in Python.

So it pays only when it saves more than 0.34 ms a call, and what it saves is the pool's
degradation under connection count: 2224 calls/s at two connections against 1197 at eight, which
is 0.45 ms against 0.84 ms per call at the pool. Sixteen hosts through one stop is roughly a wash
at today's cost and clearly wrong below eight. Making the stop cheaper is what would change that,
and the number to beat is 0.34 ms.
"""

from __future__ import annotations

import logging
import socket
import threading
import time

import torch

from sglang.srt.afd.boarding import ready_to_depart
from sglang.srt.afd.protocol import OP_FFN, Frame, decode, send_frame

logger = logging.getLogger(__name__)

# Ops whose rows are independent, so several callers' can ride one upstream call. The feed-forward
# is the only one: it reads the same weights whatever the caller's history and returns each row
# transformed on its own.
MERGEABLE = frozenset({OP_FFN})


class Stop:
    """One queue a layer, one connection upstream, and the boarding rule the pool uses."""

    def __init__(self, upstream, *, min_batch: int = 1, max_wait_s: float = 0.002) -> None:
        self.upstream = upstream
        self.min_batch = min_batch
        self.max_wait_s = max_wait_s
        self._cond = threading.Condition()
        self._waiting: dict[int, list] = {}
        self._first_seen: dict[int, float] = {}
        self.merged = 0          # upstream calls that carried more than one host
        self.departures = 0      # upstream feed-forward calls, of any size
        self.relayed = 0         # frames passed through one for one
        self.riders = 0          # hosts served across all departures

    def offer(self, frame: Frame, sock: socket.socket) -> None:
        """Take a frame from a host. Departs it here when that completes a bus."""
        if frame.op not in MERGEABLE:
            self._relay(frame, sock)
            return
        riding = None
        with self._cond:
            queue = self._waiting.setdefault(frame.layer, [])
            queue.append((frame, sock))
            if ready_to_depart(waiting=len(queue), min_batch=self.min_batch,
                               waited_s=0.0, max_wait_s=self.max_wait_s):
                riding = self._waiting.pop(frame.layer)
                self._first_seen.pop(frame.layer, None)
            else:
                self._first_seen.setdefault(frame.layer, time.perf_counter())
                self._cond.notify()
        if riding is not None:
            self._depart(frame.layer, riding)

    def due(self) -> int | None:
        """A layer whose queue has waited long enough. The timeout half, for the timer thread."""
        now = time.perf_counter()
        with self._cond:
            for layer, queue in self._waiting.items():
                if ready_to_depart(waiting=len(queue), min_batch=self.min_batch,
                                   waited_s=now - self._first_seen.get(layer, now),
                                   max_wait_s=self.max_wait_s):
                    return layer
        return None

    def depart_due(self) -> bool:
        layer = self.due()
        if layer is None:
            return False
        with self._cond:
            riding = self._waiting.pop(layer, [])
            self._first_seen.pop(layer, None)
        if not riding:
            return False
        self._depart(layer, riding)
        return True

    def _relay(self, frame: Frame, sock: socket.socket) -> None:
        """One frame up, one frame back, untouched. See MERGEABLE for why."""
        reply = self.upstream.round_trip(frame)
        send_frame(sock, reply)
        self.relayed += 1

    def _depart(self, layer: int, riding: list) -> None:
        """One upstream call for everybody at this stop, and each answer back to its own host."""
        counts = [f.tensor.shape[0] for f, _ in riding]
        joined = riding[0][0].tensor if len(riding) == 1 else torch.cat(
            [f.tensor for f, _ in riding], dim=0)
        out = self.upstream.feed_forward(riding[0][0].request_id, layer, joined)
        if out.shape[0] != sum(counts):
            raise RuntimeError(
                f"the pool returned {out.shape[0]} row(s) for {sum(counts)} sent at layer {layer}; "
                f"splitting that would give some host another host's rows"
            )
        offset = 0
        for (frame, sock), n in zip(riding, counts):
            send_frame(sock, Frame.one(frame.request_id, layer, out[offset : offset + n]))
            offset += n
        if len(riding) > 1:
            self.merged += 1
        self.departures += 1
        self.riders += len(riding)

    def report(self) -> dict:
        """Riders per DEPARTURE, not per merged call.

        The first version divided by the merged count, so a stop that never merged reported
        "riders_per_merged: 1536.0" from 1536 departures of one rider each -- a number that reads
        as the merge working spectacularly when it had not fired once. With min_batch at 1 every
        offer completes a bus on arrival and nothing ever merges, which is correct behaviour and
        must not look like the opposite.
        """
        return {"departures": self.departures, "merged_calls": self.merged,
                "relayed": self.relayed, "riders": self.riders,
                "riders_per_departure": self.riders / max(self.departures, 1)}


class PoolUpstream:
    """The one connection to the pool, behind the two calls the stop makes.

    Named as an object rather than passed as a client so a later version can hold SEVERAL pools
    and choose per call (#64) without the stop learning anything new: it asks for a feed-forward
    or a round trip, and where that goes is this object's business.
    """

    def __init__(self, address: str, connect_timeout_s: float = 10.0) -> None:
        from sglang.srt.afd.pool_client import PoolClient

        self.client = PoolClient(address, connect_timeout_s=connect_timeout_s)
        self.address = address

    def feed_forward(self, request_id: int, layer: int, joined: torch.Tensor) -> torch.Tensor:
        return self.client.collect(self.client.issue(request_id, layer, joined), "cpu")

    def round_trip(self, frame: Frame) -> Frame:
        handle = self.client.issue_frame(
            frame.request_id, frame.layer, frame.tensors, frame.op)
        return Frame(frame.request_id, frame.layer,
                     tuple(self.client.collect_frame(handle, "cpu")), frame.op)


def main() -> int:
    """Run a stop. It belongs on the node with the hosts it serves -- see the module docstring."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True, help="the pool, HOST:PORT")
    p.add_argument("--port", type=int, default=9100, help="where the hosts reach this stop")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--max-wait-ms", type=float, default=2.0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    stop = Stop(PoolUpstream(args.upstream), min_batch=args.min_batch,
                max_wait_s=args.max_wait_ms / 1000.0)
    logger.info("afd stop: upstream %s", args.upstream)
    threading.Thread(target=_report_forever, args=(stop,), daemon=True).start()
    serve(stop, args.host, args.port)
    return 0


def _report_forever(stop: Stop, every_s: float = 30.0) -> None:
    """What it merged and what it relayed, periodically. A stop that merged nothing looks exactly
    like a stop that was never used, and the difference decides whether a reading means anything."""
    while True:
        time.sleep(every_s)
        logger.info("afd stop: %s", stop.report())


def serve(stop: Stop, host: str, port: int, ready: threading.Event | None = None) -> None:
    """Accept hosts and feed their frames to the stop. Blocks."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(64)
    if ready is not None:
        ready.port = listener.getsockname()[1]
        ready.set()
    logger.info(
        "afd stop: listening on %s:%s, one connection upstream. The pool is slower reached by "
        "many callers than by one -- 2224 calls/s at two connections against 1197 at eight, with "
        "the same work on an idle GPU -- which is what this is for.",
        host, listener.getsockname()[1],
    )
    threading.Thread(target=_timer, args=(stop,), daemon=True).start()
    while True:
        conn, _ = listener.accept()
        threading.Thread(target=_handle, args=(stop, conn), daemon=True).start()


def _timer(stop: Stop) -> None:
    while True:
        if not stop.depart_due():
            time.sleep(max(stop.max_wait_s / 4, 0.0005))


def _handle(stop: Stop, sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            frame = decode(sock)
            if frame is None:
                return
            stop.offer(frame, sock)
    finally:
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
