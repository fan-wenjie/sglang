"""The pool: feed-forward weights, no request state, one departure at a time.

Two properties the arrangement rests on, and both are the reason this is a separate process
rather than a method call:

    stateless             every call is a complete transaction. A caller that stalls stops
                          calling and blocks nobody, and the pool need not be reserved for a
                          request between that request's calls -- there is nothing of it to
                          protect. That is what lets a long-context request give the pool up
                          while it sweeps and take it back afterwards.

    context-free latency  the pool reads the same weights whatever the caller's context length.
                          A request sweeping a million positions and one sweeping a thousand cost
                          it exactly the same, which is why they can share it and why the pool
                          must NOT inherit the attention side's batching.

Departures follow the rule the study's worked example uses: wait for `min_batch` callers, then
carry EVERY caller at the stop rather than the first two. `max_wait_s` is not optional decoration
-- with a strict minimum and no timeout the last caller of a draining workload waits for a partner
that never arrives, and the request hangs rather than fails.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable

import torch
from sglang.srt.afd.protocol import Frame, decode, encode, send_frame

logger = logging.getLogger(__name__)


class Departure(threading.Thread):
    """Collects frames and hands whole batches to the feed-forward."""

    def __init__(
        self,
        forward: Callable[[torch.Tensor, int], torch.Tensor],
        min_batch: int,
        max_wait_s: float,
        device: torch.device | str,
    ):
        super().__init__(name="afd-pool-departure", daemon=True)
        self.forward = forward
        self.min_batch = min_batch
        self.max_wait_s = max_wait_s
        self.device = device
        self._cond = threading.Condition()
        # per layer, because a dense stack's layer weights differ: a departure is same-layer or it
        # is not a departure. Callers at different layers wait in different queues.
        self._waiting: dict[int, list[tuple[Frame, socket.socket]]] = {}
        # when a layer's queue first became non-empty. Held here rather than in run()'s frame
        # because `offer` can now take the departure itself, and a timer that still believed the
        # popped queue was waiting would depart the NEXT caller instantly.
        self._first_seen: dict[int, float] = {}
        self._stop = False
        self.departures: list[dict] = []

    def offer(self, frame: Frame, sock: socket.socket) -> None:
        """Queue a frame, and depart it here if that completes a batch.

        The connection thread used to hand every frame to the departure thread and go back to the
        socket, which costs a condition-variable wake and a scheduler round trip per call -- on a
        single-caller pool, where the batch is complete the moment it arrives, that handoff buys
        nothing. When the queue is already full the offering thread takes the departure itself.
        The queue is popped under the lock, so exactly one thread ever carries a given set of
        riders, and the timer thread still owns the partial-batch case.
        """
        riding = None
        with self._cond:
            queue = self._waiting.setdefault(frame.layer, [])
            queue.append((frame, sock))
            if len(queue) >= self.min_batch:
                riding = self._waiting.pop(frame.layer)
                self._first_seen.pop(frame.layer, None)
            else:
                self._first_seen.setdefault(frame.layer, time.perf_counter())
                self._cond.notify()
        if riding is not None:
            self._depart(frame.layer, riding)

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def _ready_layer(self, now: float) -> int | None:
        for layer, queue in self._waiting.items():
            if not queue:
                continue
            if len(queue) >= self.min_batch:
                return layer
            if now - self._first_seen.get(layer, now) >= self.max_wait_s:
                # the timeout the strict rule needs. Without it the last caller of a draining
                # workload waits for a partner that never comes.
                return layer
        return None

    def run(self) -> None:
        """The timeout half. The full-batch half is taken by whichever thread offered the frame."""
        while True:
            with self._cond:
                while True:
                    if self._stop and not any(self._waiting.values()):
                        return
                    now = time.perf_counter()
                    for layer, queue in self._waiting.items():
                        if queue and layer not in self._first_seen:
                            self._first_seen[layer] = now
                        if not queue:
                            self._first_seen.pop(layer, None)
                    layer = self._ready_layer(now)
                    if layer is not None:
                        break
                    self._cond.wait(timeout=self.max_wait_s / 4 or 0.01)
                riding = self._waiting.pop(layer)
                self._first_seen.pop(layer, None)
            self._depart(layer, riding)

    def _depart(self, layer: int, riding: list[tuple[Frame, socket.socket]]) -> None:
        started = time.perf_counter()
        widths = {f.tensor.shape[1] for f, _ in riding}
        if len(widths) != 1:
            raise RuntimeError(f"layer {layer} departure mixes hidden widths {widths}")
        counts = [f.tensor.shape[0] for f, _ in riding]
        # one rider is the common case on a single-caller pool, and torch.cat on a one-element
        # list still copies the whole frame
        joined = riding[0][0].tensor if len(riding) == 1 else torch.cat(
            [f.tensor for f, _ in riding], dim=0
        )
        batch = joined.to(self.device)
        out = self.forward(batch, layer)
        if out.shape != batch.shape:
            raise RuntimeError(
                f"layer {layer} feed-forward returned {tuple(out.shape)} for {tuple(batch.shape)}; "
                f"the pool returns the same tokens it was given"
            )
        offset = 0
        for (frame, sock), n in zip(riding, counts):
            piece = out[offset : offset + n]
            offset += n
            try:
                send_frame(sock, Frame(frame.request_id, layer, piece))
            except OSError:
                logger.warning(
                    "caller for request %s layer %s went away before its reply",
                    frame.request_id,
                    layer,
                )
        self.departures.append(
            {
                "layer": layer,
                "riders": len(riding),
                "tokens": int(batch.shape[0]),
                "started": started,
                "seconds": time.perf_counter() - started,
            }
        )


def serve(
    forward: Callable[[torch.Tensor, int], torch.Tensor],
    host: str,
    port: int,
    min_batch: int,
    max_wait_s: float,
    device: torch.device | str,
    ready: threading.Event | None = None,
) -> Departure:
    """Run a pool until the process is killed. Returns the departure thread for inspection."""
    departure = Departure(forward, min_batch, max_wait_s, device)
    departure.start()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(64)
    bound = listener.getsockname()
    logger.info("afd pool listening on %s:%s, min_batch=%s", bound[0], bound[1], min_batch)
    if ready is not None:
        ready.port = bound[1]
        ready.set()

    def handle(sock: socket.socket) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                frame = decode(sock)
                if frame is None:
                    return
                departure.offer(frame, sock)
        finally:
            sock.close()

    while True:
        conn, _ = listener.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()
