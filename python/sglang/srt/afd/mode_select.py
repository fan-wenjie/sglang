"""Which arrangement a request should be served by, and when it changes.

Two arrangements differ only in which piece of a layer is remote:

    ffn_remote     the feed-forward is on the pool; the host sweeps its own cache
    sweep_remote   the sweep is on a cache holder; the host keeps every weight

They cross where the sweep costs what the feed-forward costs -- whichever piece is bigger is the
one that should leave -- so the rule is a CONDITION, not a context length, and turning it into a
context length needs the sweep's slope at the batch in use.

    threshold_context = feed_forward_us / slope_us_per_token

Measured on one Blackwell, bfloat16, the sweep charged to the 16 softmax layers:

    batch     ffn    slope        threshold ctx   cache there
        4    365u    4.04 us/kT          88,807      23.3 GB
       16    369u   11.51 us/kT          31,904      33.5 GB
       32    393u   20.22 us/kT          19,139      40.1 GB
       64    380u   40.95 us/kT           9,054      38.0 GB

Both numbers are hardware-specific. `python -m sglang.srt.afd.crossover` recalibrates them, and a
table carried onto another card is a table about the wrong card.

## Why a fixed batch makes this simple

A request's context only grows, so with the batch fixed the threshold stops moving and each
request crosses it exactly once. Nothing has to hysteresis against a threshold that is itself
sliding, which is what a varying batch would produce -- one finished request changes the batch,
the threshold moves, and every other request's mode changes with it.

That monotonicity is asserted rather than assumed: a request that has flipped never flips back,
and one whose context appears to SHRINK is a retraction or a bug, so it is logged and kept on the
side it reached rather than quietly reverting.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

FFN_REMOTE = "ffn_remote"
SWEEP_REMOTE = "sweep_remote"


class ModeSelector:
    """Per-request choice of arrangement, monotone in context."""

    def __init__(self, *, feed_forward_us: float, slope_us_per_token: float,
                 hysteresis: float = 1.0):
        if slope_us_per_token <= 0:
            raise ValueError(
                f"the sweep's slope must be positive, got {slope_us_per_token}. A flat sweep "
                f"means the threshold is at infinity and this selector would never flip -- which "
                f"is a calibration that was never run, not a model with a free sweep."
            )
        self.feed_forward_us = feed_forward_us
        self.slope_us_per_token = slope_us_per_token
        # >1 makes the flip late. Not needed for a fixed batch, where context alone moves and only
        # upwards; kept for a caller that varies the batch and therefore moves the threshold too.
        self.hysteresis = hysteresis
        self._lock = threading.Lock()
        self._mode: dict[int, str] = {}
        self._reached: dict[int, int] = {}
        self.flips = 0
        self.shrinks = 0

    @property
    def threshold_context(self) -> float:
        return self.feed_forward_us * self.hysteresis / self.slope_us_per_token

    def sweep_cost_us(self, context: int) -> float:
        return self.slope_us_per_token * context

    def mode_for(self, request_id: int, context: int) -> str:
        """The arrangement this request should be served by at this context."""
        with self._lock:
            reached = self._reached.get(request_id)
            if reached is not None and context < reached:
                # a retraction, or a bug. Either way the cache for this request already holds the
                # longer history, and reverting would serve it under an arrangement that does not
                # know where that history lives.
                self.shrinks += 1
                logger.warning(
                    "afd: request %s reports context %s after reaching %s. Context does not "
                    "shrink on its own; keeping it on %s rather than flipping back.",
                    request_id, context, reached, self._mode.get(request_id, FFN_REMOTE),
                )
                return self._mode.get(request_id, FFN_REMOTE)
            self._reached[request_id] = max(reached or 0, context)
            current = self._mode.get(request_id)
            if current == SWEEP_REMOTE:
                return current
            if context >= self.threshold_context:
                if current is not None:
                    self.flips += 1
                    logger.info(
                        "afd: request %s crossed %s tokens; its sweep now costs more than a "
                        "feed-forward, so the sweep is the piece that should be remote.",
                        request_id, int(self.threshold_context),
                    )
                self._mode[request_id] = SWEEP_REMOTE
                return SWEEP_REMOTE
            self._mode[request_id] = FFN_REMOTE
            return FFN_REMOTE

    def forget(self, request_id: int) -> None:
        with self._lock:
            self._mode.pop(request_id, None)
            self._reached.pop(request_id, None)

    def report(self) -> dict:
        with self._lock:
            modes = list(self._mode.values())
            return {
                "threshold_context": self.threshold_context,
                "tracked": len(modes),
                "ffn_remote": modes.count(FFN_REMOTE),
                "sweep_remote": modes.count(SWEEP_REMOTE),
                "flips": self.flips,
                "context_shrinks": self.shrinks,
            }
