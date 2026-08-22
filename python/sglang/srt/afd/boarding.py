"""When a call departs: asked again at every boundary, never decided once.

A batch is re-formed where latency VARIES, and the whole of that rule lives in one question --
"is the queue full yet?" -- asked freshly at each such boundary. Measured on this model:

    feed-forward             361 us      context-free
    linear attention         200 us      context-free
    softmax attention         19 us at 1k context -> 2397 us at 128k

Only the last one moves, and it is the stage that stays on the host. So every boundary the pool
sees is preceded by a wait whose length belongs to the CALLER's context, and the callers arrive
spread out over two orders of magnitude. Whoever is at the stop rides; whoever is still in their
attention takes the next call. Riders disperse at the destination and re-form for the following
one, which is what lets a 1k request and a 128k request share a pool without the short one waiting
out the long one's context.

## Why the answer cannot be computed once and reused

A pairing fixed in advance -- two batches offset by a round trip, taking turns so that one
computes while the other waits -- is the arrangement this replaced, and it does not survive
contact with a real batch. It assumes a stable cadence, and a batch's cadence is set by its
attention, which is set by its contexts, which are NOT uniform inside one batch: a 1k request and
a 128k request in the same batch reach the boundary 2378 us apart. An offset chosen for the batch
is wrong for every request in it, and the schedule slips a little further every layer while every
aggregate still looks healthy.

So the question is asked again instead: at each boundary, is the queue full? If it is, go now. If
it is not, wait -- but not forever, because a strict minimum with no timeout hangs the last caller
of a draining workload rather than failing it, which is a worse failure than a slow one.

## Asked after every send, by whoever sent

Not on a timer. The thread that has just answered a call is the one most likely to find another
call ready: the callers it answered are, at that instant, computing their own attentions, and
whoever finished during the departure is already waiting at the next stop. A thread that goes
straight back to reading its socket leaves them to the timer, which is a fraction of `max_wait_s`
away for work that is ready NOW -- latency added to a queue that was never empty.


## What this decides, and what it does NOT buy

It decides who rides when several callers are ready at the same instant. That is all. The payoff
it was assumed to deliver -- a second caller sharing the first one's 267 MB weight read -- was
measured and is not there: a 512-token frame amortises the layer's weights across its own rows
long before two callers can share them, and at that width the pool is saturated. Forcing two such
callers into one departure took 85641 tokens/s down to 37837.

What two callers do buy is PIPELINING: one's wire transfer against the other's compute, 2.30x at
4 tokens a call, with `min_batch` at 1 and no co-batching anywhere. So a minimum batch above 1 is
not part of this arrangement -- it would need a width where the read dominates and the pool is
idle, and the measured range contains no such width. The deployment runs min_batch 1 and this
function then answers "go now" every time, which is the intended reading rather than a degenerate
one: the pool takes whoever is ready, and being stateless is what lets it.

See AFD_FINDINGS.md, 2026-08-22, for the three widths and both settings.
"""

from __future__ import annotations


def ready_to_depart(*, waiting: int, min_batch: int, waited_s: float, max_wait_s: float) -> bool:
    """Whether the call at one boundary should go now.

    `waiting` is who is at THIS stop -- one layer's queue, not the pool's total. Departures are
    per layer because a dense stack's layer weights differ: a call is same-layer or it is not a
    call, and mixing two would read the wrong weights for one of them.

    `waited_s` is how long the FIRST of them has waited, not the last. Measuring from the last
    arrival restarts the clock on every walk-in, so a steady trickle of callers keeps the queue
    below the minimum and the timeout never fires -- the head waits forever while the queue looks
    busy the whole time.
    """
    if waiting <= 0:
        return False
    if waiting >= min_batch:
        return True
    return waited_s >= max_wait_s
