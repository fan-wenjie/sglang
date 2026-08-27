"""Who gets on the next bus, when a prefill remainder wants to ride with the decode walk-ins.

A span costs what its weight read costs, and that read is the same for one rider or sixteen
(`benchmark/afd/span_cost.py`: 2046 us at four, 2249 at sixteen). So an empty seat is pure waste,
and a prefill's leftover tokens -- the ones that did not fill a bus of their own -- are free
freight if they can be got aboard.

    decode 4 alone                 2046 us    4 rows
    decode 4 + 12 leftover tokens  2249 us   16 rows      +9.9% for 4x the work

## Measured afterwards: the shared weight read is not the payoff

These rules decide who rides when several are ready at once, and that is what they are for. The
reason first given for them -- an empty seat is pure waste because the weight read is shared --
did not survive measurement: a wide frame amortises the read across its own rows, and two callers
forced into one departure at 512 tokens went from 85641 tokens/s to 37837. The gain from several
callers is pipelining, not co-batching. AFD_FINDINGS.md, 2026-08-22.

## The rule

Capacity 16, nine walk-ins waiting, a remainder of eleven:

    the remainder boards first, whole            11 seats
    walk-ins fill the rest in arrival order       5 seats, the FIRST five
    the last four are bumped to the next bus

**Bumped from the TAIL, not the head.** The head has waited longest; the tail arrived most
recently. A bumped rider is then the OLDEST waiting when the next bus loads, so it boards then.
Bumping the head would be the opposite, and would look identical in every aggregate throughput
number.

That bounds the wait at one extra departure ONLY WHILE THERE ARE SEATS LEFT. A remainder that
fills the bus by itself leaves none, and a stream of them never drains the queue at all -- the
walk-ins wait forever while every aggregate looks healthy, because the buses are full and moving.
So the ordering rule is not enough on its own and `overdue` is not optional:

    when the oldest walk-in has waited past its deadline, the remainder YIELDS and the walk-ins
    take the whole bus

which is the same argument `max_wait_s` already makes on the other side -- a strict rule with no
timeout hangs the caller it was meant to batch.

## Why a remainder is atomic and a walk-in is not

A remainder is a chunk of ONE request's prefill, and its tokens are sequentially dependent: the
state after token n is what token n+1 reads. Split across two buses, the state advances between
them with other requests' updates interleaved, and the result is a correct-looking forward pass
over a history that never existed.

A walk-in is one decode step of one request and is independent of every other walk-in, so a set of
them splits for free. That asymmetry is the recurrence's, not a policy choice.
"""

from __future__ import annotations

from typing import NamedTuple


class Seating(NamedTuple):
    """Who rides and who waits. `bumped` keeps its order, because it is the next bus's head."""

    remainder: int
    walk_ins: int
    bumped: int

    @property
    def riders(self) -> int:
        return self.remainder + self.walk_ins


def seat(
    capacity: int, waiting: int, remainder: int = 0, overdue: bool = False
) -> Seating:
    """Load one bus. Returns how many of each kind board, and how many walk-ins are bumped.

    `remainder` is a prefill's leftover after its full buses have gone, so it is smaller than the
    capacity by construction -- a remainder that filled a bus would have taken one. It is refused
    rather than truncated if it is not, because truncating it would split a chunk.
    """
    if capacity <= 0:
        raise ValueError(f"a bus with {capacity} seats carries nobody")
    if remainder < 0 or waiting < 0:
        raise ValueError(f"negative riders: {remainder} remainder, {waiting} waiting")
    if remainder > capacity:
        raise ValueError(
            f"a remainder of {remainder} does not fit a bus of {capacity}. A remainder is what is "
            f"left after the full buses have gone, so it is smaller than one by construction; "
            f"seeing a larger one means the chunking and the capacity disagree, and truncating it "
            f"here would split a chunk across two departures."
        )
    if overdue:
        # somebody has waited too long. The remainder gives up its seats entirely rather than
        # taking what is left, because taking what is left is exactly the case that starves:
        # a remainder the size of the bus leaves nothing and the queue never moves.
        boarding = min(waiting, capacity)
        return Seating(remainder=0, walk_ins=boarding, bumped=waiting - boarding)
    seats = capacity - remainder
    boarding = min(waiting, seats)
    return Seating(remainder=remainder, walk_ins=boarding, bumped=waiting - boarding)


def load(
    capacity: int, queue: list, remainder: list | None = None, overdue: bool = False
) -> tuple[list, list]:
    """The same rule applied to actual riders. Returns (aboard, still waiting), both in order.

    The remainder rides at the FRONT of the bus. Nothing downstream depends on it, but keeping the
    order stable means a reader of the riders histogram can tell a mixed departure from a pure one
    without a second field.
    """
    remainder = list(remainder or [])
    plan = seat(capacity, len(queue), len(remainder), overdue=overdue)
    riding = remainder[: plan.remainder] + queue[: plan.walk_ins]
    return riding, queue[plan.walk_ins :]
