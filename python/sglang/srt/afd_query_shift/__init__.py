"""The early read (query shift), layered above the span cut it once carried.

The span cut moved down into `sglang.srt.afd`: it is the family's standard arrangement,
serial at shift 0, and this package is the one difference shift 1 adds -- the query read
one feed-forward early, cooked on the pool, its reading pushed back inside the window,
the read triangle riding the lane. `afd` reaches into this package only inside gates a
resolved shift of 0 never takes, so with this directory deleted shift 0 serves untouched
and a request for shift 1 is refused by name.
"""

from sglang.srt.afd_query_shift import (  # noqa: F401,E402  -- OP_STATE_EARLY
    early_contraction,
)
