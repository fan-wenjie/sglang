"""Validate and normalise the AFD server args, before anything loads a model.

Every check here exists because the failure it prevents is silent or arrives late:

  * a host with no reachable pool fails at the first token, an hour into a benchmark, with a
    connection error that reads like a network blip;
  * a negative or fractional shift is a request for a read point the residual stream does not
    have, and rounding it would run a different arm than the one the flag names;
  * a shift with the two sides colocated converts the model and buys no overlap at all, which
    measures the cost of the rewiring and reports it as the cost of the arrangement;
  * a pool that will not depart without a partner hangs the last request of a draining workload
    rather than failing, so `--afd-max-wait-ms 0` is refused rather than accepted as "no timeout".
  * `--afd-coverage all` under `--afd-span-cut` asks for a conversion the group cut does not
    perform, and the group cut ignored it in silence: every run of that arrangement passed it and
    reported it, and the linear-attention queries never moved.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

MAX_REASONABLE_SHIFT = 64


def handle_afd(server_args: ServerArgs) -> None:
    """Raise on a configuration that cannot mean what it says; warn on one that can but rarely does."""
    if server_args.afd_mode in ("host", "pool"):
        # Which of sglang's own features this arrangement can be run beside, refused by name
        # before a model loads. Everything unsupported fails the same way when unchecked: the
        # server starts, the tokens are fluent, and the measurement is of something else.
        from sglang.srt.afd.compatibility import check as check_compatibility

        check_compatibility(server_args)

    _check_coverage(server_args)

    if server_args.afd_mode not in ("null", "host", "pool"):
        raise ValueError(
            f"--afd-mode must be null, host or pool; got {server_args.afd_mode!r}"
        )

    shift = server_args.afd_query_shift_layers
    if shift is None:
        # unset: the checkpoint's own read point is resolved at load, where its config is in hand
        return _check_pool_args(server_args)
    if not isinstance(shift, int) or isinstance(shift, bool):
        raise TypeError(
            f"--afd-query-shift-layers is a layer count, got {shift!r}. There is no fractional "
            f"setting: a query read between a block's two sub-layers would be read from a point "
            f"where the residual stream has no value."
        )
    if shift < 0:
        raise ValueError(f"--afd-query-shift-layers must not be negative, got {shift}")
    if shift > MAX_REASONABLE_SHIFT:
        raise ValueError(
            f"--afd-query-shift-layers={shift} is deeper than any stack this serves. The offset is "
            f"N-0.5 layers, so N is a layer count and not a half-layer count -- the study's unit "
            f"is 2N-1, and a value that looks like a half-layer count here would silently run "
            f"twice the depth it names."
        )

    _check_pool_args(server_args)

    if shift > 0 and server_args.afd_mode == "null":
        logger.warning(
            "--afd-query-shift-layers=%s with the two sides colocated: the query is read early and "
            "there is no pool call for it to overlap. This measures what the rewiring costs, "
            "which is a real question, but it is not what the arrangement buys.",
            shift,
        )
    if shift == 0 and server_args.afd_mode != "null":
        logger.info(
            "afd %s with --afd-query-shift-layers=0: the standard read point, so the host waits for "
            "each feed-forward before its next attention. This is the synchronous baseline.",
            server_args.afd_mode,
        )


def _check_coverage(server_args: ServerArgs) -> None:
    """`--afd-coverage` under the group cut: softmax is what it does, all is refused.

    The group cut moves ONE query, in `SpanRunner._finish`: the next group's softmax attention,
    projected from the residual between the last linear attention and the last feed-forward. Its
    linear layers take q, k and v from the current hidden and move nothing. That is coverage
    "softmax", exactly, and it is not coverage "all".

    The flag was read by nobody in the span path -- it appears in neither `span.py` nor
    `span_routing.py` -- so every run of this arrangement passed `--afd-coverage all`, logged it,
    and converted 16 layers' worth of read point while reporting 63. That is the same failure as
    the shift flag not being wired, which was found and fixed here a few hours earlier: a run that
    names a setting it never applied.

    Refused only when it is set EXPLICITLY. Unset resolves to "all" downstream for the per-layer
    cut, and refusing a default nobody chose would break every command line that never mentioned
    coverage.
    """
    if not getattr(server_args, "afd_span_cut", False):
        return
    coverage = server_args.afd_coverage
    if coverage is None or coverage == "softmax":
        return
    raise ValueError(
        f"--afd-coverage={coverage!r} with --afd-span-cut. The group cut moves one query per "
        f"group -- the next softmax attention's, projected from the read point inside the span -- "
        f"and its linear-attention layers take their query, key and value from the current hidden. "
        f"That is coverage 'softmax'. Accepting 'all' here would report 63 layers of converted "
        f"read point for an arrangement that converts 16, which is the cost of a shallower "
        f"conversion under a deeper one's name."
    )


def _check_pool_args(server_args) -> None:
    """The pool and departure settings, checked whether or not a shift was asked for."""
    if server_args.afd_mode == "pool" and not server_args.sleep_on_idle:
        # A pool serves feed-forward frames on a thread and never receives a generate request, so
        # its own scheduler loop has nothing to do and spins -- and it spins holding the GIL that
        # the departure thread needs. Measured on this arrangement: a 40 KB round trip took
        # 7.08 ms against a spinning scheduler and 1.18 ms against a sleeping one, and end-to-end
        # decode went from 6.4 to 21.4 tokens per second. It is set here rather than left to the
        # operator because nothing about the symptom points at it: the network looks slow.
        logger.info(
            "afd pool: enabling --sleep-on-idle. The pool's own scheduler has no work and its "
            "idle loop competes for the GIL with the thread answering feed-forward frames; "
            "leaving it spinning cost 6x on the round trip here."
        )
        server_args.sleep_on_idle = True

    if server_args.afd_mode == "host" and server_args.afd_pool_addr:
        logger.info(
            "afd host: the feed-forward goes to %s, so its weights are built on the meta device "
            "and never allocated here. The loader reads this from the server args, not from a "
            "flag set here: the model loads in a scheduler process this one spawns.",
            server_args.afd_pool_addr,
        )

    if server_args.afd_mode == "host":
        # A host needs somewhere for SOMETHING to go, and there are two somewheres. Requiring the
        # weights pool specifically made the reversed arrangement unreachable -- feed-forward
        # local, only the sweep on a cache pool -- which is the arrangement the budget table ranks
        # first at long context and which had therefore never been run.
        if not server_args.afd_pool_addr and not server_args.afd_cache_addr:
            raise ValueError(
                "--afd-mode=host needs --afd-pool-addr or --afd-cache-addr. A host with nowhere "
                "to send anything is the ordinary stack wearing a flag, and it should fail here "
                "rather than at the first token."
            )
        for flag, value in (("--afd-pool-addr", server_args.afd_pool_addr),
                            ("--afd-cache-addr", server_args.afd_cache_addr)):
            if not value:
                continue
            host, _, port = value.rpartition(":")
            if not host or not port.isdigit():
                raise ValueError(f"{flag} wants HOST:PORT, got {value!r}")

    if server_args.afd_mode == "pool" and server_args.afd_pool_addr:
        logger.warning(
            "--afd-pool-addr is set on a pool server and is ignored; the pool listens on "
            "--afd-bootstrap-port=%s",
            server_args.afd_bootstrap_port,
        )

    if server_args.afd_min_batch < 1:
        raise ValueError(
            f"--afd-min-batch must be at least 1, got {server_args.afd_min_batch}"
        )
    if server_args.afd_max_wait_ms <= 0:
        raise ValueError(
            f"--afd-max-wait-ms must be positive, got {server_args.afd_max_wait_ms}. A pool with "
            f"no timeout and a minimum batch above one hangs the last caller of a draining "
            f"workload, waiting for a partner that never arrives; it does not fail, which is "
            f"worse."
        )
