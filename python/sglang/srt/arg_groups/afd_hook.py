"""Validate and normalise the AFD server args, before anything loads a model.

Every check here exists because the failure it prevents is silent or arrives late:

  * a host with no reachable pool fails at the first token, an hour into a benchmark, with a
    connection error that reads like a network blip;
  * a pool that will not depart without a partner hangs the last request of a draining workload
    rather than failing, so `--afd-max-wait-ms 0` is refused rather than accepted as "no timeout".

What is NOT here: any check belonging to a derived arm. An arm that adds a flag owns the refusal
that flag needs, and it validates it through `arms.check_args` -- named by nobody here. This file
naming a derived package by hand is how the shared half stopped being able to ship without it the
first time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

def handle_afd(server_args: ServerArgs) -> None:
    """Raise on a configuration that cannot mean what it says; warn on one that can but rarely does."""
    if server_args.afd_mode in ("host", "pool"):
        # Which of sglang's own features this arrangement can be run beside, refused by name
        # before a model loads. Everything unsupported fails the same way when unchecked: the
        # server starts, the tokens are fluent, and the measurement is of something else.
        from sglang.srt.afd.compatibility import check as check_compatibility

        check_compatibility(server_args)

    if server_args.afd_mode not in ("null", "host", "pool"):
        raise ValueError(
            f"--afd-mode must be null, host or pool; got {server_args.afd_mode!r}"
        )

    _check_pool_args(server_args)

    # Whatever arms this build carries check their own flags here. Nothing in this file knows
    # their names: an arm announces itself by being imported, and `load` imports whatever is
    # installed. With none installed this is a call into an empty registry.
    from sglang.srt.afd.arms import check_args, load

    load()
    check_args(server_args)


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
