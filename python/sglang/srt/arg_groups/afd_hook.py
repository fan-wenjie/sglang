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

from sglang.srt.arg_groups.overrides import declare_resolution

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

    _refuse_pool_only_on_host(server_args)

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
        # Declared, not assigned. The projection that builds the config bags reads the
        # declaration stash, so a resolution write that only sets the field is invisible to it --
        # the pool would report `sleep_on_idle=False` to anything reading the bag while its own
        # scheduler was, in fact, asleep.
        declare_resolution(server_args, "handle_afd", sleep_on_idle=True)

    if server_args.afd_mode == "pool":
        _skip_the_warmup_the_pool_never_repeats(server_args)

    if server_args.afd_mode == "host" and server_args.afd_pool_addr:
        logger.info(
            "afd host: the feed-forward goes to %s, so its weights are built on the meta device "
            "and never allocated here. The loader reads this from the server args, not from a "
            "flag set here: the model loads in a scheduler process this one spawns.",
            server_args.afd_pool_addr,
        )

    if server_args.afd_mode == "host":
        if not server_args.afd_pool_addr:
            raise ValueError(
                "--afd-mode=host needs --afd-pool-addr. A host with nowhere to send anything is "
                "the ordinary stack wearing a flag, and it should fail here rather than at the "
                "first token."
            )
        host, _, port = server_args.afd_pool_addr.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(
                f"--afd-pool-addr wants HOST:PORT, got {server_args.afd_pool_addr!r}"
            )

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


# The POOL owns the arrangement's configuration and pushes it at the HELLO; a host
# configures nothing about it and loads no configuration file. A host-side value could
# only agree with the pushed one (redundant) or disagree (a second source of truth, the
# exact thing the push exists to remove), so it is refused outright rather than
# adopted when it happens to match.
_POOL_ONLY = (
    ("afd_min_batch", 2),
    ("afd_max_wait_ms", 5),
    ("afd_transfer_backend", None),
    ("afd_bootstrap_port", 8999),
)



def _skip_the_warmup_the_pool_never_repeats(server_args) -> None:
    """A pool does not warm up, because the forward it would warm up is not the one it serves.

    sglang's startup warmup runs a whole unrouted forward: every layer, its own attention, its
    own cache. A routed pool serves frames -- spans of feed-forward and projections, with the
    attention on the host -- and takes that path never. Warming the other one primes kernels the
    pool will not call and, since `afd/remote_state.py` leaves the cache on the host, it is also
    the one thing in the process that would try to write a cache that is not here.

    So it is turned off for the same reason the cache is: the pool is not that kind of server.
    Left alone if the operator named it.
    """
    if server_args.skip_server_warmup:
        return
    logger.info(
        "afd pool: skipping the startup warmup. It runs an unrouted forward -- every layer with "
        "its own attention over its own cache -- and this pool serves spans with the attention "
        "on the host, so the warmup primes a path it never takes and is the only thing here "
        "that would write a cache this process does not hold."
    )
    declare_resolution(server_args, "handle_afd", skip_server_warmup=True)


def _refuse_pool_only_on_host(server_args) -> None:
    if getattr(server_args, "afd_mode", "null") != "host":
        return
    offending = [
        f"--{name.replace('_', '-')}"
        for name, default in _POOL_ONLY
        if getattr(server_args, name, default) != default
    ]
    if offending:
        raise ValueError(
            f"{', '.join(offending)} configure the POOL, and --afd-mode=host. A host "
            f"adopts the pool's pushed configuration at startup and sets none of the "
            f"arrangement's flags; remove them from the host's command line and set "
            f"them on the pool."
        )
