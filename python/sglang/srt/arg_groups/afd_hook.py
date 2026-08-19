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
    if server_args.afd_mode not in ("null", "host", "pool"):
        raise ValueError(
            f"--afd-mode must be null, host or pool; got {server_args.afd_mode!r}"
        )

    shift = server_args.afd_q_shift_layers
    if not isinstance(shift, int) or isinstance(shift, bool):
        raise TypeError(
            f"--afd-q-shift-layers is a layer count, got {shift!r}. There is no fractional "
            f"setting: a query read between a block's two sub-layers would be read from a point "
            f"where the residual stream has no value."
        )
    if shift < 0:
        raise ValueError(f"--afd-q-shift-layers must not be negative, got {shift}")
    if shift > MAX_REASONABLE_SHIFT:
        raise ValueError(
            f"--afd-q-shift-layers={shift} is deeper than any stack this serves. The offset is "
            f"N-0.5 layers, so N is a layer count and not a half-layer count -- the study's unit "
            f"is 2N-1, and a value that looks like a half-layer count here would silently run "
            f"twice the depth it names."
        )

    if server_args.afd_mode == "host":
        if not server_args.afd_pool_addr:
            raise ValueError(
                "--afd-mode=host needs --afd-pool-addr HOST:PORT. A host with nowhere to send "
                "its feed-forward should fail here, not at the first token."
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

    if shift > 0 and server_args.afd_mode == "null":
        logger.warning(
            "--afd-q-shift-layers=%s with the two sides colocated: the query is read early and "
            "there is no pool call for it to overlap. This measures what the rewiring costs, "
            "which is a real question, but it is not what the arrangement buys.",
            shift,
        )
    if shift == 0 and server_args.afd_mode != "null":
        logger.info(
            "afd %s with --afd-q-shift-layers=0: the standard read point, so the host waits for "
            "each feed-forward before its next attention. This is the synchronous baseline.",
            server_args.afd_mode,
        )
