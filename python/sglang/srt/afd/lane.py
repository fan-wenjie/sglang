"""The NCCL lane: a second transport beside the TCP wire, chosen by a flag.

The TCP wire's frames are parsed, staged and served by Python on both ends, and that cost
was measured to death on the derived branch: ~0.9 ms a serve, immovable by reordering, by
packing, or by pinned staging -- twenty small interpreter operations stretched under a
contended GIL. A lane is the other answer: an arrangement whose exchange ORDER is fixed
can ride a standalone NCCL pair whose receives land directly in device buffers, with no
Python on the data path. NCCL has no tags, so the order is the whole frame format; which
sequence rides the lane is each arrangement's own protocol, built on the primitives here.

NCCL is the vehicle BECAUSE the transport is its decision, not this file's: today the
communicator runs socket transport (the IB/P2P/SHM disables below are `setdefault`, so an
operator's own environment wins), and on links with RoCE or InfiniBand and GPUDirect the
same sends go RDMA, device to device, with every protocol above this file unchanged. The
fabric upgrade is a deployment's configuration, never a code change.

One lane per process, one pair per lane: the pool is rank 0 and every port derives from
the bootstrap port, so `--afd-transfer-backend nccl` on the pool is the whole
configuration -- the choice reaches the host with the rest of the pushed configuration,
and the host reuses the pool address it already has. A lane that cannot come up leaves
the arrangement on the TCP wire exactly as it was: later, not wrong.
"""

from __future__ import annotations

import logging
import os
import threading

import torch

logger = logging.getLogger(__name__)

# the lane rendezvous listens beside the pool's bootstrap port
LANE_PORT_OFFSET = 3

TRANSFER_BACKENDS = ("tcp", "nccl")

_LANE = None
_LOCK = threading.Lock()


class Lane:
    """One end of the pair. `ready` flips once, after the communicator exists."""

    def __init__(self, role: str, pool_ip: str, port: int, device):
        self.role = role  # "pool" | "host"
        self.device = device
        self.ready = False
        self.failed = False
        self._pg = None
        self._loop_fn = None
        self._init = threading.Thread(
            target=self._bring_up, args=(pool_ip, port), daemon=True
        )
        self._init.start()

    def _bring_up(self, pool_ip: str, port: int) -> None:
        try:
            import datetime

            import torch.distributed as dist

            os.environ.setdefault("NCCL_IB_DISABLE", "1")
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
            os.environ.setdefault("NCCL_SHM_DISABLE", "1")
            rank = 0 if self.role == "pool" else 1
            # a STANDALONE group, never the default one: the server already initialised
            # torch.distributed for its own parallelism, and the lane is a pair between
            # two PROCESSES that share no world with it
            store = dist.TCPStore(
                pool_ip,
                port,
                2,
                is_master=(rank == 0),
                timeout=datetime.timedelta(minutes=30),
            )
            # The pair's ops PARK by design -- the pool's serving loop sits in a receive
            # until a pass rides the lane, which can be never. c10d's watchdog treats a
            # parked op as a hung collective and ABORTS the process at the group timeout
            # (observed: the pool died of SeqNum=3 RECV at 600s while the host's first
            # pass warmed on the wire), so the timeout is effectively infinite and a
            # genuinely departed peer is handled by the arrangement's own settling, not
            # by the watchdog.
            self._pg = dist.ProcessGroupNCCL(
                store, rank, 2, datetime.timedelta(days=30)
            )
            # one tiny exchange so the communicator exists before the first real frame:
            # lazy channel setup inside a serving path would bill its several ms to one
            # token
            probe = torch.zeros(1, device=self.device)
            if rank == 0:
                self._pg.send([probe], 1, 0).wait()
                self._pg.recv([probe], 1, 0).wait()
            else:
                self._pg.recv([probe], 0, 0).wait()
                self._pg.send([probe], 0, 0).wait()
            torch.cuda.synchronize(self.device)
            if self._loop_fn is not None:
                threading.Thread(
                    target=self._guarded_loop, name="afd-lane", daemon=True
                ).start()
            self.ready = True
            logger.info("afd lane: up (%s); the fixed order leaves the wire", self.role)
        except Exception as e:  # noqa: BLE001 -- a lane that cannot come up is declined
            self.failed = True
            logger.warning("afd lane: not up (%r); everything stays on the wire", e)

    # ---------------- primitives an arrangement's protocol is built on ----------------

    @property
    def peer(self) -> int:
        return 1 if self.role == "pool" else 0

    def send(self, tensor: torch.Tensor) -> None:
        """Enqueue one tensor to the peer. Ordered against every other send."""
        self._pg.send([tensor.contiguous()], self.peer, 0)

    def recv_into(self, buf: torch.Tensor):
        """Post one receive into a device buffer. Returns the work to wait on."""
        return self._pg.recv([buf], self.peer, 0)

    def recv(self, shape, dtype) -> torch.Tensor:
        """Blocking receive into a fresh device buffer."""
        buf = torch.empty(*shape, device=self.device, dtype=dtype)
        self.recv_into(buf).wait()
        return buf

    # ---------------- the serving side's loop ----------------

    def serve_loop(self, fn) -> None:
        """Run `fn(lane)` on a dedicated thread once the lane is up.

        The arrangement's own protocol loop: it owns the receive order, and an exception
        in it takes the lane down loudly rather than desynchronising the pair.
        """
        self._loop_fn = fn
        if self.ready:
            threading.Thread(
                target=self._guarded_loop, name="afd-lane", daemon=True
            ).start()

    def _guarded_loop(self) -> None:
        try:
            self._loop_fn(self)
        except Exception as e:  # noqa: BLE001 -- recorded; the lane dies loudly
            self.failed = True
            self.ready = False
            logger.error("afd lane: the serving loop failed: %r", e)


def lane_up(role: str, pool_ip: str, port: int, device) -> Lane:
    """The process's lane, brought up once. Returns it immediately; check `.ready`."""
    global _LANE
    with _LOCK:
        if _LANE is None:
            _LANE = Lane(role, pool_ip, port, device)
        return _LANE


def the_lane():
    return _LANE
