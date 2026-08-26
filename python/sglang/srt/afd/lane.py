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

# One lane per SLOT, not one per process. A slot is one host's pairing: the arrangement's
# protocol on the lane is a fixed exchange ORDER with no metadata (`afd_query_shift/nccl_lane`),
# so two hosts sharing a communicator would interleave into each other's frames with nothing to
# say so. Separate pairs, separate ports, separate expectation queues -- and the pool's departure
# still batches riders from every host, because that is the FRAME wire's business and not this
# one's.
#
# Keyed by slot rather than by address because the slot is what the two ends agree on: the host
# states it, the pool hears it at the HELLO, and both derive the same port from it.
_LANES: dict = {}
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
        self._store = None  # the pool's rendezvous listener; outlives every pairing
        self._epoch = 0
        self._pool_ip, self._port = pool_ip, port
        self._init = threading.Thread(
            target=self._bring_up, args=(pool_ip, port), daemon=True
        )
        self._init.start()

    def relight(self) -> None:
        """The pool's half of a host restart: retire the dead pairing, arm the next.

        A pairing is one host's lifetime. When that host departs its communicator is
        dead on this side too, and a NEW host's rendezvous against the old epoch's
        keys would hang forever -- which read, from the outside, as "the lane never
        came up and the flight got 45% slower", with the text still correct. So the
        departure handler calls this: the old group is aborted (unparking anything
        that still waited on the dead peer), the epoch advances, and a fresh
        bring-up parks on the store until the next host arrives. Between the two,
        `ready` is False and every span stays on the TCP wire -- later, not wrong.
        """
        if self.role != "pool":
            return
        if not self.ready and self._pg is None:
            # Nothing paired here, so there is no dead pairing to retire and no stale
            # rendezvous to step past. Returning is not an optimisation: a relight of a lane
            # that never came up re-enters the bring-up, which re-binds a port this process
            # already holds, and the log fills with `EADDRINUSE` from a lane whose only problem
            # was being asked to restart something that had not started.
            return
        self.ready = False
        old = self._pg
        self._pg = None
        if old is not None:
            try:
                abort = getattr(old, "abort", None) or getattr(old, "_shutdown", None)
                if abort is not None:
                    abort()
            except Exception as e:  # noqa: BLE001 -- the group was already dead
                logger.info("afd lane: retiring the dead pairing raised %r", e)
        self._epoch += 1
        self.failed = False
        threading.Thread(
            target=self._bring_up,
            args=(self._pool_ip, self._port),
            daemon=True,
            name="afd-lane-relight",
        ).start()

    def _bring_up(self, pool_ip: str, port: int) -> None:
        try:
            import datetime

            import torch.distributed as dist

            # Several ranks on one device. A pool serving N hosts holds N communicators on
            # ONE card in ONE process -- rank 0 of each -- and the hosts sharing a card hold
            # one rank each of different ones. NCCL's own guidance is that assigning more than
            # one rank to a GPU needs this said out loud; without it the pairs come up and the
            # failure, when it comes, is a deadlock inside a collective rather than a refusal at
            # the door. `setdefault`, so an operator who has decided otherwise still wins.
            os.environ.setdefault("NCCL_MULTI_RANK_GPU_ENABLE", "1")
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
            os.environ.setdefault("NCCL_SHM_DISABLE", "1")
            rank = 0 if self.role == "pool" else 1
            # a STANDALONE group, never the default one: the server already initialised
            # torch.distributed for its own parallelism, and the lane is a pair between
            # two PROCESSES that share no world with it. The pool's listener is created
            # once and kept across pairings; each pairing lives under an EPOCH prefix,
            # because c10d's rendezvous keys are one host's lifetime and a second host
            # against the first's keys hangs forever (see `relight`).
            if self.role == "pool":
                if self._store is None:
                    self._store = dist.TCPStore(
                        pool_ip,
                        port,
                        2,
                        is_master=True,
                        timeout=datetime.timedelta(minutes=30),
                    )
                self._store.set("afd_lane_epoch", str(self._epoch))
                store = dist.PrefixStore(f"epoch{self._epoch}", self._store)
            else:
                self._store = dist.TCPStore(
                    pool_ip,
                    port,
                    2,
                    is_master=False,
                    timeout=datetime.timedelta(minutes=30),
                )
                self._epoch = int(self._store.get("afd_lane_epoch").decode())
                store = dist.PrefixStore(f"epoch{self._epoch}", self._store)
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
            logger.info(
                "afd lane: up (%s, pairing %s); the fixed order leaves the wire",
                self.role,
                self._epoch,
            )
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


def lane_port(base: int, slot: int) -> int:
    """The rendezvous port for one slot. Derived, so neither end configures a port list."""
    return base + slot


def lane_up(role: str, pool_ip: str, port: int, device, slot: int = 0) -> Lane:
    """This slot's lane, brought up once. Returns it immediately; check `.ready`.

    `slot` defaults to 0, which is the whole configuration for a single-host deployment and is
    what an older host announces by sending nothing. A pool arms a slot when a host claims it,
    so no flag names how many there will be.
    """
    with _LOCK:
        lane = _LANES.get(slot)
        if lane is None:
            lane = _LANES[slot] = Lane(role, pool_ip, port, device)
        elif lane.failed and not lane.ready:
            # A claim on a slot whose bring-up failed is a retry, and the only one there is:
            # `relight` declines to restart a pairing that never happened, so without this a
            # slot that lost its first bind stays down for the pool's whole life while the host
            # waits on a rendezvous nobody will answer. Building a fresh Lane rather than
            # re-entering the old one's thread, because the old one's store may be half-made.
            logger.info("afd lane: slot %s is being re-armed after a failed bring-up", slot)
            lane = _LANES[slot] = Lane(role, pool_ip, port, device)
        return lane


def the_lane(slot: int = 0):
    return _LANES.get(slot)


def lane_slots() -> tuple:
    """Which slots this process has armed. For the log line and for tests."""
    with _LOCK:
        return tuple(sorted(_LANES))


def relight_lane(slot: int | None = None) -> None:
    """Re-arm the pool's lane(s) for the next host. A no-op everywhere else.

    `None` relights every armed slot, which is what a departure handler with no slot in hand
    means by it. A departure that knows whose host left relights only that one, so the other
    hosts' pairings are not torn down for somebody else's restart.
    """
    for key, lane in list(_LANES.items()):
        if slot is None or key == slot:
            lane.relight()
