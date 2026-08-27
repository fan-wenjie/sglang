"""Hold the serving core at full clock, where the operator cannot.

The pool's serving path is a GIL-serial chain of small interpreter operations, so its
wall time scales with the CPU frequency of whichever core the serving thread lands on.
A frequency governor that follows per-core utilization (schedutil) sees a thread that
blocks between spans -- roughly 60% duty -- and answers with a mid or minimum clock:
measured live, the same boot serves a 5.5 ms span wall with the core at 4.1 GHz and a
7.4 ms one at 1.5-2.5 GHz, and which one a deployment gets is decided by scheduling
accidents. The governor is the right place to fix that, and a deployment that can set
it (performance mode, or a uclamp floor) should and needs nothing from here.

This file is for the deployments that cannot -- containers routinely mount sysfs
read-only and refuse RT and uclamp -- and it is ON BY DEFAULT
(`SGLANG_AFD_ENABLE_CLOCK_HOLD=0` turns it off): the cost is one near-idle-priority
core per serving thread on a box whose job is serving, the alternative is a per-boot
lottery, and an operator who has already pinned the governor loses nothing but the
companion by leaving it on.
The mechanism is the one that measured fastest, and the three shapes that lost are
named so nobody rebuilds them: spinning on the socket from Python lost (+10% wall; the
spin fights the serving thread for the GIL), an unpinned companion lost (the balancer
migrates the serving thread away from the core the companion keeps busy), and an
IN-PROCESS companion thread lost catastrophically (its GIL touch per burn cycle turned
one prefill into ten minutes and the host gave up -- any companion sharing the
interpreter thrashes the very thread it exists to speed up). What wins is a pinned
serving thread plus a companion PROCESS on the same core: its own interpreter, no
shared GIL, the lowest scheduling priority, and PDEATHSIG so it cannot outlive the
server. The governor sees a saturated core and answers with full clock; the
companion's nice 19 costs the serving thread ~2% of the core's cycles. Alternated A/B
on fresh boots: 21.8-22.2 s a flight held against 24.6-28.9 s unheld -- 17% off the
mean, and the held band is tighter (+-0.2 s) than any unheld boot ever measured,
because the lottery is what it removes.

Cores are taken from the tail of the allowed set: core 0 collects the kernel's own
interrupt work on most boxes, and the tail is also where an operator's `taskset` for
the server naturally leaves its quietest members.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_HELD: dict = {}  # native thread id -> core, so a thread is pinned once
_COMPANIONS: dict = {}  # core -> Popen, one per held core

# Its own process, so its own GIL. PR_SET_PDEATHSIG(SIGTERM) ties its life to the
# server's; the pure-Python spin is the point -- the child has nobody to share an
# interpreter with, so burning it is free for everyone else.
_COMPANION_CODE = """
import ctypes, os
ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 15)
os.setpriority(os.PRIO_PROCESS, 0, 19)
os.sched_setaffinity(0, {core})
while True:
    pass
"""


def hold_this_thread() -> None:
    """Pin the calling thread to one core and keep that core at full clock.

    Called at the top of a serving loop. Off unless the operator asked; a refusal
    from the OS (affinity masks are policy) is logged once and served without.
    """
    if not envs.SGLANG_AFD_ENABLE_CLOCK_HOLD.get():
        return
    tid = threading.get_native_id()
    with _LOCK:
        if tid in _HELD:
            return
        try:
            allowed = sorted(os.sched_getaffinity(0), reverse=True)
            taken = set(_HELD.values())
            core = next((c for c in allowed if c not in taken), allowed[0])
            os.sched_setaffinity(0, {core})
            _HELD[tid] = core
            if core not in _COMPANIONS:
                _COMPANIONS[core] = subprocess.Popen(
                    [sys.executable, "-c", _COMPANION_CODE.format(core={core})],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except OSError as e:
            logger.warning(
                "afd clock hold: declined by the OS (%r); serving unpinned", e
            )
            _HELD[tid] = -1
            return
    logger.info(
        "afd clock hold: serving thread %s pinned to core %s, companion keeps it awake",
        tid,
        core,
    )
