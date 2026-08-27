"""The clock hold is opt-in, pins once, and its companion shares no interpreter.

The mechanism itself (does the governor answer a saturated core with full clock) is a
property of the box and was measured live; what a unit can pin is the contract around
it: on by default, off on request with not so much as an affinity read, one core per
serving thread
with one companion each, a repeat call from the same thread a no-op, and the companion
a separate PROCESS tied to the server by PDEATHSIG -- the in-process companion thread
it replaced wedged a live prefill by thrashing the GIL, so the source is held to the
shape that cannot.
"""

import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase

_CHILD = r"""
import json, os, threading
from sglang.srt.afd import serve_clock

before = sorted(os.sched_getaffinity(0))
serve_clock.hold_this_thread()
serve_clock.hold_this_thread()  # same thread again: a no-op, not a second core
after = sorted(os.sched_getaffinity(0))
alive = [p for p in serve_clock._COMPANIONS.values() if p.poll() is None]
print(json.dumps({"before": before, "after": after, "companions": len(alive),
                  "held": len(serve_clock._HELD)}))
for p in serve_clock._COMPANIONS.values():
    p.kill()
"""


def _run_child():
    out = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    import json

    return json.loads(out.stdout.strip().splitlines()[-1])


class TestTheClockHoldContract(CustomTestCase):
    def test_explicitly_off_touches_nothing(self):
        with envs.SGLANG_AFD_ENABLE_CLOCK_HOLD.override(False):
            got = _run_child()
        self.assertEqual(got["before"], got["after"])
        self.assertEqual(got["companions"], 0)
        self.assertEqual(got["held"], 0)

    def test_the_default_is_on(self):
        self.assertIs(envs.SGLANG_AFD_ENABLE_CLOCK_HOLD.get(), True)

    def test_on_pins_to_one_core_with_one_companion(self):
        if len(os.sched_getaffinity(0)) < 2:
            self.skipTest("one allowed core: pinning would be a tautology")
        with envs.SGLANG_AFD_ENABLE_CLOCK_HOLD.override(True):
            got = _run_child()
        self.assertEqual(len(got["after"]), 1)
        self.assertEqual(got["companions"], 1)
        self.assertEqual(got["held"], 1)

    def test_the_companion_shares_no_interpreter(self):
        import inspect

        from sglang.srt.afd import serve_clock

        src = inspect.getsource(serve_clock)
        # a PROCESS, tied to the server's life -- an in-process thread here wedged a
        # live prefill (every burn cycle touches the GIL), so the source is pinned to
        # the shape that cannot
        self.assertIn("subprocess.Popen", src)
        self.assertIn("prctl(1, 15)", src)
        self.assertNotIn("threading.Thread(target=_spin", src)


if __name__ == "__main__":
    unittest.main()
