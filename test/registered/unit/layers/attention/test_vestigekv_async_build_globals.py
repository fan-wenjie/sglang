"""The async calibrated build must not reference a name it cannot resolve.

The side-stream worker catches every exception, logs one warning, and leaves
the provisional certificate serving with the slot marked as never to retry.
A NameError in that body is therefore not a crash but a silent quality
regression on every request: the calibrated tier never installs, the
conservative zp=8 rung fires the whole archive, and each serving number
still looks plausible. One such name (`lid` for `job["lid"]`) went unnoticed
across five measurement runs. This pins the property the traceback would
have shown: every global the worker loads exists.
"""

import builtins
import dis
import unittest

from sglang.srt.layers.attention import vestigekv_dsa_backend as backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _unresolvable_globals(fn):
    module = vars(backend)
    names = {
        ins.argval
        for ins in dis.get_instructions(fn)
        if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME")
    }
    return sorted(n for n in names if n not in module and not hasattr(builtins, n))


class TestAsyncBuildGlobals(CustomTestCase):
    def test_worker_loop_resolves_every_global(self):
        fn = backend.VestigeKVDSABackend._build_worker_loop
        self.assertEqual(_unresolvable_globals(fn), [])

    def test_install_resolves_every_global(self):
        # The main-thread half runs outside the worker's try/except, so an
        # error here is loud; it is checked for symmetry with the worker.
        fn = backend.VestigeKVDSABackend._install_finished_builds
        self.assertEqual(_unresolvable_globals(fn), [])


if __name__ == "__main__":
    unittest.main()
