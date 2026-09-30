"""Every global name the VestigeKV modules reference must actually resolve.

A free name in a VestigeKV module is not a crash. The asynchronous calibrated
build runs on a worker that catches Exception and keeps the provisional index
serving -- by design, because a failed fit must not take the server down and
the provisional index over-fetches rather than under-recalling, so the output
stays correct. So a NameError on a rarely-taken branch turns calibration off
for that request and reports nothing a serving number would show.

That has now happened twice. On 2026-09-23 it was `lid` for `job["lid"]`, which
went through five measurement runs. On 2026-09-30 it was `logger` in
recall_tier, used at three sites with only `import logging` in scope: it took
out the tier-2 skip, the archive-bound regime escalation, and the rank report,
and two ablation rounds came back looking like clean baselines because the
branch that was supposed to announce itself raised instead.

Both are the same defect class and neither needs a GPU to catch: a name is
either in the module's globals, its builtins, or it is a bug. This walks every
code object -- nested functions and comprehensions included -- and checks the
LOAD_GLOBAL names, which is why a branch nothing exercises is still covered.
Coverage is derived from the package directory rather than a hand-kept list, so
a new module is checked the day it lands.
"""

from __future__ import annotations

import builtins
import importlib
import pathlib
import pkgutil
import types
import unittest

_PKG = "sglang.srt.layers.attention.vestigekv"


def _code_objects(code: types.CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


class TestVestigeKVModuleGlobals(unittest.TestCase):
    def test_no_free_global_names(self):
        pkg = importlib.import_module(_PKG)
        mod_names = [
            f"{_PKG}.{m.name}"
            for m in pkgutil.iter_modules([str(pathlib.Path(pkg.__file__).parent)])
            if not m.ispkg
        ]
        self.assertGreater(len(mod_names), 5, "package discovery found almost nothing")

        missing = []
        for name in sorted(mod_names):
            mod = importlib.import_module(name)
            mod_globals = set(vars(mod)) | set(vars(builtins))
            for code in _code_objects(mod.__loader__.get_code(name)):
                for gname in code.co_names:
                    if gname in mod_globals:
                        continue
                    # co_names also holds attribute names (foo.bar puts "bar"
                    # here), so a miss is only evidence when the name is never
                    # an attribute anywhere in the module. Checking the union
                    # of every module's attribute names would be unsound in
                    # the other direction; instead require the name to appear
                    # as a bare global load.
                    if any(
                        instr.opname == "LOAD_GLOBAL" and instr.argval == gname
                        for instr in __import__("dis").get_instructions(code)
                    ):
                        missing.append(f"{name}: {gname}")

        self.assertEqual(
            [], sorted(set(missing)),
            "free global name(s) in VestigeKV modules; each one is a silent "
            "calibration failure, not a crash",
        )


if __name__ == "__main__":
    unittest.main()
