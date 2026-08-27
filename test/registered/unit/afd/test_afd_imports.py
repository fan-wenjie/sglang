"""Every module under `srt/afd` imports, from THIS tree.

The gap this fills cost a deployment round. A `pool_server.py` was copied between trees without the
`boarding.py` it imports; the pool died at startup with `ModuleNotFoundError` before it bound a
port, and the whole unit suite -- 359 cases -- was green throughout. Nothing in it asks whether the
server can start, only whether the pieces behave once imported.

Two properties, and the second is why this file also prints where the module came from:

    it imports            a module that cannot be imported cannot be tested either, so a missing
                          dependency is invisible to every other case rather than caught by one
    from this tree        the same import can succeed against an installed sglang while the tree
                          under edit is broken. A suite that passes by reading a different copy of
                          the code is worse than a suite that fails.

Deliberately not a mock or a fixture: the check is that the real import works in the real process,
which is exactly what the deployment does a second before it binds.
"""

import importlib
import os
import pkgutil
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# `split_attention.py` reaches `sgl_kernel` through `layers.attention.merge_state`, and
# this suite runs on `ubuntu-latest` with no CUDA at all. Without the stub the import fails
# at COLLECTION, which fails base-a-test-cpu, which is the pre-flight gate -- so every
# other stage skips and the run reports one root cause and a wall of grey. A no-op on a
# machine that has the package.
maybe_stub_sgl_kernel()

from sglang.test.test_utils import CustomTestCase


def _repo_root(start: str) -> str:
    """Walk up until the directory that holds `python/sglang`, rather than counting `..`.

    Counted hops break silently on a move: this file went from `test/registered/unit/` to
    `test/registered/unit/afd/` and three `..` then pointed at `<repo>/test/python/sglang/...`,
    which does not exist. A marker cannot be off by one.
    """
    here = start
    while True:
        if os.path.isdir(os.path.join(here, "python", "sglang")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            raise RuntimeError(f"no repository root above {start}")
        here = parent


HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.join(_repo_root(HERE), "python")


def afd_modules() -> list[str]:
    """Named from the directory rather than listed here, so a new file is covered on arrival."""
    import sglang.srt.afd as afd

    return sorted(
        f"sglang.srt.afd.{m.name}"
        for m in pkgutil.iter_modules(afd.__path__)
        if not m.ispkg
    )


class TestEveryAfdModuleImports(CustomTestCase):
    def test_each_one_imports_and_comes_from_this_tree(self):
        """One subtest a module, so a break names the module rather than the first of them."""
        names = afd_modules()
        self.assertGreater(
            len(names), 10, f"only {len(names)} modules found; the walk is wrong"
        )
        for name in names:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                where = os.path.abspath(module.__file__)
                self.assertTrue(
                    where.startswith(TREE),
                    f"{name} was imported from {where}, not from {TREE}. The suite would be "
                    f"testing a different copy of the code than the one being edited.",
                )

    def test_the_package_itself_and_the_argument_hook_import(self):
        """The two entry points a server crosses before it binds a port: the package, and the hook
        that validates its arguments. The hook runs in the launcher, earlier than anything else
        here, so a break in it fails before any log line an operator would think to read.
        """
        for name in ("sglang.srt.afd", "sglang.srt.arg_groups.afd_hook"):
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(name))


if __name__ == "__main__":
    unittest.main()
