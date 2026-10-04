# SPDX-License-Identifier: Apache-2.0
"""Every module the served path imports must resolve.

Bug this guards (2026-10-04): deleting the MLA-line backend left one import of
it inside `vestigekv.salience.vestigekv_backend_of`, which the live DSA indexer
calls. Nothing caught it -- the package compiled, both registered suites passed,
and the failure appeared only as "server did not start" with a
ModuleNotFoundError, three queue jobs later. py_compile checks syntax, not
imports, and the deleted module's last reference sat inside a function body
where an import error cannot surface until that function runs.

So: import every module in the vestigekv package, plus the registry and the
indexer that reach into it. A dangling import then fails here instead of at a
server launch.
"""

import importlib
import pkgutil
import unittest


class TestVestigeKVImportsResolve(unittest.TestCase):
    def test_every_vestigekv_module_imports(self):
        import sglang.srt.layers.attention.vestigekv as pkg

        failed = {}
        for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            try:
                importlib.import_module(mod.name)
            except Exception as e:  # noqa: BLE001 - report every one, not the first
                failed[mod.name] = f"{type(e).__name__}: {e}"
        self.assertEqual(failed, {}, f"modules that do not import: {failed}")

    def test_the_served_entry_points_import(self):
        for name in (
            "sglang.srt.layers.attention.vestigekv_dsa_backend",
            "sglang.srt.layers.attention.attention_registry",
            "sglang.srt.layers.attention.dsa.dsa_indexer_kpool",
        ):
            importlib.import_module(name)

    def test_backend_resolution_reaches_the_dsa_backend(self):
        """The function whose dangling import caused the outage: it must run,
        and it must answer None rather than raise for a foreign backend."""
        from sglang.srt.layers.attention.vestigekv.salience import (
            vestigekv_backend_of,
        )

        self.assertIsNone(vestigekv_backend_of(object()))


if __name__ == "__main__":
    unittest.main()
