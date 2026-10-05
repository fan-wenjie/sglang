"""__init__ must not call a method that reads an attribute __init__ sets later.

The failure this guards actually happened: a logging line added near the top of
VestigeKVDSABackend.__init__ called self._kept_m_fixed(), which reads
self._text_cfg_index_kpool -- assigned ~50 lines further down. Every unit test
passed, because the backend's own test builds instances with __new__ to skip the
heavy constructor, so the only thing that exercised the path was a server start,
and five queued jobs failed with "server did not start" before it was found.

A static check is the right shape here: the bug is an ORDERING property of one
function, it is invisible to any test that skips __init__, and constructing the
backend for real needs a model runner. Same idea as
test/registered/unit/test_pre_publish_readers.py.
"""

import ast
import pathlib
import unittest

SRC = (
    pathlib.Path(__file__).resolve().parents[5]
    / "python/sglang/srt/layers/attention/vestigekv_dsa_backend.py"
)
CLASS = "VestigeKVDSABackend"


def _self_attr_reads(node):
    """`self.X` loads inside `node` (reads only, not assignment targets)."""
    out = set()
    for n in ast.walk(node):
        if (
            isinstance(n, ast.Attribute)
            and isinstance(n.ctx, ast.Load)
            and isinstance(n.value, ast.Name)
            and n.value.id == "self"
        ):
            out.add(n.attr)
    return out


class TestInitOrdering(unittest.TestCase):
    def test_init_calls_no_method_that_reads_a_later_attribute(self):
        tree = ast.parse(SRC.read_text())
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS
        )
        methods = {
            n.name: n
            for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        init = methods["__init__"]

        # line of the first assignment to each self.X in __init__
        assigned_at = {}
        for n in ast.walk(init):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    for a in ast.walk(t):
                        if (
                            isinstance(a, ast.Attribute)
                            and isinstance(a.value, ast.Name)
                            and a.value.id == "self"
                        ):
                            assigned_at.setdefault(a.attr, n.lineno)

        bad = []
        for n in ast.walk(init):
            if not (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "self"
            ):
                continue
            callee = methods.get(n.func.attr)
            if callee is None:
                continue
            for attr in _self_attr_reads(callee):
                at = assigned_at.get(attr)
                if at is not None and at > n.lineno:
                    bad.append(
                        f"__init__:{n.lineno} calls self.{n.func.attr}() which reads "
                        f"self.{attr}, assigned at __init__:{at}"
                    )
        self.assertEqual(bad, [], "\n" + "\n".join(bad))


if __name__ == "__main__":
    unittest.main(verbosity=2)
