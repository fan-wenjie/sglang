"""Every `self.<name>(...)` the pool server calls on itself has to exist.

This file exists because deleting code by line range took two neighbours with it. Removing the
cache pool meant cutting `_answer_cache` and `_sweep_or_park`; `_write_riders` and
`_expect_tensors` sat between the second of those and the next method, and went too. Nothing
noticed:

    the module still imported          a missing METHOD is not a missing name at import time
    229 unit cases stayed green        none of them drives an OP_KVPROJ or OP_SWEEP frame
    the two-machine deployment ran     it routes feed-forwards, and the callers are on the
                                       attention path, which that run never took

An `AttributeError` on the first `--afd-pool-attention` frame is what was left, and it would have
arrived as a closed socket rather than as a name.

So this walks the AST rather than exercising the paths: it needs no GPU, no sockets and no
frames, and it covers every method at once instead of the ones somebody remembered to drive.
"""

import ast
import inspect
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd import pool_server
from sglang.test.test_utils import CustomTestCase


def _self_calls(cls) -> set:
    """Names called as `self.<name>(...)` anywhere in the class body."""
    tree = ast.parse(inspect.getsource(cls))
    found = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            found.add(node.func.attr)
    return found


def _self_assignments(cls) -> set:
    """Names bound as `self.<name> = ...` anywhere in the class body.

    A callable handed in at construction -- `self.forward = forward` -- is not a method and
    `hasattr(cls, name)` is False for it, so without this the check reports a collaborator as a
    deletion. Found the first time this file ran, which is the reason it is written down: the
    guard caught its own overreach before it could be trusted.
    """
    tree = ast.parse(inspect.getsource(cls))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    found.add(target.attr)
    return found


# Every class on the AFD call path whose methods are reached only from a live pair. A missing one
# is invisible to the unit suite -- the method that calls it is never entered here -- and arrives
# on a two-machine run as a closed socket or a 30 s watchdog, describing the other end.
UNDER_GUARD = ()


def _classes_under_guard():
    """Resolved late so importing a span module is not a hard dependency of this file."""
    from sglang.srt.afd import span

    return (pool_server.Departure, span.SpanRunner)


class TestEveryModuleFunctionCalledIsDefined(CustomTestCase):
    """A module-level helper called and never defined, in the files the arm's hot path uses.

    The class check below catches a method that lost its definition. It cannot see a plain
    function: `_early_parts` was called from a span and defined nowhere, which raised NameError
    on the first token of a two-machine run and nothing here could see it -- these files are
    imported by the unit suite and their hot paths are not entered.

    Only names the module itself defines-or-should are considered: anything imported, built in,
    or reached through an attribute is somebody else's to have.
    """

    FILES = (
        "afd/span.py",
        "afd/span_routing.py",
        "afd_query_shift/early_contraction.py",
        "afd/history_service.py",
        "afd/pool_linear.py",
    )

    def test_no_module_function_is_called_without_being_defined(self):
        import ast
        import builtins
        import pathlib

        srt = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"
        for name in self.FILES:
            if not (srt / name).exists():
                # the derived package's files exist only on the derived branch; the same
                # test serves both series, so absence is not a finding here
                continue
            with self.subTest(file=name):
                tree = ast.parse((srt / name).read_text())
                defined = {
                    n.name
                    for n in ast.walk(tree)
                    if isinstance(
                        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                }
                bound = set(dir(builtins))
                for n in ast.walk(tree):
                    if isinstance(n, (ast.Import, ast.ImportFrom)):
                        bound |= {a.asname or a.name.split(".")[0] for a in n.names}
                    elif isinstance(n, ast.Assign):
                        bound |= {t.id for t in n.targets if isinstance(t, ast.Name)}
                    elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        bound |= {a.arg for a in n.args.args}
                        bound |= {a.arg for a in n.args.kwonlyargs}
                    elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        bound.add(n.id)
                called = {
                    c.func.id
                    for c in ast.walk(tree)
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                }
                missing = sorted(called - defined - bound)
                self.assertEqual(
                    missing,
                    [],
                    f"{name} calls these and neither defines nor imports them; the failure is a "
                    f"NameError on the first frame of a live run",
                )


class TestTheDepartureCallsOnlyMethodsItHas(CustomTestCase):
    def test_every_self_call_resolves(self):
        """The whole class at once, so a future deletion cannot take a quiet neighbour."""
        for cls in _classes_under_guard():
            with self.subTest(cls=cls.__name__):
                assigned = _self_assignments(cls)
                missing = sorted(
                    name
                    for name in _self_calls(cls)
                    if not hasattr(cls, name) and name not in assigned
                )
                self.assertEqual(
                    missing,
                    [],
                    f"{cls.__name__} calls these on itself and does not define them -- a "
                    f"deletion took a neighbour, or a port took the callers and left the "
                    f"callee, and the failure arrives as a closed socket on the first frame "
                    f"that reaches the caller",
                )

    def test_the_two_that_were_lost_are_named(self):
        """Named as well as covered: the general check above is the guard, and this says which
        two it was written for, so a reader of a future failure has the precedent."""
        for name in ("_expect_tensors", "_write_riders"):
            self.assertTrue(
                hasattr(pool_server.Departure, name),
                f"Departure.{name} is called by the attention path and must exist",
            )


if __name__ == "__main__":
    unittest.main()
