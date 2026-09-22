"""A name bound only under `if SOME_CONSTEXPR > 0:` must not be read outside it.

Triton resolves such a guard at compile time, so on the geometry where the
constexpr is 0 the binding never happens and the read is a NameError -- raised
from inside `ast_to_ttir`, which runs on the first launch of that kernel and
not before. The failure therefore arrives as a server that will not start, one
model load into a run, with a traceback that names the kernel rather than the
configuration that selected it.

That is exactly how this landed. `_vk_fwd_grouped_kernel_stage1` bound
`offs_dpe`, `mask_dpe` and `qpe` under `if BLOCK_DPE > 0` and passed all three
to the row loop unconditionally. BLOCK_DPE is 0 only for rope-less MLA
(GLM-5.3-Flash); every operator test runs at Kimi geometry, where the guard is
taken, so the whole suite was green. The first GLM run on the vestigekv arm
died after loading 88 GB of weights.

This is a source check, not a compile: it needs no GPU and no geometry, which
is the point -- it covers the configurations nobody has a fixture for. It
flags a name only when the guard's body is its sole binding site: a name the
`else` branch also binds is fine, and so is one assigned elsewhere in the
function (a loop-carried accumulator), and so is a read that itself sits under
a guard on the same constexpr.
"""

import ast
import os
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

VESTIGEKV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))),
    "python", "sglang", "srt", "layers", "attention", "vestigekv",
)


def _bound(nodes):
    names = set()
    for n in nodes:
        for x in ast.walk(n):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                names.add(x.id)
    return names


def _guard_var(node):
    """The constexpr an `if` is switching on, if it looks like one."""
    if not isinstance(node, ast.If):
        return None
    test = node.test
    if (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id.isupper()
    ):
        return test.left.id
    return None


def _reads_with_guards(fn):
    """Every Name load in fn, tagged with the constexpr guards it sits under."""
    found = {}

    def walk(node, guards):
        var = _guard_var(node)
        for child in ast.iter_child_nodes(node):
            inner = guards | {var} if (var and child in node.body) else guards
            walk(child, inner)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            found[id(node)] = (node.id, node.lineno, guards)

    walk(fn, frozenset())
    return found


def leaks(fn):
    out = set()
    reads = _reads_with_guards(fn)
    for node in [n for n in ast.walk(fn) if _guard_var(n)]:
        var = _guard_var(node)
        inside = {id(x) for x in ast.walk(node)}
        only = _bound(node.body) - _bound(node.orelse)
        only -= _bound([
            s for s in ast.walk(fn)
            if isinstance(s, (ast.Assign, ast.AugAssign, ast.For))
            and id(s) not in inside
        ])
        for name, line, guards in reads.values():
            if name in only and var not in guards:
                out.add((name, line, node.lineno, var))
    return sorted(out)


def jit_functions(path):
    with open(path) as fh:
        tree = ast.parse(fh.read())
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        if any(
            getattr(d, "attr", getattr(d, "id", "")) == "jit"
            for d in fn.decorator_list
        ):
            yield fn


class TestVestigeKVConstexprGuards(CustomTestCase):
    def test_no_name_is_read_outside_the_only_guard_that_binds_it(self):
        bad = []
        for name in sorted(os.listdir(VESTIGEKV)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(VESTIGEKV, name)
            for fn in jit_functions(path):
                for leaked, use, defn, var in leaks(fn):
                    bad.append(
                        f"{name}:{use} {fn.name}: '{leaked}' is read where {var} "
                        f"may be 0, but is bound only inside the guard at line {defn}"
                    )
        self.assertEqual(bad, [], "\n" + "\n".join(bad))

    def test_the_check_catches_the_shape_that_broke_the_glm_arm(self):
        """A negative control: the rule has to fail on the original code."""
        src = (
            "@triton.jit\n"
            "def k(BLOCK_DPE: tl.constexpr):\n"
            "    base = 0\n"
            "    if BLOCK_DPE > 0:\n"
            "        offs = tl.arange(0, BLOCK_DPE)\n"
            "    row(base, offs)\n"
        )
        fn = next(iter(jit_functions_from_source(src)))
        self.assertEqual(
            [(n, g) for n, _, _, g in leaks(fn)], [("offs", "BLOCK_DPE")]
        )


def jit_functions_from_source(src):
    for fn in ast.walk(ast.parse(src)):
        if isinstance(fn, ast.FunctionDef) and any(
            getattr(d, "attr", getattr(d, "id", "")) == "jit"
            for d in fn.decorator_list
        ):
            yield fn


if __name__ == "__main__":
    unittest.main()
