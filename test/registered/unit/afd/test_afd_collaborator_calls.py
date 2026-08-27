"""What the pool server calls on its collaborators, those collaborators have.

`self.runner.release(...)` reached for a method `LinearRunner` has never had. It is inside the
branch that answers OP_RELEASE, which is only entered under `--afd-pool-linear` and only when a
request ENDS -- so the arrangement started, served, and looked correct, and died on the first
request that finished, with an AttributeError naming a class rather than the arrangement.

The self-call guard in `test_afd_pool_server_methods.py` catches the same shape WITHIN a class.
This is the shape across one: an attribute the server holds, and a method it calls on it. Both
are invisible to the unit suite, which does not enter these branches, and both surface only on a
live pair.

Only attributes whose class is known statically are checked; anything else is somebody else's to
have, and saying so is better than a check that quietly covers less than it looks like it does.
"""

import ast
import pathlib

AFD = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt" / "afd"

# The collaborator attributes the pool server holds, and the class each is. Kept explicit rather
# than inferred: an inferred map that stopped resolving would silently check nothing.
COLLABORATORS = {
    "runner": ("linear_runner.py", "LinearRunner"),
}


def _methods_of(filename: str, classname: str) -> set:
    tree = ast.parse((AFD / filename).read_text())
    cls = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == classname
    )
    names = {f.name for f in cls.body if isinstance(f, ast.FunctionDef)}
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
        ):
            names.add(node.targets[0].attr)
    return names


def test_every_method_called_on_a_collaborator_exists_on_it():
    tree = ast.parse((AFD / "pool_server.py").read_text())
    missing = []
    for node in ast.walk(tree):
        # self.<attr>.<method>(...) and nothing deeper -- a longer chain is the collaborator's
        # own business, not this file's contract with it.
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        holder = node.func.value
        if not (
            isinstance(holder, ast.Attribute)
            and isinstance(holder.value, ast.Name)
            and holder.value.id == "self"
            and holder.attr in COLLABORATORS
        ):
            continue
        filename, classname = COLLABORATORS[holder.attr]
        if node.func.attr not in _methods_of(filename, classname):
            missing.append(
                f"pool_server.py:{node.lineno} self.{holder.attr}.{node.func.attr}() "
                f"-- {classname} has no such method"
            )
    assert not missing, (
        "the pool server calls these on collaborators that do not have them:\n  "
        + "\n  ".join(missing)
        + "\nThe unit suite does not enter these branches; a live pair finds them, sometimes only "
        "when a request ends"
    )
