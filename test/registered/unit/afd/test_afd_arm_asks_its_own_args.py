"""A hook handed `server_args` decides from those, not from the process's globals.

The arm has no flag of its own any more: whether it is wanted is read off the checkpoint, or off
the setting being given at all. Two forms exist because two callers do -- the argument check runs
before anything global is set, and everything after install has only the global.

A hook that takes `server_args` and then consults the global is answering a different question
than the one it was asked. It raised "Global server args is not set yet!" from inside a loader
hook the first time, which names neither the arm nor the hook; on a real server it would be worse,
because the global IS set there and the wrong answer would be silent.
"""

import ast
import pathlib

INSTALLER = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "afd"
    / "installer.py"
)


def test_every_hook_that_takes_server_args_asks_of_them():
    tree = ast.parse(INSTALLER.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if "server_args" not in [a.arg for a in node.args.args]:
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "span_cut_wanted"
            ):
                offenders.append(node.name)
    assert not offenders, (
        f"{sorted(set(offenders))} take server_args and then ask the process's globals whether "
        f"the arm is wanted. Use span_cut_wanted_for(server_args) -- the two exist so that a "
        f"caller holding args and a caller holding none cannot get different answers"
    )
