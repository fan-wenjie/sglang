"""Two passes over the AFD sources, both looking for a name that is not there.

    module scope   a name used nowhere defined in the file. Catches a missing import, which is
                   how `unpack_positions` once made every span raise inside a departure thread,
                   swallowed into a log, and cost 134 s per suite run to report as "op 9 was
                   never answered"
    function scope a name used in one function and bound only in ANOTHER. This is the stronger
                   pass and the module one cannot see it: the name IS in the file, just not
                   anywhere this function can reach. It is what a careless extraction produces --
                   `_finish_linear` was split out of `_linear_attention` carrying a trace block
                   that referred to `request_ids`, `alpha`, `q_tilde` and `reading`, four names
                   that only raise on a first multi-row prefill

The second pass has to walk scopes properly or it is useless noise: a closure legitimately reads
its enclosing function's locals, and `pool_server._install_history_calls` defines three of them
over `riding` and `counts`. So each function is checked against its own bindings UNIONED with
every enclosing scope's, and a nested function's bindings do not leak outward.

Names bound by `global`/`nonlocal` are taken on trust; the point is unreachable names, not
shadowing.
"""

import ast
import builtins
import pathlib
import sys

BUILTINS = set(dir(builtins))


def _own_bindings(node) -> set:
    """What this scope binds, NOT descending into nested function or class bodies."""
    bound = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = node.args
        bound |= {x.arg for x in a.args + a.posonlyargs + a.kwonlyargs}
        bound |= {x.arg for x in (a.vararg, a.kwarg) if x}
    body = [node] if isinstance(node, ast.Lambda) else node.body
    stack = list(body if isinstance(body, list) else [body])
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)  # the name is bound here; the body is its own scope
            continue
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, ast.ImportFrom):
            bound |= {x.asname or x.name for x in n.names}
        elif isinstance(n, ast.Import):
            bound |= {(x.asname or x.name).split(".")[0] for x in n.names}
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound |= set(n.names)
        stack.extend(ast.iter_child_nodes(n))
    return bound


def _own_loads(node) -> list:
    """Names this scope READS directly, not descending into nested scopes."""
    out = []
    body = [node] if isinstance(node, ast.Lambda) else node.body
    stack = list(body if isinstance(body, list) else [body])
    # default values and decorators are evaluated in the ENCLOSING scope, so they are not here
    while stack:
        n = stack.pop()
        if isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.append(n)
        stack.extend(ast.iter_child_nodes(n))
    return out


def _walk_scopes(node, enclosing: set, path: str, report: list) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            visible = enclosing | _own_bindings(child)
            name = getattr(child, "name", "<lambda>")
            for used in _own_loads(child):
                if used.id not in visible:
                    report.append(
                        f"{path}:{used.lineno}: {name}() reads {used.id!r}, "
                        f"bound in no scope it can reach"
                    )
            _walk_scopes(child, visible, path, report)
        elif isinstance(child, ast.ClassDef):
            # a class body's names are NOT visible to its methods, which is the Python rule and
            # also the one people get wrong; methods see the module and any enclosing function
            _walk_scopes(child, enclosing, path, report)
        else:
            _walk_scopes(child, enclosing, path, report)


def check(path: str) -> list:
    tree = ast.parse(pathlib.Path(path).read_text())
    module = _own_bindings(ast.Module(body=tree.body, type_ignores=[]))
    report = []
    for used in _own_loads(ast.Module(body=tree.body, type_ignores=[])):
        if used.id not in module | BUILTINS:
            report.append(
                f"{path}:{used.lineno}: module scope reads {used.id!r}, defined nowhere"
            )
    _walk_scopes(tree, module | BUILTINS, path, report)
    return report


if __name__ == "__main__":
    findings = [line for f in sys.argv[1:] for line in check(f)]
    for line in findings:
        print(line)
    print(f"{len(sys.argv) - 1} file(s), {len(findings)} unreachable name(s)")
    sys.exit(1 if findings else 0)
