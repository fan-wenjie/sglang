"""Every op the host's history service answers must be routed to it as an inbound op.

The two lists are written in different files and nothing but this test connects them. When
`OP_STATE_MIX` was added to `history_service` and not to `INBOUND_OPS`, the reader thread filed
the pool's MIX frame in the reply table instead of the inbox: the pool waited 30 s for a reading
that was sitting under a key nobody would claim, and the host waited for a span reply the pool
would not send until it had that reading. Both ends timed out with tracebacks about the other.

Unit tests stayed green throughout -- a handler that is never reached still passes its own test.
"""

import ast
import pathlib

SRT = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt" / "afd"


def _ops_the_service_answers() -> set:
    """Ops the host answers: branched on in the service, or claimed through `register_op`.

    Both count. An op is served either way, and a guard that knew only about the branches would
    call a registered op unanswered -- which is how it read the moment the first handler moved
    out of the shared file and into the package that owns its arithmetic.
    """
    names = set()
    tree = ast.parse((SRT / "history_service.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.comparators) != 1:
            continue
        left, right = node.left, node.comparators[0]
        looks_like_op = (
            isinstance(left, ast.Attribute)
            and left.attr == "op"
            and isinstance(node.ops[0], ast.Eq)
        )
        if looks_like_op and isinstance(right, ast.Name):
            names.add(right.id)
    for path in sorted(SRT.parent.rglob("afd*/*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "register_op"
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                names.add(node.args[0].id)
    return names


def _inbound_ops() -> set:
    """The base assignment plus whatever an installed package adds beside its handlers.

    Membership travels with the handler (`INBOUND_OPS.update` next to `register_op`), so
    this reads both the assignment in protocol.py and every update call under afd*/ --
    the same directories the served-ops scan walks."""
    names = None
    tree = ast.parse((SRT / "protocol.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "INBOUND_OPS" for t in node.targets
        ):
            continue
        names = {
            n.id
            for n in ast.walk(node.value)
            if isinstance(n, ast.Name) and n.id.startswith("OP_")
        }
    if names is None:
        raise AssertionError("protocol.py no longer assigns INBOUND_OPS")
    for path in sorted(SRT.parent.rglob("afd*/*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "INBOUND_OPS"
            ):
                names |= {
                    n.id
                    for n in ast.walk(node.args[0])
                    if isinstance(n, ast.Name) and n.id.startswith("OP_")
                }
    return names


def test_the_service_and_the_router_name_the_same_ops():
    served, routed = _ops_the_service_answers(), _inbound_ops()
    assert (
        served
    ), "found no `frame.op == OP_*` in history_service.py; this test went blind"
    missing = served - routed
    assert not missing, (
        f"{sorted(missing)} reach the host's history service but are not in INBOUND_OPS, so the "
        f"reader files them as replies and the two ends deadlock waiting on each other"
    )


def test_the_router_does_not_route_what_nothing_answers():
    stray = _inbound_ops() - _ops_the_service_answers()
    assert not stray, (
        f"{sorted(stray)} are routed inbound but the history service answers none of them; the "
        f"inbound worker would raise on the first one and shut the socket"
    )
