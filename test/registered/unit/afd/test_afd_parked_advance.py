"""A parked recurrent advance must be applied before the next read, and by whom.

`_mix` no longer advances the state inline: it parks the advance and answers, and the inbound
worker applies it after the reply is on the wire. That is worth 0.415 ms of a 2.731 ms callback,
MEASURED -- and it is only correct while something actually drains. A drain that stopped being
called would not raise, would not change a shape, and would return a `core` contracted against a
state one token stale: fluent output that is wrong from that token on.

So the service refuses a second MIX while an advance is parked, and these pin both halves -- that
the refusal exists, and that the client's inbound path is what clears it.
"""

import ast
import pathlib

SRT = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt" / "afd"


def _method(path: str, name: str):
    tree = ast.parse((SRT / path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path} no longer defines {name}")


def test_mix_refuses_to_read_over_a_parked_advance():
    body = ast.dump(_method("history_service.py", "_mix"))
    assert "_parked" in body, "_mix no longer parks its advance"
    assert "Raise" in body, (
        "_mix parks an advance and no longer refuses when one is already parked. Without that "
        "refusal a missing drain is silent: the read contracts against a stale state"
    )


def test_the_service_still_has_a_drain():
    node = _method("history_service.py", "drain")
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "update_only"
        for n in ast.walk(node)
    ), "drain no longer applies the advance"


def test_the_inbound_path_drains_after_replying():
    """After, not before -- draining first would put the 0.415 ms back on the caller's wait."""
    node = _method("pool_client.py", "_answer_inbound")
    lines = [
        n.lineno
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "send_frame"
    ]
    drains = [
        n.lineno
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "drain"
    ]
    assert lines, "_answer_inbound no longer sends a reply"
    assert drains, (
        "_answer_inbound no longer drains. The advance `_mix` parks would never be applied, and "
        "the next MIX for that layer is refused rather than answered"
    )
    assert min(drains) > max(lines), (
        "the drain runs before the reply is sent, which puts the parked advance back on the "
        "critical path -- the whole point was to take it off"
    )
