"""Every reader of the query shift resolves it the same way: through the checkpoint.

The shift belongs to the CHECKPOINT. A model whose `W_q` was repaired to read `h_(l-1)` must be
served at that read point, and serving it at the standard one gives a query projection trained
for an input it is no longer given -- fluent output from weights that no longer match their
wiring, and nothing that fails.

`resolve_shift` is where that is decided. Reading the raw setting instead is a split brain with
no symptom, and it had already happened: the wiring resolved through the checkpoint while the
pool's span runner read the setting directly, so a checkpoint declaring `query_shift_layers: 1`
with no override installed the shifted arrangement and then ran without it.

This pins that no reader takes the raw setting except the two that are allowed to: the resolution
itself, and the argument validation that runs before a model is loaded.
"""

import ast
import pathlib

SRT = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"
QS = SRT / "afd"

SETTING = "afd_query_shift_layers"

# `checkpoint.py` IS the resolution. `arg_checks.py` runs before a checkpoint is in hand -- it
# rejects a value no checkpoint could make sensible, which is a different question from which
# value to serve at. Every other reader has a model and must go through `resolve_shift`.
# `pushed.py` is transport: the pool folds the raw REQUEST into the HELLO and the host
# adopts the same request before anything resolves, so both ends resolve from inputs the
# handshake made identical. It moves the value between the two resolutions and serves at
# neither.
MAY_READ_IT_RAW = {"checkpoint.py", "arg_checks.py", "pushed.py"}


def _value_reads(tree):
    """The attribute nodes in `tree` that use the setting's VALUE.

    Asking whether the override was GIVEN is a different question from what to serve at, and it
    has an answer the checkpoint cannot supply: `x is None` selects between "the deployment path"
    and "a measurement", and only the second has a value to resolve. So a presence test does not
    count here; using the value does.
    """
    presence = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.comparators) != 1:
            continue
        if not isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            continue
        right = node.comparators[0]
        if not (isinstance(right, ast.Constant) and right.value is None):
            continue
        if isinstance(node.left, ast.Attribute) and node.left.attr == SETTING:
            presence.add(id(node.left))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == SETTING
        and id(node) not in presence
    ]


def _readers():
    """Files under afd_query_shift that read the raw setting's value, and how many times.

    Asking whether the override was GIVEN is a different question from what to serve at, and it
    has an answer the checkpoint cannot supply: `x is None` selects between "the deployment path"
    and "a measurement", and only the second has a value to resolve. So a presence test does not
    count here; using the value does.
    """
    found = {}
    for path in sorted(QS.glob("*.py")):
        hits = len(_value_reads(ast.parse(path.read_text())))
        if hits:
            found[path.name] = hits
    return found


def test_only_the_resolution_and_the_argument_check_read_it_raw():
    stray = {name: n for name, n in _readers().items() if name not in MAY_READ_IT_RAW}
    # installer.py may name it once, where it hands it to `resolve_shift` as the override.
    if _installer_only_forwards():
        stray.pop("installer.py", None)
    assert not stray, (
        f"{stray} read {SETTING} without resolving it against the checkpoint. A reader that "
        f"takes the raw value disagrees with every reader that does not, and the disagreement "
        f"is silent: the arrangement is installed and then not run"
    )


def _installer_only_forwards() -> bool:
    """True when every mention in installer.py is an argument to a resolving call."""
    tree = ast.parse((QS / "installer.py").read_text())
    total = len(_value_reads(tree))
    forwarded = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else getattr(node.func, "attr", "")
        )
        if name not in ("resolve_shift", "install_query_shift"):
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Attribute) and arg.attr == SETTING:
                forwarded += 1
    return total > 0 and forwarded == total


def test_the_pool_asks_for_the_resolved_value():
    """Named as well as covered: this is the reader that gates the arrangement itself."""
    tree = ast.parse((QS / "installer.py").read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_span_query_shift"
    )
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "resolved_shift"
        for n in ast.walk(fn)
    ), (
        "_span_query_shift no longer asks for the resolved read point. It decides whether the "
        "span sends its early projection at all, so a value from anywhere else turns the whole "
        "arrangement off for any checkpoint that declared it"
    )


def test_the_transform_is_the_one_resolution_point():
    """`resolve_shift` is called from the arm's transform -- before any install, once.

    The failure this order prevents: a process that resolves late, or in two places,
    records one read point and serves another, and the two ends then refuse each other
    at the first frame with each naming the other's configuration."""
    tree = ast.parse((QS / "installer.py").read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "transform_model"
    )
    calls = [
        n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "resolve_shift" in calls
