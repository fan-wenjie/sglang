"""The code's own operator terms, pinned at the source.

The measurement arm this file once compared against (`key_shift_arms`) left with the
measurement scaffolding; what survives is the pin on the CODE's side of the agreement
it once enforced -- which tensor goes into which term, a property invisible in any
output, which is why it is read from the source.
"""

import ast
import pathlib

SRT = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"


def _fn(path: str, name: str):
    for node in ast.walk(ast.parse((SRT / path).read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path} no longer defines {name}")


def test_the_code_does_not_form_the_coefficient_when_given_a_query():
    """Given a query, the contraction was done elsewhere with it; recomputing the coefficient here
    would be a second one from a different projection, which is what the mismatch was.
    """
    node = _fn("afd/linear_history.py", "core_from_mixed")
    for branch in ast.walk(node):
        if not isinstance(branch, ast.If):
            continue
        if "query is None" not in ast.unparse(branch.test):
            continue
        taken = ast.unparse(ast.Module(body=branch.orelse, type_ignores=[]))
        assert (
            "query_coefficient" not in taken
        ), "the code forms a query coefficient even when it was given a query and a contraction"
        return
    raise AssertionError(
        "core_from_mixed no longer branches on whether a query was given"
    )
