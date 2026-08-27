"""Nothing follows the accept loop: `serve` never returns, so late wiring is dead code.

The riders path used to be assigned after the `serve` call and was dead for its whole
life, silently; a lane serving loop later repeated the mistake and hung a host, which is
how the pattern was found. Everything the composition root wants on the built departure
goes through `after_built`, and this pins the pattern shut.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.test.test_utils import CustomTestCase


class TestNothingFollowsTheAcceptLoop(CustomTestCase):
    def test_run_pool_wires_everything_before_serve(self):
        import ast
        import pathlib

        src = (
            pathlib.Path(__file__).resolve().parents[4]
            / "python/sglang/srt/afd/roles.py"
        ).read_text()
        tree = ast.parse(src)
        run_pool = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_pool"
        )
        serve_at = next(
            i
            for i, stmt in enumerate(run_pool.body)
            if any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == "serve"
                for c in ast.walk(stmt)
            )
        )
        after = run_pool.body[serve_at + 1 :]
        self.assertTrue(
            len(after) == 1 and isinstance(after[0], ast.Return),
            "run_pool has statements after serve(); they are dead code -- wire them "
            "through after_built instead",
        )


if __name__ == "__main__":
    unittest.main()
