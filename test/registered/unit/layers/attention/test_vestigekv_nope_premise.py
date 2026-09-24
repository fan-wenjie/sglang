"""These MLA layers must apply no rotation; VestigeKV's premise is that they do not.

Tier 1 scores the sidecar with a low-pass residual along the token axis and
tier 2's exact summand is an inner product against the same branch. Under a
rotation, equal content at distinct positions is not equal as rows and neither
holds. The paper states the premise as `skip_rope=True in sglang`.

`config.json` declares `qk_rope_head_dim = 64`, which is the width of the slot
rotation would have used, not evidence that it is used.
"""

import ast
import os
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

# CPU: this reads source, it does not run the model. Registering it on a GPU
# runner would spend a device to parse two files. It lives under unit/ rather
# than kernel/ for the same reason: it launches nothing and touches no device.
register_cpu_ci(est_time=1, suite="stage-a-test-cpu-intel")

SRT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "..",
    "..",
    "python",
    "sglang",
    "srt",
)


def _source(rel):
    with open(os.path.normpath(os.path.join(SRT, rel))) as f:
        return f.read()


class TestVestigeKVNoPEPremise(unittest.TestCase):
    def test_kimi_linear_builds_its_mla_layers_without_rope(self):
        tree = ast.parse(_source("models/kimi_linear.py"))
        passed = [
            kw.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == "skip_rope" and isinstance(kw.value, ast.Constant)
        ]
        self.assertTrue(
            passed,
            "kimi_linear.py no longer passes skip_rope at all. VestigeKV's "
            "sidecar statistic assumes an un-roped branch; if this model now "
            "ropes it, tier 1 is scoring a rotation.",
        )
        self.assertTrue(
            all(v is True for v in passed),
            f"kimi_linear.py passes skip_rope={passed}, not True.",
        )

    def test_skip_rope_suppresses_the_rotary_embedding(self):
        # A flag that is threaded but ignored leaves the premise false while
        # the call-site test above still passes, so the guard has to be read
        # where the rotation would be built.
        tree = ast.parse(_source("models/deepseek_v2.py"))
        guarded = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if "skip_rope" not in names:
                continue
            body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
            if "rotary_emb" in body:
                guarded = True
        self.assertTrue(
            guarded,
            "no `if ... skip_rope ...` branch guards the rotary embedding's "
            "construction in deepseek_v2.py. skip_rope may now be accepted and "
            "ignored, which would rope the branch while the call site still "
            "reads skip_rope=True.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
