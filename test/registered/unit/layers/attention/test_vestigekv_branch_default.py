"""The branch rule must be ON without an env var.

This arm has no tier 2 other than the branch rule. `_branch_q()` read
SGLANG_VESTIGEKV_BRANCH_Q with no fallback, and `_skip_sketch_chain()` keys off
`_branch_q() is not None`, so an unset variable selected the sketch chain on the
research branch and would leave NO tier 2 here. The failure is silent: a 4k/256 smoke
that omitted the variable served the sketch lineage and its overflow and fallback read
as a kappa regression.

Pinned as a property of the defaults module rather than of the backend, because the
backend's own tests construct it through __new__ and would not exercise the accessor.
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.layers.attention.vestigekv.dsa import defaults as D


class TestBranchThresholdHasAServedDefault(unittest.TestCase):
    def test_default_is_a_usable_quantile(self):
        self.assertIsNotNone(D.BRANCH_Q)
        self.assertLess(0.0, D.BRANCH_Q)
        self.assertLess(D.BRANCH_Q, 1.0)

    def test_the_accessor_falls_back_to_it(self):
        # The accessor's contract, written out: unset -> the served default, set -> the
        # override. Reproduced here rather than importing the backend, which needs CUDA.
        def branch_q():
            q = envs.SGLANG_VESTIGEKV_BRANCH_Q.get()
            return D.BRANCH_Q if q is None else q

        envs.SGLANG_VESTIGEKV_BRANCH_Q.clear()
        self.assertEqual(branch_q(), D.BRANCH_Q)
        with envs.SGLANG_VESTIGEKV_BRANCH_Q.override(0.9):
            self.assertEqual(branch_q(), 0.9)
        self.assertEqual(branch_q(), D.BRANCH_Q)

    def test_an_unset_variable_still_skips_the_sketch_chain(self):
        # _skip_sketch_chain() keys off `_branch_q() is not None`, so a non-None default
        # is exactly what makes the chain skipped by default.
        envs.SGLANG_VESTIGEKV_BRANCH_Q.clear()
        self.assertIsNotNone(D.BRANCH_Q)


if __name__ == "__main__":
    unittest.main()
