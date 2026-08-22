"""Argument checks for the AFD flags, where a flag that reaches nobody is the failure.

Two flags in this arrangement have now been found unwired. `--afd-query-shift-layers` was assumed
by the span rather than read by it, so a run at 0 got a shifted read anyway and reported no
difference. `--afd-coverage` reached nobody at all: it appears in neither span.py nor
span_routing.py, and every run of the group cut passed `all`, logged it, and converted the read
point of 16 layers while naming 63.
"""

import unittest

from sglang.test.test_utils import CustomTestCase


class TestCoverageUnderTheGroupCut(CustomTestCase):
    """`--afd-coverage all` with `--afd-span-cut` asks for a conversion the group cut never does.

    The check lives in the ARM now, not in the shared argument hook: a flag belongs to whoever
    honours it, and the hook naming this package by name is what stopped AFD shipping without it.

    The group cut moves ONE query per group -- the next softmax attention's, projected from the
    read point inside the span -- and its linear-attention layers take q, k and v from the current
    hidden. That is coverage "softmax".

    The flag reached nobody: it appears in neither span.py nor span_routing.py. Every run of this
    arrangement passed `--afd-coverage all`, logged it, and converted the read point of 16 layers
    while reporting 63 -- the cost of a shallower conversion under a deeper one's name, which is
    the failure `afd-early-q` names in as many words. It is the second flag found unwired in this
    cut; the first was `--afd-query-shift-layers`.
    """

    def args(self, **kw):
        from sglang.srt.server_args import ServerArgs

        a = ServerArgs.__new__(ServerArgs)
        a.afd_span_cut = kw.get("span_cut", True)
        a.afd_coverage = kw.get("coverage", None)
        return a

    def check(self, a):
        from sglang.srt.afd_query_shift.arg_checks import _check_coverage

        _check_coverage(a)

    def test_all_is_refused_under_the_span_cut(self):
        with self.assertRaises(ValueError) as caught:
            self.check(self.args(coverage="all"))
        self.assertIn("coverage 'softmax'", str(caught.exception))

    def test_softmax_is_what_the_group_cut_does(self):
        self.check(self.args(coverage="softmax"))

    def test_unset_is_not_refused(self):
        """Unset resolves to 'all' downstream for the per-layer cut. Refusing a default nobody
        chose would break every command line that never mentioned coverage."""
        self.check(self.args(coverage=None))

    def test_the_per_layer_cut_still_takes_all(self):
        """The control. Without this the refusal could be unconditional and the case above would
        pass while the other cut lost a setting it does implement."""
        self.check(self.args(coverage="all", span_cut=False))


if __name__ == "__main__":
    unittest.main()
