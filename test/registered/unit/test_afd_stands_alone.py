"""Standard AFD must run with the derived code ABSENT, not merely disabled.

AFD may land upstream before anything derived from it does. A flag defaulting to off is not the
same property: a module that imports the derived package inside an `if` still fails when the
directory is not there, and it fails at the first request rather than at startup, on a machine
that has no way to fix it.

So this checks the shape rather than the behaviour -- no file under `sglang/srt/afd/` may name a
derived package at all. A derived arm registers itself through `afd.arms`; nothing here reaches
back for it.
"""

import os
import re
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.test.test_utils import CustomTestCase

HERE = os.path.dirname(os.path.abspath(__file__))
AFD = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "python", "sglang", "srt", "afd")
)
DERIVED = re.compile(r"afd_query_shift|afd_eq\b|from\s+\.\.afd_\w+|srt\.afd_\w+")


class TestStandardAfdNamesNoDerivedArm(CustomTestCase):
    def afd_files(self):
        return [
            os.path.join(AFD, f) for f in sorted(os.listdir(AFD)) if f.endswith(".py")
        ]

    # The debt this check found on the day it was written. A RATCHET, not an exemption: nothing
    # may be added, and every entry leaves by RELOCATION rather than by editing.
    #
    # `checkpoint.py` is derived-arm code in its entirety -- its whole subject is the read point a
    # checkpoint states, and its code names the shift eighteen times. This grep catches it on one
    # literal flag name, but deleting that line would leave the file where it is and the property
    # still broken. It goes to the derived package.
    #
    # `roles.py` holds `_span_query_shift`, which validates the group cut's shift. Same: it is the
    # derived arm's rule living in the shared composition root, and it leaves when the arm's
    # installation does.
    #
    # Recorded here because the obvious repair -- edit the offending line -- would make the grep
    # pass while the architecture stayed wrong, which is worse than the debt.
    # EMPTY, and it must stay empty. Both entries it opened with left by relocation rather than
    # by editing the line the grep found -- checkpoint.py and wiring.py to the derived package,
    # and roles.py's five installer functions with them. Nothing under srt/afd names a derived
    # arm now, which is the property upstream needs and the reason this file exists.
    KNOWN = {}

    def test_no_new_file_in_afd_names_a_derived_package(self):
        """The whole property, as one grep. If this fails, deleting the derived directory breaks
        standard AFD -- which is the deployment upstream would get first.

        Green against a shrinking allowlist rather than red against the truth: the two entries in
        KNOWN are real debt with a real fix, and a red test in the tree would stop being read
        before it stopped being true.
        """
        offenders = {}
        for path in self.afd_files():
            hits = [
                f"{i}: {line.strip()}"
                for i, line in enumerate(open(path).read().split("\n"), 1)
                if DERIVED.search(line)
            ]
            if hits:
                offenders[os.path.basename(path)] = hits
        counted = {f: len(h) for f, h in offenders.items()}
        for name, allowed in self.KNOWN.items():
            self.assertLessEqual(
                counted.get(name, 0),
                allowed,
                f"{name} names a derived package more often than the {allowed} known -- the "
                f"allowlist ratchets down, never up",
            )
            counted.pop(name, None)
        self.assertEqual(
            counted,
            {},
            "these files under srt/afd name a derived package, so standard AFD cannot be shipped "
            "without it",
        )

    def test_resolving_an_arm_nobody_registered_gives_none(self):
        """`resolve` returning None is the ordinary state, not an error state.

        Asked about a name nothing will ever register, rather than about "query-shift". The first
        version asserted the registry was empty, which is a property of the TEST SESSION and not
        of the source: an argument check that imports the arm's package to see whether the build
        carries it registers it globally, and this went red the moment that check existed. The
        property worth pinning is that an unknown name resolves to nothing.
        """
        from sglang.srt.afd.arms import resolve

        self.assertIsNone(resolve("an-arm-that-does-not-exist"))
        self.assertIsNone(resolve(None))

    def test_an_arm_that_was_not_asked_for_skips_nothing(self):
        """The fifth entry point reaches the LOADER, which is where a wrong answer costs most: a
        class named here is built on the meta device before any routing exists, so an arm that
        declared one while switched off would produce a model that fails on its first token with a
        message about devices and nothing about the arm.

        Asked with every arm's own flag off. On a build with no arms this is the empty registry --
        the ordinary case -- and on a build with them it is the more interesting one.
        """
        import types

        from sglang.srt.afd.arms import absent_classes, load

        load()
        off = types.SimpleNamespace(
            afd_span_cut=False,
            afd_mode="null",
            afd_pool_addr=None,
            afd_query_shift_layers=None,
            afd_coverage=None,
        )
        self.assertEqual(absent_classes(off), ())

    def test_two_arms_under_one_name_are_refused(self):
        """Import order would otherwise decide which arrangement a run measured, and nothing in
        the output would record which."""
        from sglang.srt.afd.arms import register

        register("a-test-arm", lambda: None)
        with self.assertRaises(ValueError) as caught:
            register("a-test-arm", lambda: None)
        self.assertIn("import order", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
