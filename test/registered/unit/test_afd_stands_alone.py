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

from sglang.test.test_utils import CustomTestCase

HERE = os.path.dirname(os.path.abspath(__file__))
AFD = os.path.abspath(os.path.join(HERE, "..", "..", "..", "python", "sglang", "srt", "afd"))
DERIVED = re.compile(r"afd_query_shift|afd_eq\b|from\s+\.\.afd_\w+|srt\.afd_\w+")


class TestStandardAfdNamesNoDerivedArm(CustomTestCase):
    def afd_files(self):
        return [os.path.join(AFD, f) for f in sorted(os.listdir(AFD)) if f.endswith(".py")]

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
    KNOWN = {
        "roles.py": 1,        # _span_query_shift, leaves with the arm's installation
    }

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
                counted.get(name, 0), allowed,
                f"{name} names a derived package more often than the {allowed} known -- the "
                f"allowlist ratchets down, never up",
            )
            counted.pop(name, None)
        self.assertEqual(
            counted, {},
            "these files under srt/afd name a derived package, so standard AFD cannot be shipped "
            "without it",
        )

    def test_the_registry_is_empty_without_a_derived_package(self):
        """`resolve` returning None is the ordinary state, not an error state."""
        from sglang.srt.afd.arms import resolve

        self.assertIsNone(resolve("query-shift"))
        self.assertIsNone(resolve(None))

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
