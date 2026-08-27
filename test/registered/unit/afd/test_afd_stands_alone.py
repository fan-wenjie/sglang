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


def _repo_root(start: str) -> str:
    """Walk up until the directory that holds `python/sglang`, rather than counting `..`.

    Counted hops break silently on a move: this file went from `test/registered/unit/` to
    `test/registered/unit/afd/` and three `..` then pointed at `<repo>/test/python/sglang/...`,
    which does not exist. A marker cannot be off by one.
    """
    here = start
    while True:
        if os.path.isdir(os.path.join(here, "python", "sglang")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            raise RuntimeError(f"no repository root above {start}")
        here = parent


HERE = os.path.dirname(os.path.abspath(__file__))
AFD = os.path.join(_repo_root(HERE), "python", "sglang", "srt", "afd")
DERIVED = re.compile(r"afd_query_shift|afd_eq\b|from\s+\.\.afd_\w+|srt\.afd_\w+")


class TestStandardAfdNamesNoDerivedArm(CustomTestCase):
    def afd_files(self):
        return [
            os.path.join(AFD, f) for f in sorted(os.listdir(AFD)) if f.endswith(".py")
        ]

    LAZY_SEAMS = {
        "span.py": 7,
        "span_routing.py": 1,
        "installer.py": 1,
        "checkpoint.py": 1,
    }

    def test_the_derived_package_is_reached_only_through_gated_lazy_imports(self):
        """The boundary, after the span moved down: base serves shift 0 whole, and the
        derived package holds only the early read. Base may import it -- inside gates a
        resolved shift of 0 never takes -- so the invariant is the SHAPE of the imports:
        none at module level (deleting the derived directory must leave shift 0 serving
        and shift 1 refused by name, not an ImportError at process start), and the lazy
        seams pinned by count so they ratchet down, never quietly up."""
        module_level = {}
        lazy = {}
        for path in self.afd_files():
            for i, line in enumerate(open(path).read().split("\n"), 1):
                if "afd_query_shift" not in line or "import" not in line:
                    continue
                stripped = line.strip()
                if not (
                    stripped.startswith("from sglang.srt.afd_query_shift")
                    or stripped.startswith("import sglang.srt.afd_query_shift")
                ):
                    continue
                name = os.path.basename(path)
                if line[0] not in (" ", "\t"):
                    module_level.setdefault(name, []).append(i)
                else:
                    lazy[name] = lazy.get(name, 0) + 1
        self.assertEqual(
            module_level,
            {},
            "module-level imports of the derived package: base must start without it",
        )
        for name, count in lazy.items():
            self.assertLessEqual(
                count,
                self.LAZY_SEAMS.get(name, 0),
                f"{name} grew a lazy seam into the derived package beyond the "
                f"{self.LAZY_SEAMS.get(name, 0)} pinned -- the list ratchets down",
            )

    def test_no_file_under_afd_is_an_experiment_s_arm_registry(self):
        """A second check, shaped differently, because the grep above cannot see this class.

        `early_k_arms.py` sat under srt/afd on all three branches -- derived-line content, the
        three arms of the query-shift line's pre-registered key question -- and the grep walked
        past it every run. It never spelled the derived package's name. It did not have to: it
        imported nothing from it, and nothing imported it. A name-grep answers "does this file
        MENTION the derived line", and the property wanted is "does this file BELONG to it".

        So this asks about shape instead: an experiment's arms live in the derived package, and
        under srt/afd the only module that may hold the word is `arms.py`, which is the REGISTRY
        -- the place an arm announces itself from outside, holding no arm of its own.
        """
        offenders = []
        for path in self.afd_files():
            name = os.path.basename(path)
            if name == "arms.py":
                continue
            if name.endswith("_arms.py"):
                offenders.append(f"{name}: named as an arm module")
            elif re.search(r"^ARMS\s*=", open(path).read(), re.M):
                offenders.append(f"{name}: defines a module-level ARMS")
        self.assertEqual(
            offenders,
            [],
            "an experiment's arms belong to the derived package, not to standard AFD",
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
        # No setting given and a path with no checkpoint at it: the arm is not chosen, which is
        # the only way to say so now that it has no flag of its own.
        off = types.SimpleNamespace(
            afd_mode="null",
            afd_pool_addr=None,
            model_path="/nonexistent",
            afd_query_shift_layers=None,
            afd_coverage=None,
        )
        self.assertEqual(absent_classes(off), ())

    def test_installing_transforms_loads_the_registry_in_this_process(self):
        """The spawn boundary, guarded where it was crossed.

        sglang runs the model loader and the model runner in a scheduler process it SPAWNS. An arm
        imported by the launcher registers in the parent and leaves the child's registry empty, so
        every entry point that consults the registry has to load it in the process that consults
        it. `arms.load` exists for that and its docstring says so -- and the function added after
        it, `install_transforms`, did not call it.

        Nothing failed. The arm simply stopped installing, and a quality re-run reported the
        shifted arm's bits per byte as exactly the unshifted arm's, 0.713010 against 0.713010: an
        arm that is not installed is indistinguishable from an arm that costs nothing. It was
        caught by a measurement that was expected to differ, which is not a way to find bugs.

        Asserted through `_LOADED` rather than through an arm, so it means the same thing on a
        build that carries no arms -- which is the build this file is about.
        """
        import sglang.srt.afd.arms as arms

        arms._LOADED = False
        try:
            arms.install_transforms(
                model=object(), model_config=object(), server_args=object()
            )
        except Exception:  # noqa: BLE001
            # On a build that CARRIES an arm, the arm is then asked to transform a bare
            # object and refuses. That is fine and is not what this asserts: the load
            # happens before any arm is consulted, so the registry's state after the
            # attempt is the whole question. With no arms installed nothing raises.
            pass
        self.assertTrue(
            arms._LOADED, "install_transforms consulted a registry it never loaded"
        )

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
