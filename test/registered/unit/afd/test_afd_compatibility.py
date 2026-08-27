"""What this arrangement refuses to be run beside, and why each refusal has to fire.

Every feature listed here fails the same way when nothing checks: the server starts, the tokens
come out fluent, and the number that was supposed to be measured is a number about something else.
The cases below are the ones where getting the check itself wrong is easy.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.compatibility import Incompatible, check
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase

# EVERY field the checks read, even the ones this configuration leaves at their default. The
# checks read the server args directly rather than through `getattr(..., default)`, so a stub
# missing a field raises here -- which is the point: a stub that quietly supplies defaults for
# fields the code reads cannot detect a rename either, and the check would go on answering "no"
# forever. Adding a field to a check means adding it here, loudly.
MEASURED = dict(
    afd_mode="host",
    disable_cuda_graph=True,
    disable_decode_cuda_graph=False,
    tp_size=1,
    pp_size=1,
    enable_dp_attention=False,
    speculative_algorithm=None,
    enable_lora=False,
    lora_paths=None,
    disable_overlap_schedule=True,
    enable_multimodal=False,
    rl_on_policy_target=None,
    cuda_graph_config=None,
)


def args(**over):
    return SimpleNamespace(**{**MEASURED, **over})


class TestTheConfigurationThatWasMeasured(CustomTestCase):
    def test_it_is_accepted(self):
        """If the configuration every number was taken on were refused, nothing could be run."""
        self.assertIn("checked", check(args()))


class TestCudaGraph(CustomTestCase):
    """The check whose default direction was wrong, and the failure that hid behind it.

    The first version read the graph config object and answered False when it was absent. A server
    launched without a graph config is the ORDINARY case and has graphs enabled, so the check
    written to catch exactly that walked past it. A predicate that answers "not engaged" when it
    does not know is a predicate that never fires.
    """

    def test_an_ordinary_launch_is_refused(self):
        with self.assertRaises(Incompatible) as caught:
            check(args(disable_cuda_graph=False))
        self.assertIn("cuda_graph", str(caught.exception))

    def test_disabling_by_the_switch_is_accepted(self):
        check(args(disable_cuda_graph=True))

    def test_disabling_in_the_config_is_accepted(self):
        check(
            args(
                disable_cuda_graph=False,
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend="disabled")
                ),
            )
        )

    def test_the_decode_switch_alone_is_enough(self):
        check(args(disable_cuda_graph=False, disable_decode_cuda_graph=True))


class TestBrokenCannotBeOverridden(CustomTestCase):
    """UNTESTED is a request to find out; BROKEN is not.

    An override on a known-wrong combination produces a bug report about sglang from someone who
    was told it would not work, so the two kinds are answered differently and this pins that.
    """

    def _allow(self, value):
        """The override restores whatever was there, including the explicitly-unset case, and it
        does so even if the block raises -- which two of these three cases do on purpose.
        """
        override = envs.SGLANG_AFD_ALLOW_UNTESTED.override(value)
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)

    def test_a_broken_feature_is_refused_even_when_named(self):
        self._allow("cuda_graph,lora")
        with self.assertRaises(Incompatible):
            check(args(disable_cuda_graph=False))
        with self.assertRaises(Incompatible):
            check(args(enable_lora=True))

    def test_an_untested_feature_runs_when_named(self):
        self._allow("tp_size")
        check(args(tp_size=2))

    def test_naming_one_does_not_permit_another(self):
        self._allow("tp_size")
        with self.assertRaises(Incompatible) as caught:
            check(args(tp_size=2, pp_size=2))
        self.assertIn("pp_size", str(caught.exception))


class TestEveryRefusalNamesItselfAndItsReason(CustomTestCase):
    """A refusal that does not say which feature and why is a refusal an operator works around.

    The failure guarded here is a check that grows a case whose message says only "unsupported":
    somebody then disables the whole check to get past it, and every other case goes with it.
    """

    def test_each_message_carries_the_feature_and_a_sentence(self):
        for label, over in (
            ("tp_size", dict(tp_size=2)),
            ("pp_size", dict(pp_size=2)),
            ("dp_attention", dict(enable_dp_attention=True)),
            ("speculative_decoding", dict(speculative_algorithm="EAGLE")),
            ("lora", dict(enable_lora=True)),
            ("multimodal", dict(enable_multimodal=True)),
            ("weight_update", dict(rl_on_policy_target="x")),
            ("cuda_graph", dict(disable_cuda_graph=False)),
        ):
            with self.subTest(feature=label):
                with self.assertRaises(Incompatible) as caught:
                    check(args(**over))
                message = str(caught.exception)
                self.assertIn(label, message)
                # the reason, not just the name: the shortest here is about forty characters
                line = [l for l in message.splitlines() if label in l][0]
                self.assertGreater(len(line), 60, f"{label}'s refusal says too little")


class TestARefusalMustNotCatchTheRunningConfiguration(CustomTestCase):
    """The overlap scheduler was refused, and the deployment it was refusing had it enabled.

    It was listed as untested on the reasoning that a forward overlapping the previous one's
    output would disturb the router's per-layer state -- plausible, and false. The two-machine
    deployment runs with sglang's default (`disable_overlap_schedule=False`) and its tokens match
    the local-split arm exactly. A check that refuses the configuration it was written inside of
    is worse than no check: it is the one an operator disables wholesale to get past.
    """

    def test_the_overlap_scheduler_is_not_refused(self):
        check(args(disable_overlap_schedule=False))


class TestAnArgumentThisBuildDoesNotHave(CustomTestCase):
    """A predicate that reads a missing arg must report "not engaged", not crash.

    sglang's server args change; a check that raises AttributeError on an older or newer build
    takes down every launch rather than the one combination it was written for.
    """

    def test_a_sparse_args_object_is_accepted(self):
        check(SimpleNamespace(afd_mode="host", disable_cuda_graph=True))


if __name__ == "__main__":
    unittest.main()
