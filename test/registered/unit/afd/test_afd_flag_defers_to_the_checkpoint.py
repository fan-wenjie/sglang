"""The checkpoint names its own read point; the flag is for the measurement that overrides it.

A model trained for this rewiring declares `query_shift_layers` in its own config, and UNSET is
how a deployment gets that value -- `checkpoint.resolve` asks the flag first and the checkpoint
second, so a server told nothing serves what its weights were repaired for. The flag exists for
the one question that needs a checkpoint run at a read point it was not repaired for: what the
rewiring costs before repair.

Every other use projects each layer's query from an input the weights never saw, and the symptom
is fluent text from a model nobody trained. Nothing downstream can catch it, so the only place it
can be caught is here, at the flag.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd import checkpoint
from sglang.srt.afd.checkpoint import resolved_shift, stated_shift
from sglang.srt.afd_query_shift.arg_checks import _check_shift
from sglang.test.test_utils import CustomTestCase


def a_checkpoint(states=None):
    home = tempfile.mkdtemp(prefix="afd-shift-")
    config = {"architectures": ["Qwen3_5ForCausalLM"]}
    if states is not None:
        config["query_shift_layers"] = states
    with open(os.path.join(home, "config.json"), "w") as handle:
        json.dump(config, handle)
    return home


def resolved(server_args):
    """`resolved_shift` memoises into a module global, which is right in a server -- the read
    point is decided once -- and makes any two of these cases the same case. Cleared per call."""
    checkpoint._RESOLVED = None
    try:
        return resolved_shift(server_args)
    finally:
        checkpoint._RESOLVED = None


def args(path, flag=None, mode="pool"):
    return SimpleNamespace(
        model_path=path, afd_query_shift_layers=flag, afd_mode=mode
    )


class TestTheCheckpointIsTheDefault(CustomTestCase):
    def test_unset_reads_the_checkpoint(self):
        self.assertEqual(stated_shift(a_checkpoint(states=1)), 1)

    def test_a_checkpoint_that_states_nothing_reads_as_the_standard_wiring(self):
        """None and 0 are different things at the flag and the same thing at the read point."""
        self.assertIsNone(stated_shift(a_checkpoint()))
        self.assertEqual(resolved(args(a_checkpoint())), 0)

    def test_unset_is_silent(self):
        with self.assertNoLogs("sglang.srt.afd_query_shift.arg_checks", level="INFO"):
            _check_shift(args(a_checkpoint(states=1)))


class TestTheFlagSaysItIsOverriding(CustomTestCase):
    def _logs(self, path, flag):
        with self.assertLogs("sglang.srt.afd_query_shift.arg_checks", level="INFO") as got:
            _check_shift(args(path, flag=flag))
        return got

    def test_contradicting_the_checkpoint_warns_and_names_both(self):
        got = self._logs(a_checkpoint(states=1), flag=0)
        joined = "\n".join(got.output)
        self.assertIn("WARNING", joined)
        self.assertIn("CONTRADICTS", joined)

    def test_a_checkpoint_that_states_nothing_warns_that_this_is_a_measurement(self):
        got = self._logs(a_checkpoint(), flag=1)
        joined = "\n".join(got.output)
        self.assertIn("WARNING", joined)
        self.assertIn("measurement", joined)

    def test_agreeing_with_the_checkpoint_is_not_a_warning(self):
        """Redundant is not dangerous, and crying wolf here costs the warnings above their force."""
        got = self._logs(a_checkpoint(states=1), flag=1)
        self.assertFalse([line for line in got.output if line.startswith("WARNING")])

    def test_the_flag_still_wins_whatever_it_said(self):
        """The warning is a warning. An operator running the measurement gets the measurement."""
        self.assertEqual(resolved(args(a_checkpoint(states=1), flag=0)), 0)


class TestAHostIsExempt(CustomTestCase):
    def test_a_pool_provisioned_host_has_no_config_to_contradict(self):
        """Its path is `pool://host:port` and its word is the pool's; a host whose flag disagrees
        with what the pool pushed is refused at adoption, not here."""
        with self.assertNoLogs("sglang.srt.afd_query_shift.arg_checks", level="WARNING"):
            _check_shift(args("pool://10.0.0.1:8999", flag=1, mode="host"))


if __name__ == "__main__":
    unittest.main()
