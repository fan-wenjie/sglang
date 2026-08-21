"""The read point comes from the checkpoint; the command line may override it, loudly.

A checkpoint repaired at one read point must be served at it. Fine-tuning W_q to read h_(l-1) and
then serving at the standard read point gives that projection an input it was never trained on --
fluent output from weights that no longer match their wiring, and nothing that fails. So the
config file carries the read point and a contradicting flag warns.

The case this pins down and no other test covers: `unset` and `explicitly 0` must be DIFFERENT.
With a default of 0 they would be the same value, and the override warning could never fire for a
repaired checkpoint served at the standard wiring -- which is the failure it exists for.
"""

import logging
import unittest


class FakeText:
    pass


class FakeConfig:
    pass


def _config(shift=None, coverage=None, nested=True):
    cfg = FakeConfig()
    target = FakeText() if nested else cfg
    if shift is not None:
        target.afd_q_shift_layers = shift
    if coverage is not None:
        target.afd_coverage = coverage
    if nested:
        cfg.text_config = target
    return cfg


class TestResolveShift(unittest.TestCase):
    def test_unset_takes_the_checkpoints_own(self):
        from sglang.srt.afd.checkpoint import resolve_shift

        self.assertEqual(resolve_shift(None, _config(shift=1)), 1)

    def test_unset_on_an_unconverted_checkpoint_is_standard(self):
        from sglang.srt.afd.checkpoint import resolve_shift

        self.assertEqual(resolve_shift(None, _config()), 0)

    def test_explicit_zero_is_not_the_same_as_unset(self):
        """The whole reason the default is None. A repaired checkpoint served at the standard
        wiring must warn, and it cannot if unset and 0 are one value."""
        from sglang.srt.afd.checkpoint import resolve_shift

        with self.assertLogs("sglang.srt.afd.checkpoint", level=logging.WARNING) as caught:
            self.assertEqual(resolve_shift(0, _config(shift=1)), 0)
        self.assertIn("OVERRIDES", "".join(caught.output))

    def test_agreeing_with_the_checkpoint_is_silent(self):
        from sglang.srt.afd.checkpoint import resolve_shift

        logger = logging.getLogger("sglang.srt.afd.checkpoint")
        with self.assertNoLogs(logger, level=logging.WARNING):
            self.assertEqual(resolve_shift(1, _config(shift=1)), 1)

    def test_a_shift_on_an_unconverted_checkpoint_warns(self):
        """Legitimate -- it is how the study's forward-only numbers were taken -- and still a
        warning, because the weights were never repaired for this read point. Only a specific
        intent wants it, and a deployment that arrives here arrived by accident."""
        from sglang.srt.afd.checkpoint import resolve_shift

        with self.assertLogs("sglang.srt.afd.checkpoint", level=logging.WARNING) as caught:
            self.assertEqual(resolve_shift(1, _config()), 1)
        self.assertIn("nothing has repaired", "".join(caught.output))

    def test_standard_wiring_on_an_unconverted_checkpoint_is_silent(self):
        """The one arrangement nothing is wrong with, however it was spelled. Serving stock
        weights at the stock read point must stay quiet whether the flag was omitted or written
        out, or every ordinary launch carries an afd warning and the real ones stop being read."""
        from sglang.srt.afd.checkpoint import resolve_shift

        logger = logging.getLogger("sglang.srt.afd.checkpoint")
        with self.assertNoLogs(logger, level=logging.WARNING):
            self.assertEqual(resolve_shift(0, _config()), 0)

    def test_the_top_level_is_read_when_there_is_no_text_config(self):
        from sglang.srt.afd.checkpoint import resolve_shift

        self.assertEqual(resolve_shift(None, _config(shift=2, nested=False)), 2)

    def test_a_checkpoint_stating_nonsense_raises(self):
        from sglang.srt.afd.checkpoint import resolve_shift

        with self.assertRaises(TypeError):
            resolve_shift(None, _config(shift="half"))


class TestResolveCoverage(unittest.TestCase):
    def test_unset_takes_the_checkpoints_own(self):
        from sglang.srt.afd.checkpoint import resolve_coverage

        self.assertEqual(resolve_coverage(None, _config(coverage="softmax")), "softmax")

    def test_unset_with_nothing_stated_is_all(self):
        from sglang.srt.afd.checkpoint import resolve_coverage

        self.assertEqual(resolve_coverage(None, _config()), "all")

    def test_contradicting_the_checkpoint_warns(self):
        from sglang.srt.afd.checkpoint import resolve_coverage

        with self.assertLogs("sglang.srt.afd.checkpoint", level=logging.WARNING) as caught:
            self.assertEqual(resolve_coverage("all", _config(coverage="softmax")), "all")
        self.assertIn("overrides", "".join(caught.output))

    def test_a_checkpoint_stating_an_unknown_coverage_raises(self):
        from sglang.srt.afd.checkpoint import resolve_coverage

        with self.assertRaises(ValueError):
            resolve_coverage(None, _config(coverage="every-other"))


class TestStamp(unittest.TestCase):
    def test_a_conversion_can_write_the_read_point_into_the_config(self):
        """So the shift travels with the weights and nobody has to remember it."""
        from sglang.srt.afd.checkpoint import stamp

        out = stamp({"text_config": {"num_hidden_layers": 64}}, shift=1, coverage="all")
        self.assertEqual(out["text_config"]["afd_q_shift_layers"], 1)
        self.assertEqual(out["text_config"]["afd_coverage"], "all")
        self.assertEqual(out["text_config"]["num_hidden_layers"], 64, "the rest is untouched")

    def test_stamping_does_not_mutate_the_input(self):
        from sglang.srt.afd.checkpoint import stamp

        original = {"text_config": {"a": 1}}
        stamp(original, shift=1, coverage="all")
        self.assertNotIn("afd_q_shift_layers", original["text_config"])


if __name__ == "__main__":
    unittest.main()
