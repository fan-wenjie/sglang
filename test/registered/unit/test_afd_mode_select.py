"""Choosing the arrangement, and the two ways that choice can be silently wrong.

The rule is that whichever piece of a layer costs more should be the remote one, and the two costs
cross at a context this selector computes from a calibration. Getting it wrong is not visible in
any output: both arrangements produce the same tokens, so a request served by the slower one just
takes longer, and a request that FLIPS BACK is served by an arrangement that does not know where
its history lives.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.afd.mode_select import FFN_REMOTE, SWEEP_REMOTE, ModeSelector
from sglang.srt.afd.rendezvous import DepartureQueue
from sglang.test.test_utils import CustomTestCase

# measured at batch 64 on one Blackwell: 380 us of feed-forward, 40.95 us per 1000 tokens
BATCH_64 = dict(feed_forward_us=380.0, slope_us_per_token=0.04095)


class TestTheThresholdIsTheMeasuredOne(CustomTestCase):
    def test_it_is_the_feed_forward_divided_by_the_slope(self):
        s = ModeSelector(**BATCH_64)
        self.assertAlmostEqual(s.threshold_context, 380.0 / 0.04095, places=3)
        self.assertAlmostEqual(s.threshold_context, 9280, delta=300)

    def test_below_it_the_feed_forward_is_the_remote_piece(self):
        s = ModeSelector(**BATCH_64)
        self.assertEqual(s.mode_for(1, 1024), FFN_REMOTE)
        self.assertEqual(s.mode_for(1, 8000), FFN_REMOTE)

    def test_above_it_the_sweep_is(self):
        s = ModeSelector(**BATCH_64)
        self.assertEqual(s.mode_for(1, 32768), SWEEP_REMOTE)

    def test_a_flat_slope_is_refused(self):
        """A slope of zero puts the threshold at infinity and this selector would never flip --
        which is a calibration nobody ran, not a model whose sweep is free."""
        with self.assertRaises(ValueError):
            ModeSelector(feed_forward_us=380.0, slope_us_per_token=0.0)


class TestARequestCrossesOnceAndStays(CustomTestCase):
    def test_it_flips_exactly_once_as_the_context_grows(self):
        s = ModeSelector(**BATCH_64)
        modes = [s.mode_for(1, ctx) for ctx in (100, 5000, 9000, 9500, 20000, 100000)]
        self.assertEqual(modes.count(FFN_REMOTE), 3)
        self.assertEqual(modes.count(SWEEP_REMOTE), 3)
        self.assertEqual(s.report()["flips"], 1, "one crossing, not one per call")

    def test_it_never_flips_back(self):
        """The failure this prevents: a retraction shortens a request, the selector reverts, and
        the request is served by an arrangement that does not hold the history it already has."""
        s = ModeSelector(**BATCH_64)
        s.mode_for(1, 32768)
        self.assertEqual(s.mode_for(1, 100), SWEEP_REMOTE)
        self.assertEqual(s.report()["context_shrinks"], 1)

    def test_forgetting_a_request_lets_a_reused_id_start_over(self):
        """Slots are reused. Without this, request 7's successor inherits request 7's mode and is
        served against a cache that was released."""
        s = ModeSelector(**BATCH_64)
        s.mode_for(7, 32768)
        s.forget(7)
        self.assertEqual(s.mode_for(7, 100), FFN_REMOTE)

    def test_requests_are_tracked_apart(self):
        s = ModeSelector(**BATCH_64)
        s.mode_for(1, 32768)
        self.assertEqual(s.mode_for(2, 100), FFN_REMOTE)
        report = s.report()
        self.assertEqual((report["ffn_remote"], report["sweep_remote"]), (1, 1))


class TestPaddingIsForTheWeightSharedWorkOnly(CustomTestCase):
    def test_the_width_is_the_padded_one(self):
        q = DepartureQueue(min_batch=1, max_interval_s=1.0, pad_to=64)
        riders = q.offer(0, "a")
        self.assertEqual(len(riders), 1, "one real rider")
        self.assertEqual(q.width_for(riders), 64, "and sixty-four columns of weight-shared work")

    def test_without_padding_the_width_is_the_load(self):
        q = DepartureQueue(min_batch=1, max_interval_s=1.0)
        riders = q.offer(0, "a")
        self.assertEqual(q.width_for(riders), 1)

    def test_padding_below_the_minimum_is_refused(self):
        """It would pad a full load DOWN, which is a truncation wearing padding's name."""
        with self.assertRaises(ValueError):
            DepartureQueue(min_batch=32, max_interval_s=1.0, pad_to=16)

    def test_the_report_shows_what_the_padding_costs(self):
        """A queue that is mostly padding is a queue that should have a smaller fixed width, and
        that is only visible if the padding is counted."""
        q = DepartureQueue(min_batch=2, max_interval_s=1.0, pad_to=64)
        q.offer(0, "a")
        q.offer(0, "b")
        report = q.report()
        self.assertEqual(report["riders"], 2)
        self.assertEqual(report["padding"], 62)
        self.assertAlmostEqual(report["padding_pct"], 96.875)


if __name__ == "__main__":
    unittest.main(verbosity=2)
