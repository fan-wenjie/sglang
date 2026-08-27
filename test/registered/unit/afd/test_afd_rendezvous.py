"""The buffer where a layer's two halves wait for each other.

Every failure this class can have is silent. A slot that never completes does not raise -- one
request simply stops advancing while the others carry on. A slot completed twice advances a layer
twice. A slot keyed loosely merges one request's history into another's token. None of those
produce an exception, and none produce obviously wrong text, so each has a case here.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.afd.rendezvous import DepartureQueue, Rendezvous
from sglang.test.test_utils import CustomTestCase


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TestEitherHalfMayArriveFirst(CustomTestCase):
    def test_local_then_remote(self):
        r = Rendezvous()
        self.assertIsNone(r.put_local((1, 3), "k", "v"))
        slot = r.put_remote((1, 3), "o", "lse")
        self.assertIsNotNone(slot)
        self.assertEqual((slot.local, slot.remote), (("k", "v"), ("o", "lse")))

    def test_remote_then_local(self):
        """The symmetric case, and the reason this is not a queue: the database answering before
        the stage has finished its own key and value is normal, not early."""
        r = Rendezvous()
        self.assertIsNone(r.put_remote((1, 3), "o", "lse"))
        slot = r.put_local((1, 3), "k", "v")
        self.assertIsNotNone(slot)
        self.assertEqual((slot.local, slot.remote), (("k", "v"), ("o", "lse")))

    def test_one_half_alone_does_not_leave(self):
        r = Rendezvous()
        r.put_local((1, 3), "k", "v")
        self.assertEqual(r.outstanding(), 1)

    def test_two_keys_do_not_complete_each_other(self):
        """Loose keying merges one request's swept history into another request's token, and the
        output stays a plausible attention."""
        r = Rendezvous()
        self.assertIsNone(r.put_local((1, 3), "k1", "v1"))
        self.assertIsNone(r.put_remote((2, 3), "o2", "lse2"))
        self.assertIsNone(r.put_remote((1, 4), "o3", "lse3"))
        self.assertEqual(r.outstanding(), 3)


class TestAStuckSlotIsFindable(CustomTestCase):
    def test_stale_names_only_the_slots_past_the_deadline(self):
        clock = Clock()
        r = Rendezvous(now=clock)
        r.put_local((1, 0), "k", "v")
        clock.t = 5.0
        r.put_local((2, 0), "k", "v")
        clock.t = 8.0
        stale = r.stale(deadline_s=5.0)
        self.assertEqual([s.key for s in stale], [(1, 0)])

    def test_a_completed_slot_is_never_stale(self):
        clock = Clock()
        r = Rendezvous(now=clock)
        r.put_local((1, 0), "k", "v")
        r.put_remote((1, 0), "o", "lse")
        clock.t = 100.0
        self.assertEqual(r.stale(deadline_s=1.0), [])


class TestReissuingDoesNotCompleteTwice(CustomTestCase):
    def test_a_reply_from_a_superseded_attempt_is_dropped(self):
        """The bug this prevents: a query times out, is reissued, and then the original lands.
        Without attempt numbers both replies complete the slot and the layer advances twice.
        """
        r = Rendezvous()
        r.put_local((1, 0), "k", "v")
        attempt = r.reissue((1, 0))
        self.assertEqual(attempt, 1)
        self.assertIsNone(r.put_remote((1, 0), "old", "lse", attempt=0))
        self.assertEqual(r.outstanding(), 1, "the late original must not complete it")
        slot = r.put_remote((1, 0), "new", "lse", attempt=1)
        self.assertIsNotNone(slot)
        self.assertEqual(slot.remote, ("new", "lse"))
        self.assertEqual(r.report()["dropped_stale_replies"], 1)

    def test_a_reply_to_a_slot_that_already_left_is_dropped(self):
        r = Rendezvous()
        r.put_local((1, 0), "k", "v")
        r.reissue((1, 0))
        r.put_remote((1, 0), "new", "lse", attempt=1)
        self.assertEqual(r.outstanding(), 0)
        self.assertIsNone(r.put_remote((1, 0), "later", "lse", attempt=1))
        self.assertEqual(r.outstanding(), 0, "a reply must not reopen a finished slot")

    def test_reissue_restarts_the_clock(self):
        """Otherwise a reissued slot is stale the moment it is reissued and loops forever."""
        clock = Clock()
        r = Rendezvous(now=clock)
        r.put_local((1, 0), "k", "v")
        clock.t = 10.0
        r.reissue((1, 0))
        self.assertEqual(r.stale(deadline_s=5.0), [])


class TestWhoWasLate(CustomTestCase):
    def test_the_side_that_waits_alone_is_counted(self):
        """The cheapest instrument here: whichever half sits in the buffer is the one being
        waited for, so counting them says which side is the critical path without a benchmark.
        """
        r = Rendezvous()
        for i in range(3):
            r.put_local((i, 0), "k", "v")  # local first: the remote is late
            r.put_remote((i, 0), "o", "lse")
        r.put_remote((9, 0), "o", "lse")  # remote first: the local is late
        r.put_local((9, 0), "k", "v")
        report = r.report()
        self.assertEqual(report["waited_for_local"], 3)
        self.assertEqual(report["waited_for_remote"], 1)
        self.assertEqual(report["completed"], 4)
        self.assertAlmostEqual(report["remote_is_late_pct"], 75.0)

    def test_dropping_a_request_frees_only_its_own_slots(self):
        r = Rendezvous()
        for layer in (0, 1, 2):
            r.put_local((1, layer), "k", "v")
        r.put_local((2, 0), "k", "v")
        self.assertEqual(r.drop(1), 3)
        self.assertEqual(r.outstanding(), 1)


class TestTheSecondLayerBatches(CustomTestCase):
    """Completed pairs group by layer before the weight-shared work runs on them."""

    def test_a_full_load_departs_at_once(self):
        q = DepartureQueue(min_batch=3, max_interval_s=1.0)
        self.assertIsNone(q.offer(5, "a"))
        self.assertIsNone(q.offer(5, "b"))
        self.assertEqual(q.offer(5, "c"), ["a", "b", "c"])
        self.assertEqual(q.waiting(), 0)

    def test_two_layers_do_not_ride_together(self):
        """They share no weight, so riding together buys them nothing and costs the smaller one
        the larger one's wait."""
        q = DepartureQueue(min_batch=2, max_interval_s=1.0)
        self.assertIsNone(q.offer(5, "a"))
        self.assertIsNone(q.offer(6, "b"))
        self.assertEqual(q.waiting(), 2)
        self.assertEqual(q.offer(5, "c"), ["a", "c"])

    def test_a_short_load_departs_when_the_interval_runs_out(self):
        """The last requests of a draining workload never reach min_batch. Without this they wait
        for a partner that is not coming, and the symptom is a hang rather than a failure.
        """
        clock = Clock()
        q = DepartureQueue(min_batch=8, max_interval_s=0.5, now=clock)
        q.offer(3, "lonely")
        self.assertEqual(q.due(), [])
        clock.t = 0.6
        self.assertEqual(q.due(), [(3, ["lonely"])])
        self.assertEqual(q.waiting(), 0)

    def test_the_interval_runs_from_the_last_departure_not_from_arrival(self):
        """The rule as specified, and the two differ. Timed from arrival, a slow trickle departs
        every item alone the moment its own wait expires -- which is the unbatched arrangement
        wearing this one's name. Timed from the last departure, a trickle still collects whatever
        accumulated during the interval."""
        clock = Clock()
        q = DepartureQueue(min_batch=8, max_interval_s=1.0, now=clock)
        clock.t = 0.9
        q.offer(3, "early")
        clock.t = 1.05  # 1.05 since the last departure, 0.15 since arrival
        due = q.due()
        self.assertEqual(
            due,
            [(3, ["early"])],
            "an item that arrived recently still rides if the INTERVAL expired",
        )

        clock.t = 1.10
        q.offer(3, "next")
        clock.t = 1.9  # 0.85 since the departure above
        self.assertEqual(
            q.due(), [], "and no second departure inside the same interval"
        )

    def test_an_empty_queue_does_not_depart(self):
        clock = Clock()
        q = DepartureQueue(min_batch=2, max_interval_s=0.1, now=clock)
        clock.t = 5.0
        self.assertEqual(q.due(), [])
        self.assertEqual(q.report()["departures"], 0)

    def test_a_queue_with_no_timeout_is_refused(self):
        with self.assertRaises(ValueError):
            DepartureQueue(min_batch=4, max_interval_s=0)

    def test_the_report_separates_full_loads_from_timed_out_ones(self):
        """The diagnosis: departures that were full mean the batching is working, departures that
        timed out mean the queue is starved and min_batch is aspirational."""
        clock = Clock()
        q = DepartureQueue(min_batch=2, max_interval_s=1.0, now=clock)
        q.offer(1, "a")
        q.offer(1, "b")  # full
        clock.t = 2.0
        q.offer(1, "c")
        q.due()  # timed out
        report = q.report()
        self.assertEqual(report["departed_full"], 1)
        self.assertEqual(report["departed_on_time"], 1)
        self.assertEqual(report["riders"], 3)
        self.assertAlmostEqual(report["mean_riders"], 1.5)


class TestTheTwoLayersTogether(CustomTestCase):
    def test_a_half_reaches_the_first_layer_and_not_the_second(self):
        r, q = Rendezvous(), DepartureQueue(min_batch=1, max_interval_s=1.0)
        slot = r.put_local((1, 7), "k", "v")
        self.assertIsNone(slot)
        self.assertEqual(r.outstanding(), 1)
        self.assertEqual(q.waiting(), 0)

    def test_a_completed_pair_moves_from_the_first_layer_to_the_second(self):
        r = Rendezvous()
        q = DepartureQueue(min_batch=2, max_interval_s=1.0)
        r.put_local((1, 7), "k", "v")
        slot = r.put_remote((1, 7), "o", "lse")
        self.assertIsNotNone(slot)
        self.assertEqual(r.outstanding(), 0)
        self.assertIsNone(q.offer(slot.key[1], slot))
        self.assertEqual(
            q.waiting(), 1, "whole, and now waiting for company rather than a half"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
