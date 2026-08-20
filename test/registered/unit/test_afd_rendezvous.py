"""The buffer where a layer's two halves wait for each other.

Every failure this class can have is silent. A slot that never completes does not raise -- one
request simply stops advancing while the others carry on. A slot completed twice advances a layer
twice. A slot keyed loosely merges one request's history into another's token. None of those
produce an exception, and none produce obviously wrong text, so each has a case here.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.afd.rendezvous import Rendezvous
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
        Without attempt numbers both replies complete the slot and the layer advances twice."""
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
        waited for, so counting them says which side is the critical path without a benchmark."""
        r = Rendezvous()
        for i in range(3):
            r.put_local((i, 0), "k", "v")       # local first: the remote is late
            r.put_remote((i, 0), "o", "lse")
        r.put_remote((9, 0), "o", "lse")        # remote first: the local is late
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
