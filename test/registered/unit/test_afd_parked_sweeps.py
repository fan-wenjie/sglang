"""The two-part buffer that lets a sweep and its append arrive in either order.

The bug this guards is the one the pool's own docstring described and did not fix: a sweep whose
range is not yet in the cache used to RAISE, which is safe only because both frames travel one TCP
connection and are answered in arrival order. Sharding the client across links -- which the
measurements recommend -- gives two connections no order between them, and correct traffic would
start failing.

Parking makes order irrelevant. These cases pin the properties that has to have: a sweep parks
rather than failing, an append releases exactly the sweeps its length satisfies and no others, a
timeout removes the entry so a late append cannot answer the same frame twice, and releasing runs
callbacks outside the lock so a resume that writes to a socket cannot deadlock against a park.
"""

import threading
import time
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.parked_sweeps import ParkedSweeps
from sglang.test.test_utils import CustomTestCase


class TestParkedSweeps(CustomTestCase):
    def _buffer(self, timeout_s=10.0):
        return ParkedSweeps(timeout_s=timeout_s)

    def test_a_sweep_ahead_of_its_append_parks_instead_of_failing(self):
        buffer = self._buffer()
        buffer.park(request_id=1, layer=0, length=5, resume=lambda: None)
        self.assertEqual(buffer.held(), 1)
        self.assertEqual(buffer.release(request_id=1, layer=0, held=4), [])
        self.assertEqual(buffer.held(), 1, "four positions do not satisfy a sweep over five")

    def test_an_append_releases_exactly_what_it_satisfies(self):
        """A release must not free a sweep covering a longer range than has arrived.

        Freeing one would sweep a history with a hole in it, which is the failure the whole
        arrangement is built to make impossible: the model attends to a past missing a position
        and its output stays fluent.
        """
        buffer = self._buffer()
        for length in (3, 5, 9):
            buffer.park(request_id=1, layer=0, length=length, resume=lambda: None)
        released = buffer.release(request_id=1, layer=0, held=5)
        self.assertEqual(sorted(e.length for e in released), [3, 5])
        self.assertEqual(buffer.held(), 1)
        self.assertEqual([e.length for e in buffer.release(request_id=1, layer=0, held=9)], [9])

    def test_a_release_does_not_reach_another_layer_or_request(self):
        """The key is (request, layer); an append to one must not free a sweep on another."""
        buffer = self._buffer()
        buffer.park(request_id=1, layer=0, length=2, resume=lambda: None)
        buffer.park(request_id=1, layer=1, length=2, resume=lambda: None)
        buffer.park(request_id=2, layer=0, length=2, resume=lambda: None)
        self.assertEqual(len(buffer.release(request_id=1, layer=0, held=2)), 1)
        self.assertEqual(buffer.held(), 2)

    def test_release_returns_callbacks_rather_than_running_them(self):
        """Resumes run outside the lock, so a socket write cannot deadlock against a park.

        The caller runs them. If this object ran them itself it would hold its lock across the
        write, and the thread being written to may be parking on the same lock.
        """
        buffer = self._buffer()
        ran = []
        buffer.park(request_id=1, layer=0, length=1, resume=lambda: ran.append("resumed"))
        released = buffer.release(request_id=1, layer=0, held=1)
        self.assertEqual(ran, [], "release must not have called the resume itself")
        for entry in released:
            entry.resume()
        self.assertEqual(ran, ["resumed"])

    def test_an_expired_sweep_leaves_the_buffer(self):
        """Otherwise a late append releases it too and one frame is answered twice.

        The clock is passed in rather than waited on. A version of this slept for the timeout and
        asked whether it had elapsed, which is a test whose result depends on how loaded the
        machine running it is; `now` exists so the expiry boundary can be stated instead.
        """
        buffer = self._buffer(timeout_s=10.0)
        buffer.park(request_id=1, layer=0, length=5, resume=lambda: None)
        self.assertEqual(buffer.expired(now=time.monotonic() + 9.0), [],
                         "nine seconds into a ten second timeout is not expired")
        expired = buffer.expired(now=time.monotonic() + 11.0)
        self.assertEqual(len(expired), 1)
        self.assertEqual(buffer.held(), 0)
        self.assertEqual(buffer.release(request_id=1, layer=0, held=99), [],
                         "an expired sweep must not also be releasable")

    def test_a_departed_request_takes_its_parked_sweeps_with_it(self):
        buffer = self._buffer()
        buffer.park(request_id=7, layer=0, length=4, resume=lambda: None)
        buffer.park(request_id=7, layer=3, length=4, resume=lambda: None)
        buffer.park(request_id=8, layer=0, length=4, resume=lambda: None)
        self.assertEqual(len(buffer.drop(request_id=7)), 2)
        self.assertEqual(buffer.held(), 1)

    def test_a_timeout_of_zero_is_refused(self):
        """It would expire every sweep before its append could arrive."""
        with self.assertRaises(ValueError):
            ParkedSweeps(timeout_s=0.0)

    def test_the_peak_is_recorded_because_it_is_what_says_the_buffer_is_needed(self):
        """A peak of one means the traffic was ordered anyway and parking bought nothing."""
        buffer = self._buffer()
        for layer in range(4):
            buffer.park(request_id=1, layer=layer, length=2, resume=lambda: None)
        for layer in range(4):
            buffer.release(request_id=1, layer=layer, held=2)
        report = buffer.report()
        self.assertEqual(report["peak_held"], 4)
        self.assertEqual(report["released"], 4)
        self.assertEqual(report["held_now"], 0)

    def test_parking_and_releasing_from_many_threads_loses_nothing(self):
        """The buffer is touched by every connection thread and the timer thread at once.

        Not a probabilistic stress test for a race it cannot reproduce: this asserts the
        bookkeeping totals, which are exact regardless of interleaving. A lost park or a double
        release changes them.
        """
        buffer = self._buffer()
        threads = []
        for request_id in range(8):
            threads.append(threading.Thread(
                target=lambda r=request_id: [
                    buffer.park(request_id=r, layer=l, length=1, resume=lambda: None)
                    for l in range(8)
                ]))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(buffer.held(), 64)

        released = []
        threads = []
        for request_id in range(8):
            threads.append(threading.Thread(
                target=lambda r=request_id: released.extend(
                    entry for l in range(8)
                    for entry in buffer.release(request_id=r, layer=l, held=1))))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(released), 64)
        self.assertEqual(buffer.held(), 0)
        self.assertEqual(buffer.report()["released"], 64)


if __name__ == "__main__":
    unittest.main()
