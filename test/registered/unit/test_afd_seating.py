"""Who boards the next span, when a prefill remainder rides with the decode walk-ins.

The rule's value is in the case that no aggregate number shows. Bumping the HEAD instead of the
tail gives identical throughput, identical bus occupancy and identical riders histograms, and
starves whoever is unlucky -- so the ordering is what these cases are for.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.seating import load, seat
from sglang.test.test_utils import CustomTestCase


class TestTheWorkedExample(CustomTestCase):
    """Capacity 16, nine waiting, a remainder of eleven."""

    def test_the_counts(self):
        self.assertEqual(seat(16, 9, 11), (11, 5, 4))

    def test_the_riders(self):
        aboard, waiting = load(16, [f"d{i}" for i in range(1, 10)],
                               [f"p{i}" for i in range(1, 12)])
        self.assertEqual(len(aboard), 16)
        self.assertEqual(aboard[:11], [f"p{i}" for i in range(1, 12)])
        self.assertEqual(aboard[11:], ["d1", "d2", "d3", "d4", "d5"])
        self.assertEqual(waiting, ["d6", "d7", "d8", "d9"])


class TestBumpingFromTheTailIsWhatPreventsStarvation(CustomTestCase):
    """A bumped rider is the OLDEST waiting next time, so it boards then.

    Bumping the head instead would look the same in every aggregate -- same occupancy, same
    throughput, same histogram -- and would leave the same riders behind forever.
    """

    def test_a_bumped_rider_boards_the_next_bus(self):
        queue = [f"d{i}" for i in range(1, 10)]
        _, queue = load(16, queue, ["p"] * 11)
        self.assertEqual(queue[0], "d6")
        aboard, _ = load(16, queue, ["p"] * 11)
        self.assertIn("d6", aboard)

    def test_the_original_queue_drains_while_seats_remain(self):
        """Ten buses, a remainder of eleven every time, and everyone who was waiting gets on."""
        queue = [f"d{i}" for i in range(1, 10)]
        seen = set()
        for bus in range(10):
            aboard, queue = load(16, queue, [f"p{bus}"] * 11)
            seen.update(r for r in aboard if r.startswith("d"))
            queue = queue + [f"n{bus}"]          # a new walk-in arrives each round
        self.assertTrue({f"d{i}" for i in range(1, 10)} <= seen)
        # only the rider that arrived after the last departure is still waiting
        self.assertEqual(queue, ["n9"])


class TestAFullRemainderStarvesTheQueueWithoutTheTimeout(CustomTestCase):
    """The ordering rule alone is not enough, and this is the case that says so.

    A remainder the size of the bus leaves no seats. A stream of them never moves the queue, and
    every aggregate looks healthy the whole time -- the buses are full and departing. `overdue` is
    what breaks it, and it is the same argument max_wait_s already makes on the other side.
    """

    def test_without_the_timeout_nobody_moves(self):
        queue = [f"d{i}" for i in range(1, 10)]
        for _ in range(20):
            _, queue = load(16, queue, ["p"] * 16)
        self.assertEqual(len(queue), 9, "the queue moved, so this case no longer guards anything")

    def test_the_timeout_gives_the_bus_to_the_walk_ins(self):
        queue = [f"d{i}" for i in range(1, 10)]
        aboard, queue = load(16, queue, ["p"] * 16, overdue=True)
        self.assertEqual(aboard, [f"d{i}" for i in range(1, 10)])
        self.assertEqual(queue, [])

    def test_overdue_yields_the_whole_bus_not_the_leftover_seats(self):
        """Taking what is left is exactly the case that starves: with a full remainder there is
        nothing left, so a rule that only shrank the remainder would change nothing."""
        self.assertEqual(seat(16, 9, 16, overdue=True), (0, 9, 0))
        self.assertEqual(seat(16, 20, 11, overdue=True), (0, 16, 4))

    def test_the_head_is_never_the_one_bumped(self):
        aboard, waiting = load(16, [f"d{i}" for i in range(1, 10)], ["p"] * 11)
        self.assertIn("d1", aboard)
        self.assertNotIn("d1", waiting)


class TestARemainderIsAtomic(CustomTestCase):
    """Its tokens are sequentially dependent; splitting them across two buses interleaves
    somebody else's state updates between them, and the forward pass stays fluent."""

    def test_a_remainder_larger_than_the_bus_is_refused_not_truncated(self):
        with self.assertRaises(ValueError) as caught:
            seat(16, 0, 17)
        self.assertIn("split a chunk", str(caught.exception))

    def test_a_remainder_that_exactly_fills_the_bus_leaves_no_seats(self):
        self.assertEqual(seat(16, 9, 16), (16, 0, 9))

    def test_no_remainder_is_the_ordinary_decode_bus(self):
        self.assertEqual(seat(16, 9), (0, 9, 0))
        self.assertEqual(seat(16, 20), (0, 16, 4))


class TestTheEdgesRefuseRatherThanGuess(CustomTestCase):
    def test_a_bus_with_no_seats(self):
        with self.assertRaises(ValueError):
            seat(0, 5, 0)

    def test_negative_riders(self):
        with self.assertRaises(ValueError):
            seat(16, -1, 0)


if __name__ == "__main__":
    unittest.main()
