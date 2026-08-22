"""An op the pool does not serve is refused BY NAME, not left to fall through.

`answer_directly` returning False does not mean "unserved". It means "not answered here", and the
router then hands the frame to `offer`, where it is queued as a feed-forward and answered with
something nobody asked for. So a reserved op has to be refused above every branch that can return
False, and that placement is what this file pins.

OP_LINEAR is the reserved one. It ran a linear-attention layer on the pool with that request's
recurrent state held there; the pool had a complete handler and nothing ever sent one. It was
removed rather than finished, because a pool holding per-request state stops being stateless and
can no longer be released between one request's own calls -- and that release is what the whole
arrangement rests on. The number stays reserved so nothing reuses it and an older peer's frame
gets an explanation instead of an answer.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.pool_server import Departure
from sglang.srt.afd.protocol import OP_FFN, OP_LINEAR, Frame
from sglang.test.test_utils import CustomTestCase


class TestAReservedOpIsRefusedRatherThanQueued(CustomTestCase):
    def _departure(self):
        return Departure(lambda batch, layer: batch, 1, 0.005, "cpu")

    def test_a_linear_frame_is_refused_and_says_why(self):
        """No socket is needed: the refusal happens before anything is written."""
        departure = self._departure()
        frame = Frame(1, 0, (torch.zeros(1, 4),) * 6, OP_LINEAR)
        with self.assertRaises(ConnectionError) as caught:
            departure.answer_directly(frame, None)
        message = str(caught.exception)
        self.assertIn("RESERVED", message)
        self.assertIn("LAYER", message, "the refusal names what to send instead")

    def test_it_is_refused_on_a_pool_with_nothing_attached(self):
        """The placement, not the message. A pool with no cache, no attention and no span takes
        the earliest `return False` in the function -- and every op that reaches that line is
        queued as a feed-forward. If the refusal were below it, a LINEAR frame from an older host
        would be answered with a feed-forward of its convolved projection, which is a tensor of
        the right shape and the wrong meaning, and nothing would raise on either end."""
        departure = self._departure()
        self.assertIsNone(departure.cache)
        self.assertIsNone(departure.span)
        self.assertIsNone(departure.attention)
        with self.assertRaises(ConnectionError):
            departure.answer_directly(Frame(1, 0, (torch.zeros(1, 4),), OP_LINEAR), None)

    def test_an_op_this_pool_does_serve_still_falls_through_to_the_queue(self):
        """The negative half: the refusal must not have swallowed the ordinary path."""
        departure = self._departure()
        self.assertFalse(departure.answer_directly(Frame(1, 0, (torch.zeros(1, 4),), OP_FFN), None))


if __name__ == "__main__":
    unittest.main()
