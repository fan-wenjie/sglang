"""A request a weightless host cannot answer is refused at intake, not inside the forward.

The host holds no LM head. The pool computes each request's last-row logits and sends them, so
logits at other positions -- prompt logprobs -- have no answer here. The head shim already says
so, but it says it from inside the model forward, and the scheduler does not survive an exception
there: one such request took the whole server down, and every request in flight with it. Moving
the same "no" to intake makes it one client's error.

The boundary is the number of ROWS asked for. A start that leaves a single row is served, because
a single row is exactly what the pool sends -- refusing on the word `logprob` would turn away
requests this arrangement answers correctly today.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd import skeleton
from sglang.test.test_utils import CustomTestCase


def a_request(**kw):
    return SimpleNamespace(
        **{"return_logprob": True, "logprob_start_len": None, **kw}
    )


class TestOnlyAWeightlessHostRefuses(CustomTestCase):
    def test_an_ordinary_server_refuses_nothing(self):
        """The guard sits in the shared request path. A colocated server holds its own head and
        must be untouched by it."""
        with patch.object(skeleton, "skeleton_wanted", return_value=False):
            self.assertIsNone(
                skeleton.prompt_logprobs_refusal(a_request(logprob_start_len=0), 185)
            )


class TestWhatASkeletonHostRefuses(CustomTestCase):
    def setUp(self):
        self._p = patch.object(skeleton, "skeleton_wanted", return_value=True)
        self._p.start()
        self.addCleanup(self._p.stop)

    def test_prompt_logprobs_over_the_whole_prompt_are_refused(self):
        reason = skeleton.prompt_logprobs_refusal(a_request(logprob_start_len=0), 185)
        self.assertIsNotNone(reason)
        self.assertIn("holds", reason)

    def test_a_start_that_leaves_one_row_is_served(self):
        """185 tokens from 184 is one row, which is the row the pool sends."""
        self.assertIsNone(
            skeleton.prompt_logprobs_refusal(a_request(logprob_start_len=184), 185)
        )

    def test_output_logprobs_alone_are_served(self):
        """The ordinary shape -- `return_logprob` with no start -- asks for nothing extra."""
        self.assertIsNone(skeleton.prompt_logprobs_refusal(a_request(), 185))

    def test_a_request_that_asked_for_no_logprobs_is_served(self):
        self.assertIsNone(
            skeleton.prompt_logprobs_refusal(
                a_request(return_logprob=False, logprob_start_len=0), 185
            )
        )

    def test_a_negative_start_is_the_default_and_is_served(self):
        self.assertIsNone(
            skeleton.prompt_logprobs_refusal(a_request(logprob_start_len=-1), 185)
        )

    def test_the_reason_names_the_numbers_and_what_to_do(self):
        """A refusal a caller cannot act on sends them to the logs of the wrong process."""
        reason = skeleton.prompt_logprobs_refusal(a_request(logprob_start_len=3), 20)
        self.assertIn("17", reason)
        self.assertIn("colocated", reason)


if __name__ == "__main__":
    unittest.main()
