"""Both ends must agree on the settings neither can apply alone, checked at the HELLO.

A flag can be given to one end and decide nothing on the other, saying so nowhere. That is history
rather than a worry: on the derived branch an entire throughput comparison ran with a flag varied
on the host, where the cut in use ignores it. Both arms were the same arm, the numbers agreed for
that reason, and it read as a result.

So each end folds the settings that must match into one number and they are compared before the
first frame. `afd/` moves the number and has no idea what is in it; an arm supplies the encoding
and the sentence. This branch installs no arm that needs one, so what is tested here is the
mechanism -- that a disagreement is refused and an agreement is not.
"""

import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.pool_client import PoolClient, PoolClosed
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase


class TestTheHostRefusesAMismatchedPool(CustomTestCase):
    """`_agree_on_arrangement` against a stand-in client: it reads two fields and raises."""

    def _client(self, theirs):
        return types.SimpleNamespace(
            _arrangement=theirs,
            address="the-pool:8999",
            _agree_on_arrangement=None,
        )

    def _run(self, mine, theirs, said="the flag differs"):
        """`mine` is passed in, as the caller passes it -- see `_agree_on_arrangement`'s note on
        why it is a parameter and not read from the global server args in there."""
        stand_in = self._client(theirs)
        # A published config, not a stand-in for one: the refusal path hands the whole record to
        # the arm registry, and a SimpleNamespace pretending to be it stops intercepting the
        # moment that read moves. `explain_arrangement` stays patched because what it returns is
        # the arm's, and this branch installs no arm.
        with get_context().override_server_args(), unittest.mock.patch(
            "sglang.srt.afd.arms.explain_arrangement", lambda a, m, t: said
        ):
            PoolClient._agree_on_arrangement(stand_in, mine)

    def test_matching_words_pass(self):
        self._run(3.0, 3.0)

    def test_a_different_word_is_refused_with_the_arm_s_sentence(self):
        with self.assertRaises(PoolClosed) as caught:
            self._run(3.0, 1.0)
        self.assertIn("configured differently", str(caught.exception))
        self.assertIn("the flag differs", str(caught.exception))

    def test_a_refusal_survives_an_arm_that_cannot_explain(self):
        """The mismatch is refused whether or not anything can put it into words."""
        with self.assertRaises(PoolClosed) as caught:
            self._run(3.0, 1.0, said="")
        self.assertIn("could say which setting differs", str(caught.exception))

    def test_a_pool_that_sends_no_word_is_refused_when_this_end_has_one(self):
        """An older build on the far end is a disagreement, not a default to pass through."""
        with self.assertRaises(PoolClosed) as caught:
            self._run(3.0, None)
        self.assertIn("did not send an arrangement word", str(caught.exception))

    def test_no_arm_either_side_is_fine(self):
        """The ordinary AFD case, where nothing is installed that needs agreeing on."""
        self._run(0.0, None)


if __name__ == "__main__":
    unittest.main()
