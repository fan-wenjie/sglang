"""The backend must refuse a parallelism that splits one request's sequence.

Tier 1 selects rows by a global top-m over the sequence and tier 2 fires
against the kept maximum for a head; both are wrong on every rank once the
sequence is split, and the output stays fluent, so nothing downstream notices.
The refusal is the only thing that does.
"""

import unittest
from unittest import mock

from sglang.srt.layers.attention import vestigekv_mla_backend as B


def _parallel(*, dcp_size=1, attn_cp_size=1):
    return mock.Mock(dcp_size=dcp_size, attn_cp_size=attn_cp_size)


class TestRefusesShardedSequence(unittest.TestCase):
    def test_dcp_size_above_one_is_refused(self):
        with mock.patch.object(B, "get_parallel", lambda: _parallel(dcp_size=2)):
            with self.assertRaises(ValueError) as cm:
                B._refuse_sharded_sequence()
        self.assertIn("--dcp-size 2", str(cm.exception))

    def test_attn_cp_size_above_one_is_refused(self):
        with mock.patch.object(B, "get_parallel", lambda: _parallel(attn_cp_size=4)):
            with self.assertRaises(ValueError) as cm:
                B._refuse_sharded_sequence()
        self.assertIn("--attn-cp-size 4", str(cm.exception))

    def test_both_are_named_together(self):
        # One flag in the message would send someone to change that one and
        # meet the same refusal again.
        with mock.patch.object(
            B, "get_parallel", lambda: _parallel(dcp_size=2, attn_cp_size=2)
        ):
            with self.assertRaises(ValueError) as cm:
                B._refuse_sharded_sequence()
        self.assertIn("--dcp-size 2", str(cm.exception))
        self.assertIn("--attn-cp-size 2", str(cm.exception))

    def test_the_message_says_what_breaks_and_where_the_design_is(self):
        with mock.patch.object(B, "get_parallel", lambda: _parallel(dcp_size=2)):
            with self.assertRaises(ValueError) as cm:
                B._refuse_sharded_sequence()
        msg = str(cm.exception)
        self.assertIn("global top-m", msg)
        self.assertIn("docs/context-parallel.md", msg)

    def test_the_unsharded_case_passes(self):
        # The guard runs in every backend construction, so a false positive
        # here refuses every ordinary launch.
        with mock.patch.object(B, "get_parallel", lambda: _parallel()):
            B._refuse_sharded_sequence()


if __name__ == "__main__":
    unittest.main(verbosity=2)
