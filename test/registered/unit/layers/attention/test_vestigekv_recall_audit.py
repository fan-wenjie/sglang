"""The audit reports the recall the certificate delivers, not the one it targets.

Derived property, not a mirror: VKSTATS observes fire count, fetch and fence,
all of which are cost, so a certificate whose zp still fits its calibration
queries while the live query has drifted looks unchanged in every number the
serving path collects. What the audit has to get right is the comparison
itself -- the same any-head rule the scan fires on, and an empty
above-threshold set scored as met rather than missed, since a mean over easy
steps would otherwise sag for the wrong reason.
"""

import unittest

import torch

from sglang.srt.layers.attention.vestigekv.recall_audit import RecallAudit, step_recall
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

A, D, H = 16, 4, 2


def _case():
    """Row i scores i for head 0; the threshold admits rows 12..15."""
    arch = torch.zeros(A, D)
    arch[:, 0] = torch.arange(A, dtype=torch.float32)
    q = torch.zeros(H, D)
    q[0, 0] = 1.0
    thr = torch.tensor([11.5, 1e9])  # head 1 admits nothing
    return arch, q, thr


class TestStepRecall(CustomTestCase):
    def test_firing_every_true_row_reads_one(self):
        arch, q, thr = _case()
        fired = torch.zeros(A, dtype=torch.bool)
        fired[12:] = True
        rec, n_true, n_fired = step_recall(arch, q, thr, fired)
        self.assertEqual(int(n_true), 4)
        self.assertAlmostEqual(float(rec), 1.0, places=5)

    def test_missing_half_the_true_rows_reads_one_half(self):
        arch, q, thr = _case()
        fired = torch.zeros(A, dtype=torch.bool)
        fired[14:] = True
        rec, _, _ = step_recall(arch, q, thr, fired)
        self.assertAlmostEqual(float(rec), 0.5, places=5)

    def test_firing_extra_rows_does_not_raise_recall_above_one(self):
        arch, q, thr = _case()
        rec, _, n_fired = step_recall(arch, q, thr, torch.ones(A, dtype=torch.bool))
        self.assertAlmostEqual(float(rec), 1.0, places=5)
        self.assertEqual(int(n_fired), A)

    def test_nothing_above_threshold_counts_as_met_not_missed(self):
        arch, q, _ = _case()
        rec, n_true, _ = step_recall(
            arch, q, torch.tensor([1e9, 1e9]), torch.zeros(A, dtype=torch.bool)
        )
        self.assertEqual(int(n_true), 0)
        self.assertAlmostEqual(float(rec), 1.0, places=5)

    def test_a_row_is_true_when_any_head_wants_it(self):
        # head 1 alone admits the top rows; the any-head rule must see them
        arch, q, _ = _case()
        q2 = q.clone()
        q2[1, 0] = 1.0
        rec, n_true, _ = step_recall(
            arch, q2, torch.tensor([1e9, 11.5]), torch.zeros(A, dtype=torch.bool)
        )
        self.assertEqual(int(n_true), 4)
        self.assertAlmostEqual(float(rec), 0.0, places=5)


class TestAudit(CustomTestCase):
    """Per-step work is device-only so it captures; the host reads once."""

    def test_a_sequence_no_longer_than_the_budget_is_not_recorded(self):
        a = RecallAudit(target=0.9, every=4)
        a.observe(arch=torch.zeros(0, D), q=torch.zeros(1, D), thr=torch.zeros(1),
                  fired=torch.zeros(0, dtype=torch.bool), lid=3)
        self.assertIsNone(a.ring)

    def test_observe_touches_no_host_state(self):
        # the ring and the counter are the only things observe may write;
        # anything host-side would run once at capture and never again
        arch, q, thr = _case()
        a = RecallAudit(target=0.9, every=4)
        fired = torch.zeros(A, dtype=torch.bool); fired[12:] = True
        a.observe(arch=arch, q=q, thr=thr, fired=fired, lid=7)
        self.assertEqual(int(a.count), 1)
        self.assertEqual(a.reported, 0)

    def test_dump_reports_only_steps_since_the_last_report(self):
        arch, q, thr = _case()
        a = RecallAudit(target=0.9, every=8)
        good = torch.zeros(A, dtype=torch.bool); good[12:] = True
        bad = torch.zeros(A, dtype=torch.bool); bad[14:] = True
        with self.assertLogs("sglang.srt.layers.attention.vestigekv.recall_audit", "INFO") as cm:
            a.observe(arch=arch, q=q, thr=thr, fired=good, lid=7)
            a.observe(arch=arch, q=q, thr=thr, fired=bad, lid=7)
            a.dump(lid=7)
            a.observe(arch=arch, q=q, thr=thr, fired=good, lid=7)
            a.dump(lid=7)
        self.assertEqual(len(cm.output), 2)
        self.assertIn("n=2", cm.output[0])
        self.assertIn("below_target=0.500", cm.output[0])
        self.assertIn("base=2 n=1", cm.output[1])
        self.assertIn("below_target=0.000", cm.output[1])

    def test_the_ring_keeps_the_newest_steps_when_it_wraps(self):
        arch, q, thr = _case()
        a = RecallAudit(target=0.9, every=2)
        bad = torch.zeros(A, dtype=torch.bool); bad[14:] = True
        good = torch.zeros(A, dtype=torch.bool); good[12:] = True
        for f in (bad, bad, good, good, good):
            a.observe(arch=arch, q=q, thr=thr, fired=f, lid=3)
        with self.assertLogs("sglang.srt.layers.attention.vestigekv.recall_audit", "INFO") as cm:
            a.dump(lid=3)
        # five observed, ring of two: only the last two survive, both good
        self.assertIn("n=2", cm.output[0])
        self.assertIn("achieved_mean=1.0000", cm.output[0])


if __name__ == "__main__":
    unittest.main()
