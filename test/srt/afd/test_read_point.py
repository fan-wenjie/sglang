"""The read plan resolves to the layers and sources it claims, or says why not.

These are the first two checks in the afd-early-q skill, and they come first because the failures
they catch are the silent ones: a shift that converts nothing costs nothing, and a cost of zero
reads as tolerance rather than as a hook that never installed.
"""

import unittest

from sglang.srt.afd_query_shift.early_q import EarlyQStash, EarlyQWiring, install
from sglang.srt.afd_query_shift.read_point import full_attention_layers, plan_read_points

# Qwen3.8-27B: 64 layers, full_attention_interval 4, so the softmax layers are the last of each
# group of four. Written out rather than imported so the test states what it assumes.
QWEN38_27B_LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"]
QWEN38_27B_LAYER_TYPES = QWEN38_27B_LAYER_TYPES * 16


class TestReadPoint(unittest.TestCase):
    def test_the_model_is_the_shape_the_arm_assumes(self):
        self.assertEqual(len(QWEN38_27B_LAYER_TYPES), 64)
        full = full_attention_layers(QWEN38_27B_LAYER_TYPES)
        self.assertEqual(len(full), 16)
        self.assertEqual(full[:4], (3, 7, 11, 15))
        self.assertTrue(all(i % 4 == 3 for i in full), "the softmax layer ends each group of four")

    def test_shift_zero_is_the_standard_wiring(self):
        plan = plan_read_points(0, 64)
        self.assertEqual(plan.points, ())
        self.assertEqual(plan.half_layers, 0)
        self.assertEqual(plan.offset_layers, 0.0)

    def test_one_layer_shift_is_half_a_layer_back(self):
        plan = plan_read_points(1, 64)
        self.assertEqual(plan.offset_layers, 0.5)
        self.assertEqual(plan.half_layers, 1)
        # layer 0 is exempt under either wiring, so 63 of 64 convert
        self.assertEqual(len(plan.points), 63)
        self.assertEqual(plan.points[0].layer, 1)
        self.assertEqual(plan.points[0].source, 0)
        self.assertEqual(plan.clamped, ())

    def test_the_four_hats_agree_at_every_setting(self):
        for n in range(1, 9):
            plan = plan_read_points(n, 64)
            self.assertEqual(plan.offset_layers, n - 0.5)
            self.assertEqual(plan.half_layers, 2 * n - 1)
            for point in plan.points:
                if not point.clamped:
                    self.assertEqual(point.source, point.layer - n)

    def test_group_shift_reads_the_previous_softmax_layer(self):
        """At N=4 each full-attention layer reads the last one's h. That is the hybrid shadow."""
        full = full_attention_layers(QWEN38_27B_LAYER_TYPES)
        plan = plan_read_points(4, 64, convertible=full)
        sources = {p.layer: p.source for p in plan.points}
        self.assertEqual(sources[7], 3)
        self.assertEqual(sources[11], 7)
        self.assertEqual(sources[63], 59)
        for source in sources.values():
            if source >= 0:
                self.assertIn(source, full, "the source of a group shift is itself a softmax layer")

    def test_the_bottom_of_the_stack_clamps_and_says_so(self):
        full = full_attention_layers(QWEN38_27B_LAYER_TYPES)
        plan = plan_read_points(4, 64, convertible=full)
        # layer 3 is the first softmax layer and has no h_{-1} to read
        self.assertEqual(plan.clamped, (3,))
        self.assertEqual(plan.as_record()["n_clamped"], 1)
        self.assertNotIn(3, plan.moved)

    def test_a_fractional_or_negative_shift_is_refused(self):
        with self.assertRaises(TypeError):
            plan_read_points(1.5, 64)
        with self.assertRaises(ValueError):
            plan_read_points(-1, 64)

    def test_a_shift_that_converts_nothing_raises(self):
        """The silent no-op. A one-layer stack has only the exempt layer 0."""
        with self.assertRaises(RuntimeError):
            install(1, n_layers=1)

    def test_record_carries_what_a_run_has_to_report(self):
        full = full_attention_layers(QWEN38_27B_LAYER_TYPES)
        record = plan_read_points(4, 64, convertible=full).as_record()
        for key in ("shift_layers", "offset_layers", "half_layers", "considered", "moved",
                    "clamped", "n_considered", "n_moved", "n_clamped", "sources"):
            self.assertIn(key, record)
        # every softmax layer is considered; the one that clamps did not actually move, and the
        # record has to distinguish the two or it reports coverage it does not have.
        self.assertEqual(record["n_considered"], len(full))
        self.assertEqual(record["n_moved"], len(full) - 1)
        self.assertEqual(record["n_clamped"], 1)
        self.assertNotIn("3", record["sources"], "a clamped layer has no source to record")


class TestStash(unittest.TestCase):
    def test_the_stash_refuses_a_tensor_from_the_wrong_stage(self):
        """Reading the layer OUTPUT gives x_(l+1), which is the standard wiring with extra steps."""
        stash = EarlyQStash()
        with self.assertRaises(ValueError):
            stash.put(1, 0, _t(), stage="layer_output")
        stash.put(1, 0, _t(), stage=EarlyQStash.STAGE)
        self.assertTrue(stash.has(1, 0))

    def test_slots_are_keyed_by_request_and_layer(self):
        """Keyed by layer alone, two requests at one layer would overwrite each other."""
        stash = EarlyQStash()
        a, b = _t(1.0), _t(2.0)
        stash.put(101, 7, a, EarlyQStash.STAGE)
        stash.put(202, 7, b, EarlyQStash.STAGE)
        self.assertEqual(stash.take(101, 7).item(), 1.0)
        self.assertEqual(stash.take(202, 7).item(), 2.0)

    def test_dropping_one_request_leaves_the_other(self):
        stash = EarlyQStash()
        stash.put(101, 7, _t(), EarlyQStash.STAGE)
        stash.put(202, 7, _t(), EarlyQStash.STAGE)
        stash.drop_request(101)
        self.assertFalse(stash.has(101, 7))
        self.assertTrue(stash.has(202, 7))

    def test_a_clamped_layer_falls_back_to_its_own_input(self):
        full = full_attention_layers(QWEN38_27B_LAYER_TYPES)
        wiring = EarlyQWiring(plan_read_points(4, 64, convertible=full))
        x_l = _t(9.0)
        self.assertIs(wiring.query_source(1, 3, x_l), x_l, "layer 3 clamps and keeps x_l")

    def test_only_the_named_sources_are_stashed(self):
        full = full_attention_layers(QWEN38_27B_LAYER_TYPES)
        wiring = EarlyQWiring(plan_read_points(4, 64, convertible=full))
        self.assertTrue(wiring.needs_stash(3))
        self.assertFalse(wiring.needs_stash(2), "a layer nobody reads is not held")


def _t(value: float = 0.0):
    import torch

    return torch.full((1,), value)


if __name__ == "__main__":
    unittest.main()


class TestCoverage(unittest.TestCase):
    """Which layers move is not which layers sweep a cache, and conflating them caps coverage.

    Guards a real capping: the installer used `full_attention_layers` for both questions, so a
    run asking for the full shift silently converted 16 of 64 layers and reported the cost under
    the fuller coverage's name. The study measured both -- +0.0181 bits per byte at 16/64 and
    +0.0211 at 63/64 -- so the two are distinguishable and the mix-up is not cosmetic.
    """

    def test_the_three_questions_have_three_answers(self):
        from sglang.srt.afd_query_shift.read_point import (
            convertible_layers,
            full_attention_layers,
            stateful_layers,
        )

        kinds = QWEN38_27B_LAYER_TYPES
        sweeps = full_attention_layers(kinds)
        moves = convertible_layers(kinds)
        stateful = stateful_layers(kinds)
        self.assertEqual(len(sweeps), 16, "only the softmax layers sweep a cache")
        self.assertEqual(len(moves), 64, "every layer has a query to move")
        self.assertEqual(len(stateful), 64, "a linear layer's recurrent state is state too")
        self.assertNotEqual(len(sweeps), len(moves))

    def test_full_coverage_is_63_of_64_at_one_layer_back(self):
        from sglang.srt.afd_query_shift.read_point import convertible_layers

        plan = plan_read_points(1, 64, convertible=convertible_layers(QWEN38_27B_LAYER_TYPES))
        # layer 0 is exempt, everything else moves
        self.assertEqual(len(plan.moved), 63)
        self.assertEqual(plan.clamped, ())

    def test_softmax_only_coverage_is_16_of_64(self):
        from sglang.srt.afd_query_shift.read_point import full_attention_layers

        plan = plan_read_points(1, 64, convertible=full_attention_layers(QWEN38_27B_LAYER_TYPES))
        self.assertEqual(len(plan.moved), 16)
