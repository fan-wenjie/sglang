"""A rotation's positions crossing the wire without losing an axis.

A text stack's positions are one row per token. A multimodal one's are several -- mrope carries a
temporal, a height and a width row, so the tensor is (3, tokens) -- and flattening that into
(3 * tokens, 1) produces a frame whose positions are three times as long as its hidden states.

The far end then learns about it from a broadcast failure inside apply_rotary_emb, which names
neither the frame nor the axis that was lost. That is what arm C did on the very model this was
written for: Qwen3.8-27B is a conditional-generation wrapper, so its positions are two
dimensional, and 3 x 5 tokens arrived as 15.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch
from sglang.srt.afd.protocol import pack_positions, unpack_positions
from sglang.test.test_utils import CustomTestCase


class TestPositionsSurviveTheWire(CustomTestCase):
    def test_a_plain_sequence_comes_back_as_one(self):
        positions = torch.arange(7)
        self.assertTrue(torch.equal(unpack_positions(pack_positions(positions)), positions))

    def test_a_multi_axis_rotation_keeps_its_axes(self):
        positions = torch.arange(3 * 5).reshape(3, 5)
        packed = pack_positions(positions)
        self.assertEqual(tuple(packed.shape), (3, 5), "the wire carries it as it is")
        self.assertTrue(torch.equal(unpack_positions(packed), positions))

    def test_the_row_count_is_what_carries_the_meaning(self):
        """One row means a sequence; more means the rows are axes. Anything that flattened both
        would make a (1, T) and a (3, T/3) indistinguishable on arrival."""
        self.assertEqual(pack_positions(torch.arange(6)).shape[0], 1)
        self.assertEqual(pack_positions(torch.arange(6).reshape(3, 2)).shape[0], 3)

    def test_the_length_matches_the_tokens_it_belongs_to(self):
        """The property the failure violated: however many axes there are, the tokens are the
        columns, and a frame's positions have to line up with its hidden states row for row."""
        for tokens in (1, 5, 128):
            self.assertEqual(pack_positions(torch.arange(tokens)).shape[1], tokens)
            self.assertEqual(pack_positions(torch.arange(3 * tokens).reshape(3, tokens)).shape[1],
                             tokens)

    def test_an_impossible_shape_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            pack_positions(torch.arange(8).reshape(2, 2, 2))

    def test_it_is_integral_on_the_wire(self):
        """Sending positions as floats rounds past 2^24, which for a long context is inside the
        range a rotation is applied at."""
        self.assertEqual(pack_positions(torch.arange(4)).dtype, torch.int64)
        self.assertEqual(pack_positions(torch.arange(4).to(torch.int32)).dtype, torch.int64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
