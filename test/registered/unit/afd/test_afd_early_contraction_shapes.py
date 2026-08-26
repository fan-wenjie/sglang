"""The push handlers run, against real shapes, with the widths the model actually has.

The pool cooks and assembles; this side contracts-and-answers, then applies-and-says-nothing.
These hold the two handlers to their contracts: the reading comes back for exactly the rows
that asked, a chunk is refused on both ops, the apply advances the state and the ring by
exactly what was sent, and an out-of-date arity is refused as the older protocol it is.
"""

import torch

from sglang.srt.afd.linear_history import HistoryCache
from sglang.srt.afd.protocol import OP_STATE_APPLY, OP_STATE_EARLY, Frame
from sglang.srt.afd_query_shift.early_contraction import apply_advance, contract_early
from sglang.test.test_utils import CustomTestCase

# Qwen3.8-27B's linear attention, which is the only family this arm is exercised on.
KEY_HEADS, VALUE_HEADS, HEAD_K, HEAD_V = 16, 48, 128, 128
KEY_DIM = KEY_HEADS * HEAD_K
CONV_WIDTH = 2 * KEY_DIM + VALUE_HEADS * HEAD_V
TAPS = 4


class _Service:
    """The parts of the history service the handlers reach for, and nothing else."""

    def __init__(self, rows):
        self.cache = HistoryCache(
            slots=4,
            layers=2,
            value_heads=VALUE_HEADS,
            head_k_dim=HEAD_K,
            head_v_dim=HEAD_V,
            device=torch.device("cpu"),
            conv_width=CONV_WIDTH,
            conv_taps=TAPS,
        )
        self.dims = (KEY_HEADS, VALUE_HEADS, HEAD_K, HEAD_V)
        self._parked = None
        self.reads = 0
        self.updates = 0
        self._rows = rows
        # A state per slot, distinct. On a fresh cache every contraction is zero and two rows
        # agreeing says nothing about whether they were kept apart.
        for slot in range(self.cache.state.shape[1]):
            self.cache.state[:, slot] = torch.randn_like(self.cache.state[:, slot]) * (
                slot + 1
            )
        self.cache.conv.copy_(torch.randn_like(self.cache.conv) * 0.1)

    def rows_of(self, frame):
        return self._rows


def _early_frame(rows, layer=1):
    return Frame(7, layer, (torch.randn(rows, VALUE_HEADS * HEAD_K),), OP_STATE_EARLY)


def _apply_parts(rows):
    return (
        torch.randn(rows, VALUE_HEADS * HEAD_K),
        torch.randn(rows, VALUE_HEADS * HEAD_V, dtype=torch.bfloat16),
        torch.rand(rows, VALUE_HEADS),
        torch.rand(rows, VALUE_HEADS),
        torch.randn(rows, CONV_WIDTH, dtype=torch.bfloat16),
    )


def _apply_frame(rows, layer=1, parts=None):
    # packed as the pool packs it: one float32 tensor, [k | v | alpha | beta | column]
    parts = parts if parts is not None else _apply_parts(rows)
    packed = torch.cat([t.reshape(rows, -1).float() for t in parts], dim=-1)
    return Frame(7, layer, (packed,), OP_STATE_APPLY)


class TestTheEarlyContractionAnswers(CustomTestCase):
    def test_one_row_comes_back(self):
        service = _Service([3])
        (reading,) = contract_early(service, _early_frame(1))
        self.assertEqual(tuple(reading.shape), (1, VALUE_HEADS * HEAD_V))

    def test_several_requests_each_get_their_own(self):
        service = _Service([3, 0, 2])
        (reading,) = contract_early(service, _early_frame(3))
        self.assertEqual(reading.shape[0], 3)
        rows = reading.reshape(3, VALUE_HEADS, HEAD_V)
        # distinct states per slot, so identical readings would mean rows were conflated
        self.assertFalse(torch.allclose(rows[0], rows[1]))

    def test_a_chunk_is_refused(self):
        service = _Service([3, 3])
        with self.assertRaisesRegex(RuntimeError, "more than one row"):
            contract_early(service, _early_frame(2))

    def test_the_older_two_tensor_protocol_is_refused(self):
        service = _Service([3])
        stale = Frame(
            7,
            1,
            (
                torch.randn(1, VALUE_HEADS * HEAD_K),
                torch.randn(1, VALUE_HEADS * HEAD_K),
            ),
            OP_STATE_EARLY,
        )
        with self.assertRaisesRegex(RuntimeError, "older protocol"):
            contract_early(service, stale)


class TestTheApplyAdvances(CustomTestCase):
    def test_state_and_ring_advance_and_nothing_is_answered(self):
        service = _Service([3])
        parts = _apply_parts(1)
        frame = _apply_frame(1, parts=parts)
        slot = service.cache.slot_of(3)
        state_before = service.cache.state[1][slot].clone()
        ring_before = service.cache.conv[1][slot].clone()
        answered = apply_advance(service, frame)
        self.assertIsNone(answered)
        self.assertEqual(service.updates, 1)
        self.assertFalse(torch.allclose(service.cache.state[1][slot], state_before))
        after = service.cache.conv[1][slot]
        torch.testing.assert_close(after[..., :-1], ring_before[..., 1:])
        torch.testing.assert_close(after[..., -1], parts[4][0].to(after.dtype))

    def test_a_chunk_is_refused(self):
        service = _Service([3, 3])
        with self.assertRaisesRegex(RuntimeError, "more than one row"):
            apply_advance(service, _apply_frame(2))

    def test_a_parked_advance_is_refused(self):
        service = _Service([3])
        service._parked = (0, None, None, None, None, None)
        with self.assertRaisesRegex(RuntimeError, "still parked"):
            apply_advance(service, _apply_frame(1))


if __name__ == "__main__":
    import unittest

    unittest.main()
