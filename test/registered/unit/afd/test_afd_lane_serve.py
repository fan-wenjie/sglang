"""The lane's serves against the frame handlers: same inputs, same state after.

The lane strips the wire ceremony -- frame parsing, `rows_of`, host-to-device staging --
and keeps the kernels. What must not drift is the arithmetic: a contraction served off the
lane and one served off a frame must read the same reading, and a landed advance must
leave the state and the ring exactly where `apply_advance` leaves them.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.afd.linear_history import HistoryCache
from sglang.srt.afd.protocol import OP_STATE_APPLY, OP_STATE_EARLY, Frame
from sglang.srt.afd_query_shift.early_contraction import apply_advance, contract_early
from sglang.srt.afd_query_shift.lane_serve import land_advance, lane_contract
from sglang.test.test_utils import CustomTestCase

CUDA = torch.cuda.is_available()
VALUE_HEADS, HEAD_K, HEAD_V, TAPS = 4, 8, 8, 4
CONV_WIDTH = 2 * VALUE_HEADS * HEAD_K + VALUE_HEADS * HEAD_V


def _service(rows):
    cache = HistoryCache(
        slots=4,
        layers=2,
        value_heads=VALUE_HEADS,
        head_k_dim=HEAD_K,
        head_v_dim=HEAD_V,
        device=torch.device("cuda"),
        conv_width=CONV_WIDTH,
        conv_taps=TAPS,
    )
    for r in rows:
        cache.slot_of(r)
    return SimpleNamespace(
        cache=cache,
        dims=(VALUE_HEADS, VALUE_HEADS, HEAD_K, HEAD_V),
        rows_of=lambda frame: rows,
        reads=0,
        updates=0,
        _parked=None,
    )


@unittest.skipUnless(CUDA, "the lane lands in device buffers")
class TestTheLaneServesTheSameArithmetic(CustomTestCase):
    def test_the_contraction_agrees(self):
        torch.manual_seed(3)
        a, b = _service([7]), _service([7])
        seed_state = torch.randn_like(a.cache.state)
        a.cache.state.copy_(seed_state)
        b.cache.state.copy_(seed_state)
        q = torch.randn(1, VALUE_HEADS * HEAD_K, device="cuda")
        off_frame = contract_early(a, Frame(7, 1, (q.cpu(),), OP_STATE_EARLY))[0]
        off_lane = lane_contract(b, 1, 7, q)
        torch.testing.assert_close(off_lane.cpu(), off_frame.cpu(), rtol=0, atol=0)

    def test_the_advance_lands_the_same_state_and_ring(self):
        torch.manual_seed(4)
        a, b = _service([7]), _service([7])
        seed_state = torch.randn_like(a.cache.state)
        seed_ring = torch.randn_like(a.cache.conv)
        for s in (a, b):
            s.cache.state.copy_(seed_state)
            s.cache.conv.copy_(seed_ring)
        k_w, v_w = VALUE_HEADS * HEAD_K, VALUE_HEADS * HEAD_V
        slab = torch.cat(
            [
                torch.randn(1, k_w, device="cuda"),
                torch.randn(1, v_w, device="cuda"),
                torch.rand(1, VALUE_HEADS, device="cuda"),
                torch.rand(1, VALUE_HEADS, device="cuda"),
                torch.randn(1, CONV_WIDTH, device="cuda"),
            ],
            dim=-1,
        )
        apply_advance(a, Frame(7, 1, (slab.cpu(),), OP_STATE_APPLY))
        land_advance(b, 1, 7, slab)
        torch.testing.assert_close(b.cache.state, a.cache.state, rtol=0, atol=0)
        torch.testing.assert_close(b.cache.conv, a.cache.conv, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
