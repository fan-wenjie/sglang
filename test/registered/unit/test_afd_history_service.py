"""The host answering the pool's calls for the history, and what it refuses.

Every case guards a failure with no output symptom. Both states ARE the whole history compressed,
so a row served from the wrong slot, or an update applied with three tensors where four belong,
produces fluent text conditioned on something that is not this request's past.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.history_service import HistoryService
from sglang.srt.afd.linear_history import HistoryCache
from sglang.srt.afd.protocol import OP_STATE_READ, OP_STATE_UPDATE, Frame
from sglang.test.test_utils import CustomTestCase

VH, D = 3, 4


def a_service(ids=(11,)):
    cache = HistoryCache(slots=4, layers=2, value_heads=VH, head_k_dim=D, head_v_dim=D,
                         conv_width=8, conv_taps=3, device=torch.device("cpu"))
    return cache, HistoryService(cache, rows_of=lambda frame: list(ids))


class TestTheReadIsOneContractionAndWritesNothing(CustomTestCase):
    """The critical path's whole job. A write here would be the deferred update run early.

    It would not change this step's answer -- the reading is of the old state either way -- and it
    would double-apply when the deferred update arrived.
    """

    def test_it_returns_the_state_contracted_against_the_coefficient(self):
        cache, service = a_service()
        slot = cache.slot_of(11)
        cache.state[0, slot] = torch.randn(VH, D, D)
        q = torch.randn(1, VH * D)
        out = service(Frame(1, 0, (q,), OP_STATE_READ))[0]
        want = torch.einsum("hvk,hk->hv", cache.state[0, slot], q.reshape(VH, D))
        torch.testing.assert_close(out.reshape(VH, D), want)

    def test_it_leaves_the_state_alone(self):
        cache, service = a_service()
        cache.state[0, cache.slot_of(11)] = torch.randn(VH, D, D)
        before = cache.state.clone()
        service(Frame(1, 0, (torch.randn(1, VH * D),), OP_STATE_READ))
        torch.testing.assert_close(cache.state, before)

    def test_the_decay_is_not_applied_here(self):
        """alpha is the pool's, and a value that crosses the wire to be multiplied there and back
        is a value that should not have gone. The reading is raw."""
        cache, service = a_service()
        cache.state[0, cache.slot_of(11)] = torch.ones(VH, D, D)
        q = torch.zeros(1, VH * D)
        q[0, 0] = 2.0
        out = service(Frame(1, 0, (q,), OP_STATE_READ))[0]
        self.assertAlmostEqual(out[0, 0].item(), 2.0, places=5)


class TestTheUpdateIsDeferredAndAnswersNothing(CustomTestCase):
    """The state only has to be right by the NEXT step, which is what keeps v off the wire's path."""

    def test_it_returns_none(self):
        cache, service = a_service()
        cache.slot_of(11)
        frame = Frame(1, 0, (torch.randn(1, VH * D), torch.randn(1, VH * D),
                             torch.rand(1, VH), torch.rand(1, VH)), OP_STATE_UPDATE)
        self.assertIsNone(service(frame))

    def test_it_advances_the_state(self):
        cache, service = a_service()
        slot = cache.slot_of(11)
        cache.state[0, slot] = torch.randn(VH, D, D)
        before = cache.state[0, slot].clone()
        service(Frame(1, 0, (torch.randn(1, VH * D), torch.randn(1, VH * D),
                             torch.full((1, VH), 0.5), torch.full((1, VH), 0.5)),
                      OP_STATE_UPDATE))
        self.assertFalse(torch.allclose(cache.state[0, slot], before))

    def test_a_short_update_is_refused(self):
        """Applying three tensors where four belong advances the state by something else."""
        cache, service = a_service()
        cache.slot_of(11)
        with self.assertRaises(RuntimeError) as caught:
            service(Frame(1, 0, (torch.randn(1, VH * D), torch.randn(1, VH * D),
                                 torch.rand(1, VH)), OP_STATE_UPDATE))
        self.assertIn("two gates", str(caught.exception))


class TestRowIdsAreNotOptional(CustomTestCase):
    def test_a_count_mismatch_is_refused_rather_than_broadcast(self):
        cache, service = a_service(ids=(11,))
        with self.assertRaises(RuntimeError) as caught:
            service(Frame(1, 0, (torch.randn(2, VH * D),), OP_STATE_READ))
        self.assertIn("whose history it reads", str(caught.exception))

    def test_two_requests_read_their_own_slots(self):
        cache, service = a_service(ids=(11, 22))
        cache.state[0, cache.slot_of(11)].fill_(1.0)
        cache.state[0, cache.slot_of(22)].fill_(2.0)
        q = torch.zeros(2, VH * D)
        q[:, 0] = 1.0
        out = service(Frame(1, 0, (q,), OP_STATE_READ))[0]
        self.assertAlmostEqual(out[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(out[1, 0].item(), 2.0, places=5)


class TestAnOpThisSideDoesNotHold(CustomTestCase):
    def test_it_names_the_disagreement(self):
        _, service = a_service()
        with self.assertRaises(RuntimeError) as caught:
            service(Frame(1, 0, (torch.zeros(1, 1),), 0))
        self.assertIn("disagree about what lives here", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
