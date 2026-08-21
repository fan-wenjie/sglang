"""The one-pass state read, and the fallback that stands in for it without a GPU.

The kernel itself is checked against `linear_history.read`/`.update` on a GPU by
`benchmark/afd/state_passes.py`; these cases pin the contract that holds on either path, and the
two mistakes available here that produce fluent output from the wrong history: writing the state
somewhere other than the caller's buffer, and advancing a row against the wrong slot.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.linear_history import read, update
from sglang.srt.afd.split_read_kernel import read_two_and_update
from sglang.test.test_utils import CustomTestCase

SLOTS, VH, D = 4, 3, 8


def a_batch(rows=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return dict(q=r(rows, VH, D), k=r(rows, VH, D), v=r(rows, VH, D),
                alpha=torch.rand(rows, VH, generator=g) * 0.5 + 0.5,
                beta=torch.rand(rows, VH, generator=g))


class TestItIsTheSameArithmeticAsTheSplit(CustomTestCase):
    """One pass has to produce exactly what two reads and an update produce.

    If it does not, the fast path and the reference disagree and only one of them is the model --
    with no way to tell from the output which.
    """

    def test_both_readings_and_the_new_state_match(self):
        state = torch.randn(SLOTS, VH, D, D) * 0.1
        kept = state.clone()
        batch = a_batch()
        slots = torch.tensor([2, 0], dtype=torch.int32)
        h_q, h_k = read_two_and_update(state, slots, **batch)
        held = kept.index_select(0, slots.long())
        want_q, want_k = read(held, batch["q"], batch["k"])
        want_state = update(held, want_k, v=batch["v"], k=batch["k"],
                            alpha=batch["alpha"], beta=batch["beta"])
        torch.testing.assert_close(h_q, want_q)
        torch.testing.assert_close(h_k, want_k)
        torch.testing.assert_close(state.index_select(0, slots.long()), want_state)


class TestTheStateIsWrittenInPlace(CustomTestCase):
    """The caller's buffer IS the history. A copy written back elsewhere is a lost update.

    It would not raise and it would not change one step's output -- the reading is of the old
    state either way. It would show up thousands of tokens later as a model that never remembers.
    """

    def test_the_callers_tensor_changes(self):
        state = torch.randn(SLOTS, VH, D, D) * 0.1
        before = state.clone()
        read_two_and_update(state, torch.tensor([1], dtype=torch.int32), **a_batch(rows=1))
        self.assertFalse(torch.allclose(state[1], before[1]))

    def test_only_the_named_slots_change(self):
        state = torch.randn(SLOTS, VH, D, D) * 0.1
        before = state.clone()
        read_two_and_update(state, torch.tensor([1], dtype=torch.int32), **a_batch(rows=1))
        for other in (0, 2, 3):
            torch.testing.assert_close(state[other], before[other])


class TestTheSlotsAreNotOptional(CustomTestCase):
    """A decode batch carries one token from each of several requests.

    A row advanced against the wrong slot folds one request's history into another, and there is
    nothing in the output to say so -- the state IS the history, compressed.
    """

    def test_two_rows_read_their_own_slots(self):
        state = torch.zeros(SLOTS, VH, D, D)
        state[0].fill_(1.0)
        state[3].fill_(2.0)
        batch = a_batch(rows=2)
        batch["q"] = torch.zeros(2, VH, D)
        batch["q"][:, :, 0] = 1.0
        batch["beta"] = torch.zeros(2, VH)
        h_q, _ = read_two_and_update(state, torch.tensor([0, 3], dtype=torch.int32), **batch)
        self.assertAlmostEqual(h_q[0, 0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(h_q[1, 0, 0].item(), 2.0, places=5)

    def test_a_state_whose_shape_does_not_match_is_refused(self):
        if not torch.cuda.is_available():
            self.skipTest("the shape check guards the kernel path, which needs a device")
        state = torch.randn(SLOTS, VH + 1, D, D, device="cuda") * 0.1
        with self.assertRaises(ValueError):
            read_two_and_update(state, torch.tensor([0], dtype=torch.int32, device="cuda"),
                                **{k: v.cuda() for k, v in a_batch(rows=1).items()})


if __name__ == "__main__":
    unittest.main()
