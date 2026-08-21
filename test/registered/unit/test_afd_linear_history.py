"""The split of a linear layer into a read and a mix, pinned against the kernel it replaces.

`benchmark/afd/gdn_split.py` is the measurement that says the split is exact; it needs a GPU. This
pins the same identity on the CPU against a reference recurrence written out in the open, so the
next edit to these five functions has something to fail against without one.

The failure this guards is the arrangement's usual one. Every mistake available here -- expanding
the heads by the wrong factor, reading the NEW state instead of the old, applying the gates in the
wrong order, dropping the L2 normalisation -- produces a model that generates fluent text from a
history that is not the request's.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.linear_history import (
    expand_to_value_heads,
    gates,
    mix,
    normalise,
    read,
    update,
)
from sglang.test.test_utils import CustomTestCase

ROWS, KH, VH, KD, VD = 3, 2, 4, 8, 8


def a_step(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return dict(
        q=r(ROWS, KH, KD), k=r(ROWS, KH, KD), v=r(ROWS, VH, VD),
        a=r(ROWS, VH), b=r(ROWS, VH), A_log=r(VH), dt_bias=r(VH),
        state=r(ROWS, VH, VD, KD) * 0.1,
    )


def reference(step, scale):
    """The gated delta rule, written out: update the state, then read it with the query.

    Deliberately in the FUSED order -- the state is advanced first and the output is the new
    state contracted with the query -- because that is the order the split has to reproduce
    without ever forming the new state before the read.
    """
    alpha, beta = gates(step["a"], step["b"], step["A_log"], step["dt_bias"])
    q, k = normalise(step["q"], step["k"], scale=scale)
    q, k = expand_to_value_heads(q, VH), expand_to_value_heads(k, VH)
    S = step["state"]
    al, be = alpha.unsqueeze(-1), beta.unsqueeze(-1)
    written = be * (step["v"].float() - al * torch.einsum("bhvk,bhk->bhv", S, k))
    S_new = al.unsqueeze(-1) * S + written.unsqueeze(-1) * k.unsqueeze(-2)
    return torch.einsum("bhvk,bhk->bhv", S_new, q), S_new


def split(step, scale):
    alpha, beta = gates(step["a"], step["b"], step["A_log"], step["dt_bias"])
    q, k = normalise(step["q"], step["k"], scale=scale)
    q, k = expand_to_value_heads(q, VH), expand_to_value_heads(k, VH)
    h_q, h_k = read(step["state"], q, k)
    out = mix(h_q, h_k, v=step["v"], k=k, q=q, alpha=alpha, beta=beta)
    S_new = update(step["state"], h_k, v=step["v"], k=k, alpha=alpha, beta=beta)
    return out, S_new, h_q, h_k


class TestTheSplitIsTheSameArithmetic(CustomTestCase):
    """Two readings of the OLD state and some scalars reproduce a fused update-then-read.

    This is the whole claim. If it fails, a linear layer cannot wear a softmax layer's interface
    and the Early-Q window goes back to existing at one layer in four.
    """

    def test_the_output_matches(self):
        for seed in range(3):
            with self.subTest(seed=seed):
                step = a_step(seed)
                want, _ = reference(step, KD**-0.5)
                got, _, _, _ = split(step, KD**-0.5)
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    def test_the_state_update_matches(self):
        step = a_step()
        _, want = reference(step, KD**-0.5)
        _, got, _, _ = split(step, KD**-0.5)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    def test_the_query_reading_needs_no_key(self):
        """The point of the split: h_q is a function of the state and the query alone.

        It is what makes the reading sendable one message ahead of the key and value, which is the
        window. A version that folded anything of k into h_q would still be correct and would have
        nothing to send early.
        """
        step = a_step()
        _, _, h_q, _ = split(step, KD**-0.5)
        moved = dict(step)
        moved["k"] = torch.randn(ROWS, KH, KD, generator=torch.Generator().manual_seed(99))
        _, _, h_q_other, _ = split(moved, KD**-0.5)
        torch.testing.assert_close(h_q, h_q_other)

    def test_both_readings_are_of_the_state_before_this_step(self):
        """Reading the NEW state would be the fused kernel with extra steps, and correct."""
        step = a_step()
        _, S_new, h_q, _ = split(step, KD**-0.5)
        q, _ = normalise(step["q"], step["k"], scale=KD**-0.5)
        after = torch.einsum("bhvk,bhk->bhv", S_new, expand_to_value_heads(q, VH))
        self.assertFalse(torch.allclose(h_q, after))


class TestTheHeadExpansion(CustomTestCase):
    """The state is per value head and the query is per key head; one has to be expanded.

    Getting the factor wrong reads a neighbouring head's history. There is no output symptom --
    the text stays fluent and is conditioned on the wrong thing.
    """

    def test_it_repeats_each_key_head_across_its_value_heads(self):
        x = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
        out = expand_to_value_heads(x, 4)
        self.assertEqual(out.shape, (1, 4, 2))
        self.assertEqual(out[0, :, 0].tolist(), [1.0, 1.0, 2.0, 2.0])

    def test_a_factor_that_is_not_an_integer_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            expand_to_value_heads(torch.zeros(1, 3, 2), 4)
        self.assertIn("do not divide", str(caught.exception))

    def test_an_equal_count_is_the_identity(self):
        x = torch.randn(2, 4, 2)
        torch.testing.assert_close(expand_to_value_heads(x, 4), x)


class TestTheReadRefusesAMismatch(CustomTestCase):
    """A broadcast that papered over disagreeing shapes would read the wrong head's history."""

    def test_disagreeing_heads(self):
        with self.assertRaises(ValueError):
            read(torch.zeros(2, 4, VD, KD), torch.zeros(2, 2, KD), torch.zeros(2, 2, KD))

    def test_disagreeing_rows(self):
        with self.assertRaises(ValueError):
            read(torch.zeros(2, VH, VD, KD), torch.zeros(3, VH, KD), torch.zeros(3, VH, KD))

    def test_a_key_that_is_not_the_query_s_shape(self):
        with self.assertRaises(ValueError):
            read(torch.zeros(2, VH, VD, KD), torch.zeros(2, VH, KD), torch.zeros(2, VH, KD + 1))


class TestNormalisationIsNotDecoration(CustomTestCase):
    """`use_qk_l2norm_in_kernel=True` changes what the state is read with.

    Dropping it is a different model that still produces text, which is why it lives in one place
    that both ends of the split call rather than at each site that reads.
    """

    def test_the_query_is_scaled_and_the_key_is_not(self):
        q, k = normalise(torch.randn(2, KH, KD), torch.randn(2, KH, KD), scale=0.5)
        torch.testing.assert_close(q.norm(dim=-1), torch.full((2, KH), 0.5))
        torch.testing.assert_close(k.norm(dim=-1), torch.ones(2, KH))

    def test_reading_unnormalised_gives_a_different_answer(self):
        step = a_step()
        q = expand_to_value_heads(step["q"].float(), VH)
        k = expand_to_value_heads(step["k"].float(), VH)
        raw, _ = read(step["state"], q, k)
        qn, kn = normalise(step["q"], step["k"], scale=KD**-0.5)
        normed, _ = read(step["state"], expand_to_value_heads(qn, VH),
                         expand_to_value_heads(kn, VH))
        self.assertFalse(torch.allclose(raw, normed))


class TestTheGatesAreTheLayersOwn(CustomTestCase):
    """A_log and dt_bias are per-layer parameters, so they belong with the layer's weights.

    Computed on the weight-holding side means the history-holding side never has to be told about
    a model, or kept in sync with one.
    """

    def test_the_decay_is_in_the_unit_interval(self):
        alpha, beta = gates(torch.randn(4, VH) * 5, torch.randn(4, VH) * 5,
                            torch.randn(VH), torch.randn(VH))
        self.assertTrue(bool(((alpha > 0) & (alpha <= 1)).all()))
        self.assertTrue(bool(((beta > 0) & (beta < 1)).all()))

    def test_the_bias_moves_the_decay(self):
        args = (torch.zeros(2, VH), torch.zeros(2, VH), torch.zeros(VH))
        near_one, _ = gates(*args, dt_bias=torch.full((VH,), -8.0))
        near_zero, _ = gates(*args, dt_bias=torch.full((VH,), 8.0))
        self.assertTrue(bool((near_one > near_zero).all()))


if __name__ == "__main__":
    unittest.main()


class TestTheHostSideCache(CustomTestCase):
    """The host holds both states, reads them, advances them. The pool holds none of it.

    Every case here guards a slot-table failure, and slot-table failures in a linear layer have no
    output symptom: both states ARE the whole history compressed, so a request served from the
    wrong slot produces fluent text conditioned on somebody else's prompt.
    """

    def a_cache(self, slots=2, layers=2):
        from sglang.srt.afd.linear_history import HistoryCache

        return HistoryCache(slots=slots, layers=layers, value_heads=VH, head_k_dim=KD,
                            head_v_dim=VD, conv_width=8, conv_taps=3,
                            device=torch.device("cpu"))

    def test_it_reproduces_the_reference_step(self):
        """The same identity as above, through the object that holds the state."""
        cache = self.a_cache()
        step = a_step()
        cache.state[0, 0] = step["state"][0]
        alpha, beta = gates(step["a"], step["b"], step["A_log"], step["dt_bias"])
        q, k = normalise(step["q"], step["k"], scale=KD**-0.5)
        q, k = expand_to_value_heads(q, VH), expand_to_value_heads(k, VH)
        h_q, h_k = cache.read_and_update(
            [7], 0, q=q[:1], k=k[:1], v=step["v"][:1], alpha=alpha[:1], beta=beta[:1])
        want_q, want_k = read(step["state"][:1], q[:1], k[:1])
        torch.testing.assert_close(h_q, want_q)
        torch.testing.assert_close(h_k, want_k)

    def test_the_reading_is_of_the_state_before_the_update(self):
        """Reading after advancing is the fused kernel with extra steps -- correct, and no window."""
        cache = self.a_cache()
        step = a_step()
        cache.state[0, 0] = step["state"][0]
        before = cache.state[0, 0].clone()
        alpha, beta = gates(step["a"][:1], step["b"][:1], step["A_log"], step["dt_bias"])
        q, k = normalise(step["q"][:1], step["k"][:1], scale=KD**-0.5)
        q, k = expand_to_value_heads(q, VH), expand_to_value_heads(k, VH)
        h_q, _ = cache.read_and_update([7], 0, q=q, k=k, v=step["v"][:1],
                                       alpha=alpha, beta=beta)
        torch.testing.assert_close(h_q, torch.einsum("hvk,hk->hv", before, q[0]).unsqueeze(0))
        self.assertFalse(torch.allclose(cache.state[0, 0], before))

    def test_a_slot_is_stable_across_layers_and_steps(self):
        cache = self.a_cache()
        first = cache.slot_of(11)
        self.assertEqual(cache.slot_of(11), first)
        self.assertNotEqual(cache.slot_of(22), first)

    def test_release_zeroes_both_states(self):
        cache = self.a_cache()
        slot = cache.slot_of(11)
        cache.state[:, slot].fill_(3.0)
        cache.conv[:, slot].fill_(5.0)
        self.assertTrue(cache.release(11))
        self.assertEqual(cache.state[:, slot].abs().sum().item(), 0.0)
        self.assertEqual(cache.conv[:, slot].abs().sum().item(), 0.0)

    def test_a_reused_slot_starts_from_nothing(self):
        """The failure with no symptom: the next occupant inheriting a stranger's memory."""
        cache = self.a_cache(slots=1)
        slot = cache.slot_of(11)
        cache.state[:, slot].fill_(3.0)
        cache.release(11)
        self.assertEqual(cache.slot_of(22), slot)
        self.assertEqual(cache.state[:, slot].abs().sum().item(), 0.0)

    def test_a_full_table_refuses_rather_than_evicts(self):
        cache = self.a_cache(slots=1)
        cache.slot_of(11)
        with self.assertRaises(RuntimeError) as caught:
            cache.slot_of(22)
        self.assertIn("cannot be evicted", str(caught.exception))

    def test_row_ids_that_do_not_match_the_rows_are_refused(self):
        cache = self.a_cache()
        with self.assertRaises(RuntimeError) as caught:
            cache.read_and_update([7], 0, q=torch.zeros(2, VH, KD), k=torch.zeros(2, VH, KD),
                                  v=torch.zeros(2, VH, VD), alpha=torch.zeros(2, VH),
                                  beta=torch.zeros(2, VH))
        self.assertIn("whose history it reads", str(caught.exception))

    def test_two_requests_do_not_read_each_others_history(self):
        cache = self.a_cache()
        cache.state[0, cache.slot_of(11)].fill_(1.0)
        cache.state[0, cache.slot_of(22)].fill_(2.0)
        q = torch.zeros(2, VH, KD); q[:, :, 0] = 1.0
        h_q, _ = cache.read_and_update([11, 22], 0, q=q, k=torch.zeros(2, VH, KD),
                                       v=torch.zeros(2, VH, VD),
                                       alpha=torch.ones(2, VH), beta=torch.zeros(2, VH))
        self.assertEqual(h_q[0, 0, 0].item(), 1.0)
        self.assertEqual(h_q[1, 0, 0].item(), 2.0)
