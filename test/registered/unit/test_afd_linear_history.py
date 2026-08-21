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
        """alpha lies in [0, 1] -- CLOSED at zero, which is not a rounding concession.

        Both bounds are CLOSED, and that is not a rounding concession. alpha =
        exp(-exp(A_log) * softplus(a + dt_bias)) underflows to exactly zero when A_log and the
        gate are both large -- the head forgets everything and keeps only this step -- and
        sigmoid saturates to exactly 1.0 in float32 at about |b| > 17. An earlier version
        asserted open bounds on UNSEEDED inputs and went red about once a run on legitimate
        values, which is a flaky test guarding nothing.

        What is worth pinning is that the formula is not INVERTED: a decay outside [0, 1] grows
        the state without bound, and one that ran backwards would remember the future.
        """
        g = torch.Generator().manual_seed(3)
        r = lambda *s: torch.randn(*s, generator=g) * 5
        alpha, beta = gates(r(4, VH), r(4, VH), r(VH) / 5, r(VH) / 5)
        self.assertTrue(bool(((alpha >= 0) & (alpha <= 1)).all()))
        self.assertTrue(bool(((beta >= 0) & (beta <= 1)).all()))

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


class TestTheQueryCoefficient(CustomTestCase):
    """One contraction of the state where there were two, because the state enters linearly.

    alpha h_q - alpha beta (k.q) h_k = alpha S [q - beta (k.q) k]. The bracket is computable on the
    weight side, so the history side reads once and never sees the key on the critical path.

    The failure this guards is a sign or a factor: the coefficient is a SUBTRACTION of the key's
    own correction, and getting it wrong produces a model that reads its history with a query
    nobody wrote -- fluent, and not the model.
    """

    def test_one_reading_equals_the_two_it_replaces(self):
        from sglang.srt.afd.linear_history import query_coefficient

        for seed in range(3):
            with self.subTest(seed=seed):
                step = a_step(seed)
                alpha, beta = gates(step["a"], step["b"], step["A_log"], step["dt_bias"])
                q, k = normalise(step["q"], step["k"], scale=KD**-0.5)
                q, k = expand_to_value_heads(q, VH), expand_to_value_heads(k, VH)
                h_q, h_k = read(step["state"], q, k)
                want = mix(h_q, h_k, v=step["v"], k=k, q=q, alpha=alpha, beta=beta)
                qt, s = query_coefficient(q, k, beta)
                hist = alpha.unsqueeze(-1) * torch.einsum("bhvk,bhk->bhv", step["state"], qt)
                got = hist + s.unsqueeze(-1) * step["v"].float()
                torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-6)

    def test_the_coefficient_is_the_query_minus_the_keys_correction(self):
        from sglang.srt.afd.linear_history import query_coefficient

        q = torch.zeros(1, VH, KD); q[..., 0] = 1.0
        k = torch.zeros(1, VH, KD); k[..., 0] = 1.0
        qt, s = query_coefficient(q, k, torch.full((1, VH), 0.25))
        # k.q is 1 here, so the coefficient is q - 0.25 k
        self.assertAlmostEqual(qt[0, 0, 0].item(), 0.75, places=6)
        self.assertAlmostEqual(s[0, 0].item(), 0.25, places=6)

    def test_a_key_orthogonal_to_the_query_leaves_it_alone(self):
        """No overlap, no correction -- and a sign error would show here as a change."""
        from sglang.srt.afd.linear_history import query_coefficient

        q = torch.zeros(1, VH, KD); q[..., 0] = 1.0
        k = torch.zeros(1, VH, KD); k[..., 1] = 1.0
        qt, s = query_coefficient(q, k, torch.full((1, VH), 0.5))
        torch.testing.assert_close(qt, q)
        self.assertEqual(s.abs().max().item(), 0.0)


class TestAPrefillChunkEqualsTheSameTokensDecodedOneAtATime(CustomTestCase):
    """The test that was missing three times, in three places.

    In DECODE a row is a request and a token at once, so code written for one works for the other
    and nothing says which was meant. In PREFILL a row is a token and a chunk's tokens are
    sequentially dependent. Treating the chunk as a batch runs every token against the same
    starting state, never advances it, and produces a model with no memory of its own prompt --
    fluent, and wrong from the first generated token onward.
    """

    def a_chunk(self, n=5, seed=1):
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: torch.randn(*s, generator=g)
        return dict(q=r(n, VH, KD), k=r(n, VH, KD), v=r(n, VH, VD),
                    alpha=torch.rand(n, VH, generator=g) * 0.5 + 0.5,
                    beta=torch.rand(n, VH, generator=g))

    def test_the_scan_matches_the_same_tokens_one_at_a_time(self):
        from sglang.srt.afd.linear_history import prefill_scan

        chunk = self.a_chunk()
        start = torch.randn(VH, VD, KD) * 0.1

        got, got_state = prefill_scan(start.clone(), **chunk)

        state, wanted = start.clone(), []
        for t in range(chunk["q"].shape[0]):
            one = {key: val[t : t + 1] for key, val in chunk.items()}
            h_q, h_k = read(state.unsqueeze(0), one["q"], one["k"])
            wanted.append(mix(h_q, h_k, v=one["v"], k=one["k"], q=one["q"],
                              alpha=one["alpha"], beta=one["beta"])[0])
            state = update(state.unsqueeze(0), h_k, v=one["v"], k=one["k"],
                           alpha=one["alpha"], beta=one["beta"])[0]
        torch.testing.assert_close(got, torch.stack(wanted, dim=0))
        torch.testing.assert_close(got_state, state)

    def test_treating_the_chunk_as_a_batch_gives_a_different_answer(self):
        """The bug this guards, made explicit. If these ever agree the test is worthless."""
        from sglang.srt.afd.linear_history import prefill_scan

        chunk = self.a_chunk()
        start = torch.randn(VH, VD, KD) * 0.1
        scanned, _ = prefill_scan(start.clone(), **chunk)
        n = chunk["q"].shape[0]
        as_batch, _ = read(start.unsqueeze(0).expand(n, -1, -1, -1), chunk["q"], chunk["k"])
        batched = mix(as_batch, _, v=chunk["v"], k=chunk["k"], q=chunk["q"],
                      alpha=chunk["alpha"], beta=chunk["beta"])
        self.assertFalse(torch.allclose(scanned, batched))

    def test_a_chunk_of_one_is_a_decode_step(self):
        from sglang.srt.afd.linear_history import prefill_scan

        chunk = self.a_chunk(n=1)
        start = torch.randn(VH, VD, KD) * 0.1
        got, _ = prefill_scan(start.clone(), **chunk)
        h_q, h_k = read(start.unsqueeze(0), chunk["q"], chunk["k"])
        want = mix(h_q, h_k, v=chunk["v"], k=chunk["k"], q=chunk["q"],
                   alpha=chunk["alpha"], beta=chunk["beta"])
        torch.testing.assert_close(got, want)


class TestAChunksConvolutionEqualsTheSameStepsOneAtATime(CustomTestCase):
    """Same difference, in the other stateful piece of the layer."""

    def test_the_chunk_matches_the_ring_advanced_token_by_token(self):
        from sglang.srt.afd.linear_history import prefill_convolve

        torch.manual_seed(2)
        C, K, n = 6, 4, 5
        weight = torch.randn(C, K)
        ring = torch.randn(C, K)
        x = torch.randn(n, C)

        got, got_ring = prefill_convolve(ring.clone(), x, weight)

        held, wanted = ring.clone(), []
        for t in range(n):
            window = torch.cat([held[:, 1:], x[t].unsqueeze(-1)], dim=-1)
            wanted.append(torch.nn.functional.silu((window * weight).sum(-1)))
            held = window
        torch.testing.assert_close(got, torch.stack(wanted, dim=0), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(got_ring, held, rtol=1e-5, atol=1e-6)

    def test_a_ring_of_the_wrong_width_is_refused(self):
        """K-1 was the first version, and it does not raise -- it broadcasts."""
        from sglang.srt.afd.linear_history import prefill_convolve

        with self.assertRaises(ValueError) as caught:
            prefill_convolve(torch.zeros(6, 3), torch.zeros(2, 6), torch.zeros(6, 4))
        self.assertIn("it broadcasts", str(caught.exception))
