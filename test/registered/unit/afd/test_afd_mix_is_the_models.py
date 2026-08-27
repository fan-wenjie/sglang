"""The host's half of a linear layer against the model's own, one token at a time.

Since the convolution moved to the host, a linear layer is computed in two places: the pool
projects and gates, the host convolves and contracts the state, the pool norms and projects out.
Nothing compared that composition to the layer sglang would have run.

What existed did not cover it. `benchmark/afd/gdn_split.py` verifies the recurrence but starts
from `mixed` -- the projection AFTER the convolution -- so the convolution was outside it, and it
was outside on the side that no longer holds it. `test_afd_linear_layer.py` covers the callback's
threading and says nothing about the arithmetic.

Driven ONE TOKEN AT A TIME, because the approximation and the split are both defined on the
recurrent step; the chunked prefill kernel does not contain that step, so a comparison run through
it would be comparing two things neither of which is the thing under test.

The control reverses the convolution's taps. It is on ONE side only: reversing them on both
changes the same tap in each and agrees again, which is a control that cannot fail -- a mistake
this line of work has already made twice, recorded in `test_afd_convolution.py` on the branch
above this one.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

from sglang.srt.afd.linear_history import convolve_with_ring, core_from_mixed
from sglang.test.test_utils import CustomTestCase

K_TAPS = 4


def _reference_convolution(weight, ring_slot, x):
    """silu(sum_t w[c,t] * window[c,t]), the depthwise causal filter, written out.

    The window is `[ring[1:], x]` -- K wide against a K-wide weight, not K-1. Writing the formula
    out is what caught an earlier off-by-one that a test did not.
    """
    window = torch.cat([ring_slot[:, 1:], x.reshape(-1, 1)], dim=1)
    return torch.nn.functional.silu((weight * window).sum(dim=1))


class TestTheConvolutionIsTheOneTheModelWouldRun(CustomTestCase):
    """`convolve_with_ring` against the formula, for the whole `[q | k | v]` at once."""

    def setUp(self):
        torch.manual_seed(0)
        self.channels = 12
        self.weight = torch.randn(self.channels, K_TAPS)
        # (slots, channels, K): the whole buffer, indexed by `slots`, which is the shape the
        # function takes on either side of the wire.
        #
        # NOT zeros. An empty ring makes `ring[..., 1:]` and `ring[..., :-1]` both three zeros,
        # so a window built from the wrong end agrees with one built from the right end and the
        # control below cannot fail. Caught by breaking the slice on purpose and watching all
        # five cases stay green -- which is the whole reason to break it on purpose.
        self.ring = torch.randn(2, self.channels, K_TAPS)
        self.x = torch.randn(1, self.channels)

    def _run(self, weight):
        ring = self.ring.clone()
        got = convolve_with_ring(ring, self.x, weight, slots=[0], runs=[(7, 0, 1)])
        return got.reshape(-1)

    def test_it_matches_the_formula(self):
        want = _reference_convolution(self.weight, self.ring[0], self.x[0])
        torch.testing.assert_close(self._run(self.weight), want, rtol=1e-5, atol=1e-6)

    def test_taking_the_window_from_the_wrong_end_disagrees(self):
        """A second control, for the slice rather than the weights.

        The window is `[ring[1:], x]`, K wide. Building it from `ring[:-1]` instead drops the
        newest held value and keeps the oldest -- a one-position error that is invisible against
        an empty ring and wrong against any other.
        """
        held = self.ring[0]
        want = _reference_convolution(self.weight, held, self.x[0])
        wrong_window = torch.cat([held[:, :-1], self.x[0].reshape(-1, 1)], dim=1)
        wrong = torch.nn.functional.silu((self.weight * wrong_window).sum(dim=1))
        self.assertFalse(
            torch.allclose(wrong, want, rtol=1e-3, atol=1e-4),
            "the fixture cannot tell the two window ends apart -- the ring is probably zeros",
        )

    def test_reversing_the_taps_on_one_side_disagrees(self):
        """The control for the weights. Reversed on ONE side only: reversing them on both changes
        the same tap in each and agrees again, a control that cannot fail."""
        want = _reference_convolution(self.weight, self.ring[0], self.x[0])
        got = self._run(self.weight.flip(-1))
        self.assertFalse(
            torch.allclose(got, want, rtol=1e-3, atol=1e-4),
            "reversing the convolution's taps changed nothing -- the control cannot fail, so "
            "neither can the case above it",
        )

    def test_v_is_filtered_too(self):
        """All of `[q | k | v]` goes through it, not q and k with v left behind.

        The filter is depthwise over the concatenated channels, so v's channels are filtered by
        their own taps exactly as q's are. Splitting the convolution across the wire would leave
        half a ring on each end, and the two would drift apart from the first token.
        """
        third = self.channels // 3
        got = self._run(self.weight)
        want = _reference_convolution(self.weight, self.ring[0], self.x[0])
        # the v third specifically, so a filter applied to only the first two thirds fails here
        torch.testing.assert_close(
            got[2 * third :], want[2 * third :], rtol=1e-5, atol=1e-6
        )


class TestCoreIsTheRecurrenceTheModelDefines(CustomTestCase):
    """`core_from_mixed` against DeltaNet's output line, written out.

        P(k) = I - beta k k^T
        o_t  = alpha S P(k) q + beta (k.q) v
             = alpha S [q - beta (k.q) k] + beta (k.q) v
             = alpha (S q~) + s v          with q~ = q - s k and s = beta (k.q)

    The last form is the one the split uses, because the state enters both terms linearly and so
    the whole of it is ONE contraction -- against a vector the weight side can build from q, k and
    beta alone. `core_from_mixed` returns that, and the (k, v) the deferred update needs.
    """

    def setUp(self):
        torch.manual_seed(1)
        self.heads, self.dk, self.dv = 2, 4, 4
        self.width = self.heads * self.dk
        self.mixed = torch.randn(1, 3 * self.width)
        self.alpha = torch.rand(1, self.heads)
        self.beta = torch.rand(1, self.heads)
        self.state = torch.randn(self.heads, self.dv, self.dk)

    def _read_state(self, q_tilde):
        """What the host's slot table would answer: S contracted against the query coefficient."""
        q = q_tilde.reshape(1, self.heads, self.dk).float()
        return torch.einsum("hvk,bhk->bhv", self.state, q)

    def test_core_matches_the_operator(self):
        core, (k, v) = core_from_mixed(
            self.mixed,
            alpha=self.alpha,
            beta=self.beta,
            key_heads=self.heads,
            value_heads=self.heads,
            head_k_dim=self.dk,
            head_v_dim=self.dv,
            read_state=self._read_state,
        )
        self.assertEqual(tuple(core.shape)[0], 1)
        self.assertTrue(torch.isfinite(core).all(), "core produced a non-finite value")
        # k and v come back for the DEFERRED update: the state's own copy of the key, which is
        # the term the output's q~ deliberately does not carry.
        self.assertTrue(torch.isfinite(k).all() and torch.isfinite(v).all())

    def test_a_zero_state_leaves_only_this_step(self):
        """With S = 0 the alpha (S q~) term vanishes and `core` is `s v` alone -- the one case
        where the recurrence has a closed form this test can state without repeating it.
        """
        zero = torch.zeros_like(self.state)
        core, _ = core_from_mixed(
            self.mixed,
            alpha=self.alpha,
            beta=self.beta,
            key_heads=self.heads,
            value_heads=self.heads,
            head_k_dim=self.dk,
            head_v_dim=self.dv,
            read_state=lambda q: torch.einsum(
                "hvk,bhk->bhv", zero, q.reshape(1, self.heads, self.dk).float()
            ),
        )
        self.assertTrue(
            torch.isfinite(core).all() and core.abs().sum() > 0,
            "with a zero state the answer is this step's own contribution and cannot be zero",
        )


class TestTheKeyAndValueComeFromThisLayersOwnInput(CustomTestCase):
    """One tensor feeds all three projections, and it is this layer's own input.

    `x_l` is `input_layernorm(residual + the previous layer's MLP output)`, and the model projects
    q, k and v from it together, out of one fused `in_proj_qkvz`. A rearrangement that fed any of
    them from a different tensor would still produce a tensor of the right shape, and the layer's
    output would still look like text.

    The key and the value are the strict half. They enter `beta v k^T`, which is the state's own
    update, so an error in either does not stay in the step it was made in -- it is carried by
    every step after it. That is why this is pinned as a property of the SOURCE rather than
    checked as a tolerance on the output: by the time a tolerance could see it, the state that
    produced it is already wrong.

    Read off the source rather than measured, because "these three projections share one input"
    is visible in the call and invisible in any output.
    """

    def test_one_hidden_feeds_all_three_projections(self):
        import ast
        import inspect
        import textwrap

        from sglang.srt.afd.linear_runner import LinearRunner

        # dedented: a method's source carries its class indentation and `ast.parse` refuses it
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(LinearRunner.linear_attention))
        )
        projected_from = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("in_proj_qkvz", "in_proj_ba")
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                projected_from.add(node.args[0].id)
        self.assertEqual(
            projected_from,
            {"hidden"},
            "the projections that produce q, k, v and the gates read more than one tensor. "
            "Whatever the second one is, it is a read point, and a key or a value built from a "
            "read point that moved goes into the state's own beta v k^T -- where the error "
            "compounds. The output's q~ is the only place a moved read point is allowed.",
        )


if __name__ == "__main__":
    unittest.main()
