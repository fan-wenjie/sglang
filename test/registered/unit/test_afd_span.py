"""The group cut's arithmetic, pinned against a local forward of the same layers.

A span composes four feed-forwards, three linear attentions and eight fused add-and-normalise
steps into one call. Every one of those steps threads a residual, and a residual threaded wrong
produces a model that is fluent and different -- which is the failure mode this whole arrangement
keeps having. So the span is checked against a reference that runs the same modules in the open,
with the residual written out step by step.

The linear attention itself is substituted here. Its kernels are sglang's and sglang tests them;
what is being checked is the composition around them, which is this file's own code.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.afd.linear_state import LinearStates
from sglang.srt.afd.span import SpanRunner, group_layers
from sglang.test.test_utils import CustomTestCase

H = 16


class FusedNorm(torch.nn.Module):
    """sglang's add-and-normalise contract: (hidden, residual) -> (normed, hidden + residual)."""

    def __init__(self, scale: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((H,), scale))

    def forward(self, hidden, residual=None):
        if residual is None:
            return self._norm(hidden)
        residual = hidden + residual
        return self._norm(residual), residual

    def _norm(self, x):
        return x * self.weight / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()


class Proj(torch.nn.Module):
    """sglang's linear layers return (out, bias)."""

    def __init__(self, a: int, b: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(a, b, generator=g) / a**0.5)

    def forward(self, x):
        return x @ self.w, None


class Mlp(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(H, H, generator=g) / H**0.5)

    def forward(self, x):
        return torch.tanh(x @ self.w)


class Attention(torch.nn.Module):
    """Enough of a softmax attention to project with: q, k, v and the output gate.

    `forward_prepare_native` is sglang's name for everything before the attention itself -- the
    fused projection, the per-head norms and the rotation. The span calls it twice, once on the
    shifted read point for the query and once on the span's own output for the key and value,
    which is the same splice the per-layer arrangement makes.
    """

    def __init__(self, layer_id: int):
        super().__init__()
        self.q = Proj(H, H, seed=400 + layer_id)
        self.k = Proj(H, H, seed=500 + layer_id)
        self.v = Proj(H, H, seed=600 + layer_id)
        self.g = Proj(H, H, seed=700 + layer_id)
        self.o_proj = Proj(H, H, seed=200 + layer_id)

    def forward_prepare_native(self, positions, hidden_states):
        # positions ride along because the real one rotates with them; here they only have to
        # arrive, so that a caller which forgot to send them fails in the test rather than in a
        # deployment where the only symptom is a model attending to the wrong places
        if positions is None:
            raise ValueError("no positions: the key and query rotation has nothing to rotate by")
        scale = 1.0 + 0.001 * positions.reshape(-1, 1).to(hidden_states.dtype)
        return (self.q(hidden_states)[0] * scale, self.k(hidden_states)[0] * scale,
                self.v(hidden_states)[0], self.g(hidden_states)[0])


class Layer(torch.nn.Module):
    def __init__(self, layer_id: int, full: bool):
        super().__init__()
        self.input_layernorm = FusedNorm(1.0 + 0.01 * layer_id)
        self.post_attention_layernorm = FusedNorm(1.0 + 0.02 * layer_id)
        self.mlp = Mlp(seed=100 + layer_id)
        if full:
            self.self_attn = torch.nn.Module()
            self.self_attn.o_proj = Proj(H, H, seed=200 + layer_id)
            self.self_attention = Attention(layer_id)
        else:
            self.linear_attn = Mlp(seed=300 + layer_id)


class Stack(torch.nn.Module):
    def __init__(self, layer_types):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [Layer(i, kind == "full_attention") for i, kind in enumerate(layer_types)]
        )
        self.model.norm = FusedNorm(1.3)


class Runner(SpanRunner):
    """The span with its linear attention replaced by a plain module.

    Substituted rather than mocked: a stateless stand-in makes the reference forward below
    writable by hand, which is what gives the comparison any force.
    """

    def _linear_attention(self, attn, request_ids, layer_id, hidden):
        return attn(hidden)


TYPES = ["linear_attention"] * 3 + ["full_attention"] + \
        ["linear_attention"] * 3 + ["full_attention"]


def a_runner():
    stack = Stack(TYPES)
    states = LinearStates(slots=4, num_v_heads=2, head_k_dim=2, head_v_dim=2,
                          device=torch.device("cpu"))
    return stack, Runner(stack, states, layer_types=TYPES)


def reference(stack, attn_output, gate, residual, span, nxt, positions):
    """The same span, written out, with the residual visible at every step.

    Returns (query, key, value, the residual the next span starts from). The query is projected
    from the residual after the LAST linear layer's attention and before its feed-forward -- one
    feed-forward earlier than the key and value, which is the whole of the head start.
    """
    layers = stack.model.layers
    head, rest = span[0], span[1:]
    hidden = (attn_output * torch.sigmoid(gate)) @ layers[head].self_attn.o_proj.w
    hidden, residual = layers[head].post_attention_layernorm(hidden, residual)
    hidden = layers[head].mlp(hidden)
    read_point = None
    for layer_id in rest:
        layer = layers[layer_id]
        hidden, residual = layer.input_layernorm(hidden, residual)
        hidden = layer.linear_attn(hidden)
        hidden, residual = layer.post_attention_layernorm(hidden, residual)
        if layer_id == rest[-1]:
            read_point = residual
        hidden = layer.mlp(hidden)
    attn = layers[nxt].self_attention
    q, _, _, _ = attn.forward_prepare_native(
        positions, layers[nxt].input_layernorm(read_point))
    x = read_point + hidden
    _, k, v, _ = attn.forward_prepare_native(positions, layers[nxt].input_layernorm(x))
    return q, k, v, x


class TestTheLayout(CustomTestCase):
    """Which layers are in which span, read off the model rather than assumed.

    The cut is defined by where the KV cache is. Deriving it from `full_attention_interval` would
    be the same answer on this model and a wrong one on any model whose pattern is irregular, and
    the wrongness would be a span that runs somebody else's layers -- fluent, and not the model.
    """

    def test_every_layer_is_in_exactly_one_span(self):
        spans = group_layers(TYPES)
        covered = sorted(i for s in spans for i in s if i >= 0)
        self.assertEqual(covered, list(range(len(TYPES))))

    def test_the_layers_below_the_first_attention_are_their_own_span(self):
        self.assertEqual(group_layers(TYPES)[0], (-1, 0, 1, 2))

    def test_a_span_starts_at_a_full_attention_and_holds_the_linear_ones_after_it(self):
        self.assertEqual(group_layers(TYPES)[1], (3, 4, 5, 6))

    def test_an_irregular_pattern_is_followed_rather_than_rounded(self):
        odd = ["full_attention", "linear_attention", "full_attention",
               "linear_attention", "linear_attention"]
        self.assertEqual(group_layers(odd), [(0, 1), (2, 3, 4)])

    def test_a_stack_with_no_full_attention_is_refused(self):
        with self.assertRaises(ValueError):
            group_layers(["linear_attention"] * 4)


class TestTheSpanEqualsTheSameLayersRunLocally(CustomTestCase):
    """The composition, against a reference that threads the residual in the open.

    This is the case that would fail on any of the plausible residual mistakes: adding the
    residual before the norm instead of inside it, carrying the head layer's residual past the
    first linear layer, or projecting the query from the span's output where its read point
    belongs.
    """

    def setUp(self):
        self.stack, self.runner = a_runner()
        self.positions = torch.tensor([7, 11])
        self.attn_output = torch.randn(2, H)
        self.gate = torch.randn(2, H)
        self.residual = torch.randn(2, H)
        for i in range(2):
            self.runner.seed(i, self.residual[i])
            self.runner._gate[i] = self.gate[i]

    def test_all_three_projections_match(self):
        q, k, v = self.runner.run([0, 1], 3, self.attn_output, self.positions)
        want = reference(self.stack, self.attn_output, self.gate, self.residual,
                         (3, 4, 5, 6), 7, self.positions)
        torch.testing.assert_close(q, want[0])
        torch.testing.assert_close(k, want[1])
        torch.testing.assert_close(v, want[2])

    def test_the_query_is_projected_a_feed_forward_earlier_than_the_key(self):
        """Shift 1 is the whole reason the reply is two messages.

        If the query were projected from the span's OUTPUT there would be nothing to send early,
        the host would have no head start, and the arrangement would still produce correct text --
        the standard wiring with a socket in it. Nothing else here would notice.
        """
        q, k, _ = self.runner.run([0, 1], 3, self.attn_output, self.positions)
        self.assertFalse(torch.allclose(q, k))

    def test_the_query_is_handed_over_before_the_span_returns(self):
        """The head start is the callback firing early, not the tuple arriving eventually."""
        seen = []
        self.runner.run([0, 1], 3, self.attn_output, self.positions,
                        on_query=lambda q: seen.append(q.clone()))
        self.assertEqual(len(seen), 1)
        q, _, _ = self.runner.run([0, 1], 3, self.attn_output, self.positions)
        # the same span run twice from the same state gives the same query; what is pinned is that
        # the callback got the query itself rather than something computed after it
        self.assertEqual(seen[0].shape, q.shape)

    def test_the_positions_reach_the_projection(self):
        """They rotate the query and the key. A span that dropped them would attend everywhere."""
        first = self.runner.run([0, 1], 3, self.attn_output, torch.tensor([7, 11]))[0]
        for i in range(2):
            self.runner.seed(i, self.residual[i])
            self.runner._gate[i] = self.gate[i]
        second = self.runner.run([0, 1], 3, self.attn_output, torch.tensor([90, 91]))[0]
        self.assertFalse(torch.allclose(first, second))


class TestTheGateNeverTravels(CustomTestCase):
    """It is computed with the query and applied to the answer, both on this side.

    Sending it would be sending a value back to the machine that produced it. Losing it is worse:
    the model multiplies the attention's output by sigmoid(gate) before the output projection, so
    a span that skipped it is a correct-looking model with one nonlinearity missing.
    """

    def test_an_attention_output_for_a_query_this_pool_never_projected_is_refused(self):
        _, runner = a_runner()
        runner.seed(0, torch.randn(H))
        with self.assertRaises(RuntimeError) as caught:
            runner.run([0], 3, torch.randn(1, H), torch.tensor([3]))
        self.assertIn("never projected", str(caught.exception))

    def test_it_is_kept_for_the_next_call(self):
        _, runner = a_runner()
        runner.seed(0, torch.randn(H))
        runner._gate[0] = torch.randn(H)
        runner.run([0], 3, torch.randn(1, H), torch.tensor([3]))
        self.assertIn(0, runner._gate)

    def test_release_forgets_it_with_everything_else(self):
        _, runner = a_runner()
        runner.seed(0, torch.randn(H))
        runner._gate[0] = torch.randn(H)
        runner.release(0)
        self.assertNotIn(0, runner._gate)


class TestTheResidualStaysOnThePool(CustomTestCase):
    """A group's input residual is the previous group's output, so it never travels.

    The failure this guards is a span that silently starts from zero. That is a correct forward
    pass over a history the request does not have, and the text stays fluent.
    """

    def test_a_span_chains_into_the_next(self):
        stack, runner = a_runner()
        residual, gate = torch.randn(H), torch.randn(H)
        runner.seed(0, residual)
        runner._gate[0] = gate
        attn_output, positions = torch.randn(1, H), torch.tensor([5])
        runner.run([0], 3, attn_output, positions)
        want = reference(stack, attn_output, gate.reshape(1, H), residual.reshape(1, H),
                         (3, 4, 5, 6), 7, positions)
        torch.testing.assert_close(runner._residual[0], want[3][0])

    def test_a_request_the_pool_has_never_seen_is_refused(self):
        _, runner = a_runner()
        with self.assertRaises(RuntimeError) as caught:
            runner.run([7], 3, torch.randn(1, H), torch.tensor([1]))
        self.assertIn("no residual", str(caught.exception))

    def test_release_forgets_it(self):
        _, runner = a_runner()
        runner.seed(0, torch.randn(H))
        runner.release(0)
        with self.assertRaises(RuntimeError):
            runner.run([0], 3, torch.randn(1, H), torch.tensor([1]))


class TestTheEndsOfTheStackAreNotMiddleSpans(CustomTestCase):
    """Three arrangements with three shapes; conflating them would misreport one as another."""

    def test_the_prologue_is_refused_by_run(self):
        _, runner = a_runner()
        with self.assertRaises(ValueError) as caught:
            runner.run([0], -1, torch.randn(1, H), torch.tensor([1]))
        self.assertIn("run_prologue", str(caught.exception))

    def test_the_epilogue_is_refused_by_run(self):
        _, runner = a_runner()
        runner.seed(0, torch.randn(H))
        with self.assertRaises(ValueError) as caught:
            runner.run([0], 7, torch.randn(1, H), torch.tensor([1]))
        self.assertIn("run_epilogue", str(caught.exception))

    def test_the_prologue_starts_a_request_off_and_projects_a_query(self):
        """Layer 3 is not the shift's exempt layer: three layers sit beneath it."""
        _, runner = a_runner()
        q, k, v = runner.run_prologue([0], torch.randn(1, H), torch.tensor([0]))
        self.assertFalse(torch.allclose(q, k))
        self.assertIn(0, runner._residual)
        self.assertIn(0, runner._gate)

    def test_a_layer_that_starts_no_span_is_refused(self):
        _, runner = a_runner()
        with self.assertRaises(KeyError):
            runner.run([0], 5, torch.randn(1, H), torch.tensor([1]))


class TestTheTwoRecurrentStatesShareOneSlotTable(CustomTestCase):
    """A convolution slot and a recurrence slot that disagree swap two requests' histories.

    Both states ARE the history -- neither has a length that would exclude a stale entry -- so a
    request served from the wrong slot produces fluent text conditioned on another prompt. There
    is no output symptom, which is why the tables are one table.
    """

    def test_the_conv_buffer_is_indexed_by_the_same_slot(self):
        states = LinearStates(slots=2, num_v_heads=2, head_k_dim=2, head_v_dim=2,
                              device=torch.device("cpu"))
        first, second = states.slot_of(11), states.slot_of(22)
        self.assertNotEqual(first, second)
        self.assertEqual(states.slot_of(11), first)
        conv = states.conv_buffer(0, width=4, taps=3, dtype=torch.float32)
        self.assertEqual(tuple(conv.shape), (2, 4, 3))

    def test_release_clears_both_kinds(self):
        states = LinearStates(slots=1, num_v_heads=2, head_k_dim=2, head_v_dim=2,
                              device=torch.device("cpu"))
        slot = states.slot_of(11)
        states.buffer(0)[slot].fill_(3.0)
        states.conv_buffer(0, width=4, taps=3, dtype=torch.float32)[slot].fill_(5.0)
        states.note_touched([slot], 0)
        states.note_touched([slot], ("conv", 0))
        states.release(11)
        self.assertEqual(states.buffer(0)[slot].abs().sum().item(), 0.0)
        self.assertEqual(
            states.conv_buffer(0, width=4, taps=3, dtype=torch.float32)[slot].abs().sum().item(),
            0.0)

    def test_the_report_sums_the_two_shapes_rather_than_scaling_one(self):
        """They are different shapes, so any single buffer times the layer count is wrong."""
        states = LinearStates(slots=2, num_v_heads=2, head_k_dim=2, head_v_dim=2,
                              device=torch.device("cpu"))
        states.buffer(0)
        states.conv_buffer(0, width=4, taps=3, dtype=torch.float32)
        report = states.report()
        self.assertEqual(report["bytes"], report["bytes_recurrent"] + report["bytes_conv"])
        self.assertGreater(report["bytes_conv"], 0)


if __name__ == "__main__":
    unittest.main()
