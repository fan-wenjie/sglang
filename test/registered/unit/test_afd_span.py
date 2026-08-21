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


class Layer(torch.nn.Module):
    def __init__(self, layer_id: int, full: bool):
        super().__init__()
        self.input_layernorm = FusedNorm(1.0 + 0.01 * layer_id)
        self.post_attention_layernorm = FusedNorm(1.0 + 0.02 * layer_id)
        self.mlp = Mlp(seed=100 + layer_id)
        if full:
            self.self_attn = torch.nn.Module()
            self.self_attn.o_proj = Proj(H, H, seed=200 + layer_id)
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


def reference(stack, o, residual, span):
    """The same span, written out, with the residual visible at every step.

    Returns (read point, span output). The read point is the residual after the LAST linear
    layer's attention and before its feed-forward -- one feed-forward earlier than the output.
    """
    layers = stack.model.layers
    head, rest = span[0], span[1:]
    hidden = o @ layers[head].self_attn.o_proj.w
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
    return read_point, hidden


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
    first linear layer, or returning the span's output where its read point belongs.
    """

    def test_both_returned_tensors_match(self):
        stack, runner = a_runner()
        o = torch.randn(2, H)
        residual = torch.randn(2, H)
        for i in range(2):
            runner.seed(i, residual[i])
        read_point, out = runner.run([0, 1], 3, o)
        want_read, want_out = reference(stack, o, residual, (3, 4, 5, 6))
        torch.testing.assert_close(read_point, want_read)
        torch.testing.assert_close(out, want_out)

    def test_the_read_point_is_a_feed_forward_earlier_than_the_output(self):
        """Shift 1 is the whole reason the reply is two messages.

        If the read point were the span's output there would be nothing to send early, the host
        would have no head start, and the arrangement would still produce correct text -- the
        standard wiring with a socket in it. Nothing else here would notice.
        """
        stack, runner = a_runner()
        runner.seed(0, torch.randn(H))
        read_point, out = runner.run([0], 3, torch.randn(1, H))
        self.assertFalse(torch.allclose(read_point, out))


class TestTheResidualStaysOnThePool(CustomTestCase):
    """A group's input residual is the previous group's output, so it never travels.

    The failure this guards is a span that silently starts from zero. That is a correct forward
    pass over a history the request does not have, and the text stays fluent.
    """

    def test_a_span_chains_into_the_next(self):
        stack, runner = a_runner()
        runner.seed(0, torch.randn(H))
        read_point, out = runner.run([0], 3, torch.randn(1, H))
        held = runner._residual[0]
        torch.testing.assert_close(held, (read_point + out)[0])

    def test_a_request_the_pool_has_never_seen_is_refused(self):
        _, runner = a_runner()
        with self.assertRaises(RuntimeError) as caught:
            runner.run([7], 3, torch.randn(1, H))
        self.assertIn("no residual", str(caught.exception))

    def test_release_forgets_it(self):
        _, runner = a_runner()
        runner.seed(0, torch.randn(H))
        runner.release(0)
        with self.assertRaises(RuntimeError):
            runner.run([0], 3, torch.randn(1, H))


class TestTheEndsOfTheStackAreNotMiddleSpans(CustomTestCase):
    """Three arrangements with three shapes; conflating them would misreport one as another."""

    def test_the_prologue_is_refused_by_run(self):
        _, runner = a_runner()
        with self.assertRaises(ValueError) as caught:
            runner.run([0], -1, torch.randn(1, H))
        self.assertIn("run_prologue", str(caught.exception))

    def test_the_epilogue_is_refused_by_run(self):
        _, runner = a_runner()
        runner.seed(0, torch.randn(H))
        with self.assertRaises(ValueError) as caught:
            runner.run([0], 7, torch.randn(1, H))
        self.assertIn("run_epilogue", str(caught.exception))

    def test_the_prologue_starts_a_request_off_and_hands_over_a_read_point(self):
        """Layer 3 is not the shift's exempt layer: three layers sit beneath it."""
        _, runner = a_runner()
        read_point, out = runner.run_prologue([0], torch.randn(1, H))
        self.assertFalse(torch.allclose(read_point, out))
        torch.testing.assert_close(runner._residual[0], (read_point + out)[0])

    def test_a_layer_that_starts_no_span_is_refused(self):
        _, runner = a_runner()
        with self.assertRaises(KeyError):
            runner.run([0], 5, torch.randn(1, H))


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
