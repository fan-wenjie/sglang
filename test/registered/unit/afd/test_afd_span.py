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

# The tiny stack serves at shift 1 -- its span pushes the early read -- so the whole
# file needs the early-read package. On the base tree (which serves shift 0 and refuses
# shift 1 by name) it skips whole; the base arrangement's behavior is pinned by the
# serving tests around it and by the live equivalence checks.
try:
    import sglang.srt.afd_query_shift  # noqa: F401

    _HAS_EARLY = True
except ImportError:  # base tree: the derived package is absent by design
    _HAS_EARLY = False

if not _HAS_EARLY:
    import pytest

    pytestmark = pytest.mark.skip(
        reason="the tiny stack serves at shift 1; the early-read package is absent"
    )

H = 16


class FusedNorm(torch.nn.Module):
    """sglang's add-and-normalise contract: (hidden, residual) -> (normed, hidden + residual)."""

    def __init__(self, scale: float):
        super().__init__()
        # PER CHANNEL, not a constant. RMSNorm with a constant weight is idempotent --
        # norm(norm(x)) == norm(x) -- so a fake built with `torch.full` cannot tell one
        # application from two, and the final norm being applied twice went unnoticed by every
        # case in this file. The deployed checkpoint's own final norm runs from -0.285 to 1.711,
        # so it is the varying weight that is faithful and the constant that was the fake.
        #
        # GEMMA CONVENTION: the scale applied is `1 + weight`, because that is what the model's
        # `GemmaRMSNorm` applies and what the checkpoint's near -1 stored weights mean. A fake
        # with the plain convention kept every case here green while the install-time norm
        # ratio, computed from the stored weights of the real model, served the wrong sign.
        self.weight = torch.nn.Parameter(scale * torch.linspace(0.4, 1.6, H) - 1.0)

    def forward(self, hidden, residual=None):
        if residual is None:
            return self._norm(hidden)
        residual = hidden + residual
        return self._norm(residual), residual

    def _norm(self, x):
        return x * (1.0 + self.weight) / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()


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
    """Shaped like `Qwen3HybridAttentionDecoderLayer`, which is the point of this class.

    The projections and the attention hang off the DECODER LAYER; `self_attention` is a METHOD on
    it, not a submodule. An earlier version of this fake put them on a `self_attention` object and
    a `self_attn` object, and every case here passed against a structure the model does not have --
    the span reached through `layer.self_attention.attn` and died at the first token with
    "'function' object has no attribute 'attn'". A fake whose shape differs from the real one
    tests the fake.

    `forward_prepare_native` is sglang's name for everything before the attention itself -- the
    fused projection, the per-head norms and the rotation. The span calls it twice, once on the
    shifted read point for the query and once on the span's own output for the key and value,
    which is the same splice the per-layer arrangement makes.
    """

    def __init__(self, layer_id: int, full: bool):
        super().__init__()
        self.input_layernorm = FusedNorm(1.0 + 0.01 * layer_id)
        self.post_attention_layernorm = FusedNorm(1.0 + 0.02 * layer_id)
        self.mlp = Mlp(seed=100 + layer_id)
        if full:
            self.q = Proj(H, H, seed=400 + layer_id)
            self.k = Proj(H, H, seed=500 + layer_id)
            self.v = Proj(H, H, seed=600 + layer_id)
            self.g = Proj(H, H, seed=700 + layer_id)
            self.o_proj = Proj(H, H, seed=200 + layer_id)
        else:
            self.linear_attn = Mlp(seed=300 + layer_id)

    def forward_prepare_native(self, positions, hidden_states):
        # positions ride along because the real one rotates with them; here they only have to
        # arrive, so that a caller which forgot to send them fails in the test rather than in a
        # deployment where the only symptom is a model attending to the wrong places
        if positions is None:
            raise ValueError(
                "no positions: the key and query rotation has nothing to rotate by"
            )
        scale = 1.0 + 0.001 * positions.reshape(-1, 1).to(hidden_states.dtype)
        return (
            self.q(hidden_states)[0] * scale,
            self.k(hidden_states)[0] * scale,
            self.v(hidden_states)[0],
            self.g(hidden_states)[0],
        )


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

    def _send_early(
        self, layer, attn, request_ids, layer_id, prev_attn_residual, raw_residual=None
    ):
        # Nothing, deliberately. The real one projects the shifted query and key off the previous
        # residual, which needs a linear-attention module with `key_dim` and `in_proj_qkvz` -- and
        # this stand-in's linear attention is a plain `Mlp`, because what these cases compare is
        # the reference forward written by hand below. A second fake shaped like the real module
        # would be a second thing to keep in step with the model.
        #
        # `RecordingRunner` pins the ORDER the real one is called in, which is the part that
        # matters here.
        return None

    def _linear_attention(
        self,
        layer,
        attn,
        request_ids,
        layer_id,
        hidden,
        prev_attn_residual=None,
    ):
        # what this substitutes is the arithmetic; the comparison below is about which layers
        # ran, not about when their projections were sent.
        return attn(hidden)


TYPES = (
    ["linear_attention"] * 3
    + ["full_attention"]
    + ["linear_attention"] * 3
    + ["full_attention"]
)


def a_runner():
    stack = Stack(TYPES)
    states = LinearStates(
        slots=4, num_v_heads=2, head_k_dim=2, head_v_dim=2, device=torch.device("cpu")
    )
    # shift 1 is the operating point and the value every case below was written against. Given
    # explicitly because the span no longer assumes it: a run at 0 reads the query from the
    # group's output instead, and a default here would let a caller that forgot to choose get
    # whichever the last edit happened to prefer.
    return stack, Runner(stack, states, layer_types=TYPES, query_shift=1)


def reference(stack, attn_output, gate, residual, span, nxt, positions):
    """The same span, written out, with the residual visible at every step.

    Returns (query, key, value, the residual the next span starts from). The query is projected
    from the residual after the LAST linear layer's attention and before its feed-forward -- one
    feed-forward earlier than the key and value, which is the whole of the head start.
    """
    layers = stack.model.layers
    head, rest = span[0], span[1:]
    hidden = (attn_output * torch.sigmoid(gate)) @ layers[head].o_proj.w
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
    attn = layers[nxt]
    q, _, _, _ = attn.forward_prepare_native(
        positions, attn.input_layernorm(read_point)
    )
    x = read_point + hidden
    _, k, v, _ = attn.forward_prepare_native(positions, attn.input_layernorm(x))
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
        odd = [
            "full_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
        ]
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
            self.runner._gate[i] = self.gate[i].reshape(1, -1)

    def test_all_three_projections_match(self):
        q, k, v = self.runner.run([0, 1], 3, self.attn_output, self.positions)
        want = reference(
            self.stack,
            self.attn_output,
            self.gate,
            self.residual,
            (3, 4, 5, 6),
            7,
            self.positions,
        )
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
        self.runner.run(
            [0, 1],
            3,
            self.attn_output,
            self.positions,
            on_query=lambda q: seen.append(q.clone()),
        )
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
            self.runner._gate[i] = self.gate[i].reshape(1, -1)
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
        runner._gate[0] = torch.randn(1, H)
        runner.run([0], 3, torch.randn(1, H), torch.tensor([3]))
        self.assertIn(0, runner._gate)

    def test_release_forgets_it_with_everything_else(self):
        _, runner = a_runner()
        runner.seed(0, torch.randn(H))
        runner._gate[0] = torch.randn(1, H)
        runner.release(0)
        self.assertNotIn(0, runner._gate)


class TestTheResidualStaysOnThePool(CustomTestCase):
    """A group's input residual is the previous group's output, so it never travels.

    The failure this guards is a span that silently starts from zero. That is a correct forward
    pass over a history the request does not have, and the text stays fluent.
    """

    def test_a_span_chains_into_the_next(self):
        stack, runner = a_runner()
        residual, gate = torch.randn(1, H), torch.randn(1, H)
        runner.seed(0, residual)
        runner._gate[0] = gate
        attn_output, positions = torch.randn(1, H), torch.tensor([5])
        runner.run([0], 3, attn_output, positions)
        want = reference(
            stack,
            attn_output,
            gate.reshape(1, H),
            residual.reshape(1, H),
            (3, 4, 5, 6),
            7,
            positions,
        )
        torch.testing.assert_close(runner._residual[0], want[3][0].reshape(1, H))

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
        states = LinearStates(
            slots=2,
            num_v_heads=2,
            head_k_dim=2,
            head_v_dim=2,
            device=torch.device("cpu"),
        )
        first, second = states.slot_of(11), states.slot_of(22)
        self.assertNotEqual(first, second)
        self.assertEqual(states.slot_of(11), first)
        conv = states.conv_buffer(0, width=4, taps=3, dtype=torch.float32)
        self.assertEqual(tuple(conv.shape), (2, 4, 3))

    def test_release_clears_both_kinds(self):
        states = LinearStates(
            slots=1,
            num_v_heads=2,
            head_k_dim=2,
            head_v_dim=2,
            device=torch.device("cpu"),
        )
        slot = states.slot_of(11)
        states.buffer(0)[slot].fill_(3.0)
        states.conv_buffer(0, width=4, taps=3, dtype=torch.float32)[slot].fill_(5.0)
        states.note_touched([slot], 0)
        states.note_touched([slot], ("conv", 0))
        states.release(11)
        self.assertEqual(states.buffer(0)[slot].abs().sum().item(), 0.0)
        self.assertEqual(
            states.conv_buffer(0, width=4, taps=3, dtype=torch.float32)[slot]
            .abs()
            .sum()
            .item(),
            0.0,
        )

    def test_the_report_sums_the_two_shapes_rather_than_scaling_one(self):
        """They are different shapes, so any single buffer times the layer count is wrong."""
        states = LinearStates(
            slots=2,
            num_v_heads=2,
            head_k_dim=2,
            head_v_dim=2,
            device=torch.device("cpu"),
        )
        states.buffer(0)
        states.conv_buffer(0, width=4, taps=3, dtype=torch.float32)
        report = states.report()
        self.assertEqual(
            report["bytes"], report["bytes_recurrent"] + report["bytes_conv"]
        )
        self.assertGreater(report["bytes_conv"], 0)


if __name__ == "__main__":
    unittest.main()


class TestTheEpilogueHandsBackAResidualStream(CustomTestCase):
    """The last thing the pool returns is normalised by the MODEL, not by the pool.

    sglang's `Qwen3_5Model.forward` applies `self.norm` after its layer loop, and the closing head
    hands it `residual=None`, so it takes the `self.norm(hidden_states)` branch on whatever comes
    back over the wire. The pool normalised as well, so the final RMSNorm ran TWICE -- with a
    learned weight w that is w squared elementwise, and the deployed checkpoint's
    `model.language_model.norm.weight` runs from -0.285 to 1.711, so squaring flips the sign of
    every negative channel and rescales the rest between 0.08x and 2.93x.

    No case here caught it, and the reason is in `FusedNorm` above: it was built with a CONSTANT
    weight, and RMSNorm with a constant weight is idempotent. The fake could not tell one
    application from two. It now has a per-channel weight, which is what makes this case possible
    at all.
    """

    def test_what_it_returns_is_not_already_normalised(self):
        """A normalised vector's row RMS is pinned by the norm's weight and says nothing about its
        input -- `rms(w * x/rms(x))` is `rms(w)` for every x. So two epilogues run on inputs of
        very different magnitude have nearly equal row RMS if the pool normalised, and different
        row RMS if what comes back is the residual stream.

        The first version of this case asserted that the model's final norm CHANGES what came
        back. That passes either way: this norm is not idempotent, so norming an
        already-normalised vector changes it too. The assertion held on the bug and on the fix and
        guarded nothing, which is why the discriminator here is a property of the norm's IMAGE
        rather than of applying it once more.
        """
        stack, runner = a_runner()
        small = self.an_epilogue(runner, stack, rid=7, scale=1.0)
        large = self.an_epilogue(runner, stack, rid=9, scale=40.0)

        def rms(t):
            return float(t.pow(2).mean(-1).sqrt().mean())

        ratio = rms(large) / rms(small)
        self.assertGreater(
            ratio,
            3.0,
            f"a 40x larger input moved the epilogue's output by {ratio:.2f}x. A residual stream "
            f"tracks its input; a normalised vector does not, because the pool normalised what "
            f"the model is about to normalise again",
        )

    def an_epilogue(self, runner, stack, *, rid, scale):
        rows = 1
        embedded = torch.randn(rows, H) * scale
        positions = torch.zeros(3, rows, dtype=torch.long)
        runner.run_prologue([rid], embedded, positions)
        attn = torch.randn(rows, stack.model.layers[3].o_proj.w.shape[0]) * scale
        return runner.run_epilogue([rid], 7, attn)

    def test_the_norms_image_is_flat_so_the_case_above_can_fail(self):
        """The control. If the norm's output RMS tracked its input, the ratio above would exceed 3
        whether the pool normalised or not and the case would be guarding nothing -- which is
        exactly what happened to its first version."""
        stack, runner = a_runner()
        small = stack.model.norm(torch.randn(4, H) * 1.0)
        large = stack.model.norm(torch.randn(4, H) * 40.0)

        def rms(t):
            return float(t.pow(2).mean(-1).sqrt().mean())

        self.assertLess(
            abs(rms(large) / rms(small) - 1.0),
            0.2,
            "this norm does not flatten its input's scale, so the discriminator the "
            "case above relies on does not hold",
        )


class TestAChunksRowsEachKeepTheirOwnResidual(CustomTestCase):
    """A prefill chunk is one request's tokens, and each token has its own residual.

    The tables were keyed by REQUEST. A decode batch is one row a request, so that was right there
    and only there. A 122-token prefill is 122 rows of one request: each row overwrote the last,
    the FINAL token's residual survived, and the next span handed it back to all 122 positions.
    Every position in the prompt then ran the rest of the stack on the last token's state.

    Measured at the boundary on the deployed model before this was fixed: 76.76 written leaving
    the prologue, 18.833 read entering the next group, same request, same boundary. Nothing
    raised; the arrangement served fluent, wrong text.
    """

    def test_each_row_comes_back_as_itself(self):
        _, runner = a_runner()
        rows = 4
        kept = torch.randn(rows, H)
        runner._keep_residual([3] * rows, kept)
        got = runner._take_residual([3] * rows, rows, kept)
        torch.testing.assert_close(got, kept)

    def test_the_last_row_is_not_broadcast_over_the_others(self):
        """The control. If the table still held one row a request, the line above would compare a
        broadcast final row against itself on the last position and pass on three quarters of a
        wrong answer."""
        _, runner = a_runner()
        rows = 4
        kept = torch.randn(rows, H)
        runner._keep_residual([3] * rows, kept)
        got = runner._take_residual([3] * rows, rows, kept)
        self.assertFalse(
            torch.allclose(got[0], got[-1]),
            "row 0 came back equal to the last row, which is what keying by request did",
        )

    def test_two_requests_in_one_batch_keep_their_own_rows(self):
        _, runner = a_runner()
        ids = [3, 3, 7]
        kept = torch.randn(3, H)
        runner._keep_residual(ids, kept)
        torch.testing.assert_close(runner._take_residual(ids, 3, kept), kept)
        self.assertEqual(runner._residual[3].shape[0], 2)
        self.assertEqual(runner._residual[7].shape[0], 1)

    def test_asking_for_a_different_row_count_is_refused(self):
        """Within a step the prologue writes the rows the rest of the step reads, so a mismatch
        means the arrangement lost track of who is in the batch. Refused rather than broadcast.
        """
        _, runner = a_runner()
        kept = torch.randn(4, H)
        runner._keep_residual([3] * 4, kept)
        with self.assertRaises(RuntimeError) as caught:
            runner._take_residual([3], 1, kept[:1])
        self.assertIn("kept 4 row(s)", str(caught.exception))

    def test_the_gate_is_kept_per_row_too(self):
        _, runner = a_runner()
        rows = 3
        gate = torch.randn(rows, H)
        runner._keep_gate([5] * rows, gate)
        torch.testing.assert_close(runner._take_gate([5] * rows, gate), gate)


class RecordingRunner(Runner):
    """A span that writes down WHEN each read was sent and when it was collected.

    The order is the whole of what the pipelined arrangement changed. Every reordering of it --
    issuing after the feed-forward instead of before, collecting a layer early, handing a layer
    its neighbour's read -- produces the SAME tokens, because the same arithmetic runs on the same
    values in a different sequence. The only visible difference is a benchmark number, and a
    benchmark number has many other explanations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedule: list[tuple[str, int]] = []
        for index, layer in enumerate(self.model.model.layers):
            layer.mlp.register_forward_pre_hook(self._note_mlp(index))

    def _note_mlp(self, index):
        def hook(_module, _args):
            self.schedule.append(("mlp", index))

        return hook

    def _send_early(
        self, layer, attn, request_ids, layer_id, prev_attn_residual, raw_residual=None
    ):
        # The real one needs `issue_host` and a linear-attention module, and installing a whole
        # socket pair here would test the transport rather than the order. What it must NOT do is
        # skip the residual: a version that sent from the wrong tensor would still record a send
        # in the right place, so the residual is asserted on rather than ignored.
        assert prev_attn_residual is not None
        self.schedule.append(("issue", layer_id))

    def _linear_attention(
        self,
        layer,
        attn,
        request_ids,
        layer_id,
        hidden,
        prev_attn_residual=None,
    ):
        self.schedule.append(("ask", layer_id))
        return attn(hidden)


class TestTheReadIsSentBeforeTheFeedForwardBesideIt(CustomTestCase):
    """#74's other half: the SPAN's schedule, asserted by order rather than by a timing.

    `test_sweep_window.py` asserts the per-layer router's three steps. A span has its own, and it
    was measured (61% of a read's latency elapsed under the pool's own work) before it was ever
    asserted. A measurement is the wrong instrument for this: it answers "is it faster" when the
    question is "did the read leave first", and those come apart exactly when something else on
    the machine is also slow.
    """

    def setUp(self):
        stack = Stack(TYPES)
        states = LinearStates(
            slots=4,
            num_v_heads=2,
            head_k_dim=2,
            head_v_dim=2,
            device=torch.device("cpu"),
        )
        self.stack = stack
        self.runner = RecordingRunner(stack, states, layer_types=TYPES, query_shift=1)
        residual = torch.randn(2, H)
        for i in range(2):
            self.runner.seed(i, residual[i])
            self.runner._gate[i] = torch.randn(1, H)
        self.runner.run([0, 1], 3, torch.randn(2, H), torch.tensor([7, 11]))

    def test_every_linear_layer_of_the_span_gets_its_read_sent_ahead(self):
        """Three of three on the decode path, which took three separate places to reach: the
        loop's own next layer, `ahead_of` for the one `_finish` runs, and `run` for the first,
        from the head layer's residual. Any of the three regressing leaves a feed-forward with
        nothing beside it and shows up only as a smaller gain.
        """
        issued = [layer for kind, layer in self.runner.schedule if kind == "issue"]
        # group 3's span is layers 3,4,5,6: the head is the full attention at 3 and
        # the linear ones after it are 4, 5 and 6
        self.assertEqual(issued, [4, 5, 6])
        # every layer still makes its own call. The early send does not replace it -- it is an
        # extra frame nobody waits for, and the far end has already contracted by the time the
        # call arrives. A layer that stopped calling would have stopped computing.
        asked = [layer for kind, layer in self.runner.schedule if kind == "ask"]
        self.assertEqual(asked, [4, 5, 6])

    def test_each_send_leaves_before_the_feed_forward_it_is_meant_to_hide_behind(self):
        """The whole claim, as an order: issue(L) precedes an mlp, which precedes ask(L).

        The mlp between them is the work being overlapped -- the far end contracts the state
        while it runs. Written the other way round, the span computes exactly the same tokens
        and hides nothing, which is the arrangement this replaced and the one a reordering
        would silently restore.
        """
        for layer in (4, 5, 6):
            issue = self.runner.schedule.index(("issue", layer))
            ask = self.runner.schedule.index(("ask", layer))
            self.assertLess(
                issue, ask, f"layer {layer} was called before its projection was sent"
            )
            between = [
                step
                for step in self.runner.schedule[issue + 1 : ask]
                if step[0] == "mlp"
            ]
            self.assertTrue(
                between,
                f"layer {layer}: nothing runs between its send and its call, so the far end "
                f"contracts with this side idle -- the schedule this replaced. "
                f"schedule={self.runner.schedule}",
            )

    def test_every_layer_that_was_sent_for_is_the_layer_that_calls(self):
        """The send and the call are chosen in different functions -- `_linear_run` for the
        middle layers, `run` for the first, `ahead_of` for the last -- so an off-by-one drifts
        them apart without raising anywhere. The far end would contract a real state and return
        a real vector, and the output is fluent text from the wrong history.
        """
        asked = [layer for kind, layer in self.runner.schedule if kind == "ask"]
        issued = [layer for kind, layer in self.runner.schedule if kind == "issue"]
        self.assertEqual(asked, issued)
