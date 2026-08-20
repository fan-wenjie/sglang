"""The split reproduces fused attention, on this model's shapes, before anything claims a speedup.

Check 3 in the afd-early-q skill, and the one that gates the others: a protocol that is 1e-2
relative to the attention it replaces is not the same model, and a throughput number measured on
it is a number for a different model.

What is checked, in the order a failure would appear:

    exactness    sweep-then-join against one fused call, in float64, where the only difference
                 permitted is the order of the additions
    dtype        the same in bfloat16, against an fp32 fused reference, to the precision bf16
                 actually has -- so a later regression is read as a regression and not as rounding
    absence      the sweep's signature does not contain the current value. That is the protocol:
                 if v_j were needed, nothing could start before the feed-forward that produces it
    sharding     a cache cut anywhere and merged in any order gives what one sweep would

The shapes come from Qwen3.8-27B's own config -- 24 query heads, 4 key-value heads, head_dim 256 --
because a protocol verified at a shape the model does not have is verified against nothing. They
are written out here rather than read from the checkpoint so the test runs without one.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.test.test_utils import CustomTestCase

import itertools
import unittest

import torch
from sglang.srt.afd.protocol import Frame, decode, encode

# Qwen3.8-27B text config: num_attention_heads 24, num_key_value_heads 4, head_dim 256.
N_Q_HEADS, N_KV_HEADS, HEAD_DIM = 24, 4, 256


def fused(q, k, v):
    """One attention call over every position, as a framework's kernel would compute it."""
    scale = q.shape[-1] ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    return torch.softmax(scores, dim=-1) @ v


def sweep(q_j, k_cached, v_cached):
    """The cache side. Returns (weighted sum, log partition) -- and never sees the current value."""
    scale = q_j.shape[-1] ** -0.5
    s = torch.einsum("hd,hjd->hj", q_j, k_cached) * scale
    lse = torch.logsumexp(s, dim=-1)
    return torch.einsum("hj,hjd->hd", torch.softmax(s, dim=-1), v_cached), lse


def join(o_lt, lse_lt, s_jj, v_j):
    """The compute side. A two-way softmax is a logistic, so the whole history is one pseudo-token."""
    w = torch.sigmoid(lse_lt - s_jj).unsqueeze(-1)
    return torch.lerp(v_j, o_lt, w)


def merge(states):
    """Combine sweeps of disjoint shards, in any order: (W, Z) adds componentwise."""
    o, lse = states[0]
    o, lse = o.clone(), lse.clone()
    for o_b, lse_b in states[1:]:
        m = torch.maximum(lse, lse_b)
        wa, wb = torch.exp(lse - m), torch.exp(lse_b - m)
        o = (o * wa.unsqueeze(-1) + o_b * wb.unsqueeze(-1)) / (wa + wb).unsqueeze(-1)
        lse = m + torch.log(wa + wb)
    return o, lse


def _qkv(positions, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(N_Q_HEADS, positions, HEAD_DIM, generator=g, dtype=dtype)
    k = torch.randn(N_KV_HEADS, positions, HEAD_DIM, generator=g, dtype=dtype)
    v = torch.randn(N_KV_HEADS, positions, HEAD_DIM, generator=g, dtype=dtype)
    # grouped-query: expand the key-value heads the way the kernel does
    rep = N_Q_HEADS // N_KV_HEADS
    return q, k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)


class TestSplitIsTheSameAttention(CustomTestCase):
    def test_exact_in_float64_at_this_model_shapes(self):
        worst = 0.0
        for positions in (16, 129, 1024):
            q, k, v = _qkv(positions)
            reference = fused(q, k, v)[:, -1]
            o_lt, lse_lt = sweep(q[:, -1], k[:, :-1], v[:, :-1])
            s_jj = (q[:, -1] * k[:, -1]).sum(-1) * HEAD_DIM**-0.5
            got = join(o_lt, lse_lt, s_jj, v[:, -1])
            worst = max(worst, float((reference - got).abs().max()))
        self.assertLess(worst, 1e-12, f"the split is not the same attention: {worst:.2e}")

    def test_bfloat16_agrees_to_the_precision_bfloat16_has(self):
        """Pinned so a later regression reads as a regression and not as rounding."""
        q, k, v = _qkv(512, dtype=torch.float64)
        reference = fused(q, k, v)[:, -1]
        qb, kb, vb = (t.to(torch.bfloat16).to(torch.float32) for t in (q, k, v))
        o_lt, lse_lt = sweep(qb[:, -1], kb[:, :-1], vb[:, :-1])
        s_jj = (qb[:, -1] * kb[:, -1]).sum(-1) * HEAD_DIM**-0.5
        got = join(o_lt, lse_lt, s_jj, vb[:, -1])
        rel = float((reference - got.double()).abs().max() / reference.abs().max())
        self.assertLess(rel, 5e-2, f"bf16 split diverges by {rel:.2e} relative")

    def test_the_sweep_never_sees_the_current_value(self):
        """The protocol itself. If v_j were needed, nothing could start before its feed-forward."""
        q, k, v = _qkv(64)
        a = sweep(q[:, -1], k[:, :-1], v[:, :-1])
        v_moved = v.clone()
        v_moved[:, -1] += 1000.0                       # change only the current value
        b = sweep(q[:, -1], k[:, :-1], v_moved[:, :-1])
        self.assertTrue(torch.equal(a[0], b[0]), "the sweep moved when only v_j changed")
        self.assertTrue(torch.equal(a[1], b[1]))

    def test_any_cut_in_any_order_gives_one_sweep(self):
        """A cache split across devices, merged in any order, is the cache."""
        q, k, v = _qkv(400)
        whole = sweep(q[:, -1], k[:, :-1], v[:, :-1])
        worst = 0.0
        for cuts in ([0, 1, 399], [0, 200, 399], [0, 100, 250, 399], [0, 399]):
            edges = sorted(set(cuts))
            parts = [sweep(q[:, -1], k[:, a:b], v[:, a:b])
                     for a, b in itertools.pairwise(edges) if b > a]
            for order in itertools.permutations(range(len(parts))):
                o, lse = merge([parts[i] for i in order])
                worst = max(worst, float((o - whole[0]).abs().max()),
                            float((lse - whole[1]).abs().max()))
        self.assertLess(worst, 1e-12, f"a split cache disagreed with one sweep by {worst:.2e}")

    def test_a_boundary_off_by_one_is_not_a_rounding(self):
        """What the runtime verify number has to be read against.

        `--afd-verify-split` reports how far the partitioned attention sits from the fused one on
        real traffic, and a small number only means something if a WRONG partition would have
        given a large one. The realistic mistake is at the boundary: the sweep is meant to cover
        every cached position but this step's, and an off-by-one drops the position just before
        it -- the one a decode attends to most.

        Both are measured here in float64, where the honest split has nothing but summation order
        between it and the fused answer, so the two failure modes cannot be confused with each
        other or with the arithmetic.
        """
        torch.manual_seed(0)
        cached = 32
        q = torch.randn(N_Q_HEADS, HEAD_DIM, dtype=torch.float64)
        k = torch.randn(N_Q_HEADS, cached + 1, HEAD_DIM, dtype=torch.float64)
        v = torch.randn(N_Q_HEADS, cached + 1, HEAD_DIM, dtype=torch.float64)
        reference = fused(q.unsqueeze(1), k, v).squeeze(1)
        scale = HEAD_DIM**-0.5

        def merged(prefix_end):
            o_lt, lse_lt = sweep(q, k[:, :prefix_end], v[:, :prefix_end])
            s_jj = (q * k[:, -1]).sum(-1) * scale
            return join(o_lt, lse_lt, s_jj, v[:, -1])

        honest = float((merged(cached) - reference).abs().max() / reference.abs().max())
        short = float((merged(cached - 1) - reference).abs().max() / reference.abs().max())
        print(f"    honest {honest:.2e}   one position short {short:.2e}")
        self.assertLess(honest, 1e-12, "a partition of the cache reproduces the whole of it")
        self.assertGreater(
            short, 1e-2,
            "a dropped position must be visible; if it is not, the runtime verify number is "
            "measuring nothing and a wrong boundary would pass it",
        )

    def test_counting_a_position_twice_is_not_a_rounding(self):
        """The other boundary error: the sweep keeps this step's slot, which the join adds again.

        This one is worth its own case because it is what a partition built from the FULL sequence
        length produces, and that length is the one every other part of the backend uses.
        """
        torch.manual_seed(1)
        cached = 32
        q = torch.randn(N_Q_HEADS, HEAD_DIM, dtype=torch.float64)
        k = torch.randn(N_Q_HEADS, cached + 1, HEAD_DIM, dtype=torch.float64)
        v = torch.randn(N_Q_HEADS, cached + 1, HEAD_DIM, dtype=torch.float64)
        reference = fused(q.unsqueeze(1), k, v).squeeze(1)
        o_lt, lse_lt = sweep(q, k, v)                       # the whole run, current token included
        s_jj = (q * k[:, -1]).sum(-1) * HEAD_DIM**-0.5
        double = join(o_lt, lse_lt, s_jj, v[:, -1])
        gap = float((double - reference).abs().max() / reference.abs().max())
        print(f"    one position counted twice {gap:.2e}")
        self.assertGreater(gap, 1e-2)

    def test_a_frame_of_this_model_width_survives_the_wire(self):
        """The hidden width the pool actually carries, not a toy one."""
        import socket

        a, b = socket.socketpair()
        hidden = torch.randn(7, 5120).to(torch.bfloat16)      # Qwen3.8-27B hidden_size
        a.sendall(encode(Frame.one(42, 7, hidden)))
        got = decode(b)
        self.assertTrue(torch.equal(got.tensor, hidden))
        self.assertEqual(got.key, (42, 7))


if __name__ == "__main__":
    unittest.main()
