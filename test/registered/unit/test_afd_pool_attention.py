"""The pool holds the cache and sweeps it; the host folds in this step's token. Same answer?

This is the gate on moving the KV cache across the wire. The split is the same algebra the
colocated partition already uses -- softmax is a mergeable aggregate -- but the halves now run in
different processes on different cards, and the half that used to be a kernel call is now a
GEMM over a cache the host cannot see. If the two do not agree here, nothing downstream can tell:
the output stays a plausible attention over the wrong set.

Shapes are Qwen3.8-27B's own: 24 query heads, 4 key-value heads, head_dim 256, so the
grouped-query expansion (6 query heads per key-value head) is exercised rather than assumed away.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch
def _one(q, k, v, **kw):
    """These cases were written for a single token, and sweep_cache now takes a batch of them.

    Kept as a shim rather than rewritten because what the cases assert -- that a boundary error is
    not a rounding error, that the group size is checked -- is about the arithmetic and not about
    the batch axis, and adding a leading 1 to every tensor in every case would bury that.
    """
    from sglang.srt.afd.pool_attention import sweep_cache as _sweep

    o, lse = _sweep(q.unsqueeze(0), k, v, **kw)
    return o[0], lse[0]


from sglang.srt.afd.pool_attention import KVHolder, sweep_cache
from sglang.test.test_utils import CustomTestCase

N_Q, N_KV, HEAD_DIM = 24, 4, 256
GROUP = N_Q // N_KV
SCALE = HEAD_DIM**-0.5


def fused(q, k, v):
    """One attention call over every position, grouped-query, as a kernel would compute it."""
    k = k.repeat_interleave(GROUP, dim=0)
    v = v.repeat_interleave(GROUP, dim=0)
    scores = torch.einsum("hd,hjd->hj", q, k) * SCALE
    return torch.einsum("hj,hjd->hd", torch.softmax(scores, dim=-1), v)


def join(o_swept, lse_swept, q, k_now, v_now):
    """The host's half: fold this step's token in. A two-way softmax is a logistic."""
    k = k_now.repeat_interleave(GROUP, dim=0)
    v = v_now.repeat_interleave(GROUP, dim=0)
    s = (q * k).sum(-1) * SCALE
    return torch.lerp(v, o_swept, torch.sigmoid(lse_swept - s).unsqueeze(-1))


class TestPoolSweepPlusHostJoin(CustomTestCase):
    def test_the_two_halves_equal_one_fused_attention(self):
        torch.manual_seed(0)
        cached = 40
        q = torch.randn(N_Q, HEAD_DIM, dtype=torch.float64)
        k = torch.randn(N_KV, cached + 1, HEAD_DIM, dtype=torch.float64)
        v = torch.randn(N_KV, cached + 1, HEAD_DIM, dtype=torch.float64)

        reference = fused(q, k, v)
        o_swept, lse_swept = _one(
            q, k[:, :cached], v[:, :cached], scaling=SCALE, kv_group=GROUP
        )
        merged = join(o_swept, lse_swept, q, k[:, -1], v[:, -1])
        gap = float((merged - reference).abs().max() / reference.abs().max())
        print(f"    pool sweep + host join vs fused: {gap:.2e}")
        self.assertLess(gap, 1e-12)

    def test_a_boundary_error_is_not_a_rounding(self):
        """What the exactness number has to be read against: the realistic mistake is the pool
        sweeping one position too many or too few, and that must be visibly larger."""
        torch.manual_seed(1)
        cached = 40
        q = torch.randn(N_Q, HEAD_DIM, dtype=torch.float64)
        k = torch.randn(N_KV, cached + 1, HEAD_DIM, dtype=torch.float64)
        v = torch.randn(N_KV, cached + 1, HEAD_DIM, dtype=torch.float64)
        reference = fused(q, k, v)

        for name, end in (("one short", cached - 1), ("one long", cached + 1)):
            o, lse = _one(q, k[:, :end], v[:, :end], scaling=SCALE, kv_group=GROUP)
            merged = join(o, lse, q, k[:, -1], v[:, -1])
            gap = float((merged - reference).abs().max() / reference.abs().max())
            print(f"    sweep {name}: {gap:.2e}")
            self.assertGreater(gap, 1e-2, f"a {name} sweep must not look like rounding")

    def test_the_grouped_query_expansion_is_the_models_own(self):
        """A cache stores what the projection produced -- 4 key-value heads, not 24. Expanding it
        the wrong way round gives every query head the same key and no test of the logits would
        obviously fail."""
        q = torch.randn(N_Q, HEAD_DIM, dtype=torch.float64)
        k = torch.randn(N_KV, 8, HEAD_DIM, dtype=torch.float64)
        v = torch.randn(N_KV, 8, HEAD_DIM, dtype=torch.float64)
        o, lse = _one(q, k, v, scaling=SCALE, kv_group=GROUP)
        self.assertEqual(tuple(o.shape), (N_Q, HEAD_DIM))
        self.assertEqual(tuple(lse.shape), (N_Q,))
        with self.assertRaises(RuntimeError):
            _one(q, k, v, scaling=SCALE, kv_group=GROUP + 1)


class TestKVHolder(CustomTestCase):
    def test_appending_keeps_step_order(self):
        """The sweep reads positions in time order; a cache that appended out of order would
        still produce a plausible attention over a shuffled history."""
        holder = KVHolder("cpu", max_context=16)
        steps = [torch.full((N_KV, 1, HEAD_DIM), float(i)) for i in range(5)]
        for i, s in enumerate(steps):
            k_all, _ = holder.append(1, 0, s, s)
            self.assertEqual(k_all.shape[1], i + 1)
        self.assertEqual([float(k_all[0, j, 0]) for j in range(5)], [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_two_requests_do_not_share_a_cache(self):
        holder = KVHolder("cpu", max_context=16)
        one = torch.ones(N_KV, 1, HEAD_DIM)
        holder.append(1, 0, one, one)
        holder.append(1, 0, one, one)
        k_other, _ = holder.append(2, 0, one * 9, one * 9)
        self.assertEqual(holder.positions(1, 0), 2)
        self.assertEqual(k_other.shape[1], 1, "request 2 must not inherit request 1's history")

    def test_two_layers_of_one_request_do_not_share_a_cache(self):
        holder = KVHolder("cpu", max_context=16)
        one = torch.ones(N_KV, 1, HEAD_DIM)
        holder.append(1, 0, one, one)
        k_layer_3, _ = holder.append(1, 3, one, one)
        self.assertEqual(k_layer_3.shape[1], 1)

    def test_running_past_the_room_refuses_rather_than_overwrites(self):
        """An append-only cache cannot evict. Overwriting would serve attention over a history
        with a hole in it, which reads as a fluent model that has forgotten the middle."""
        holder = KVHolder("cpu", max_context=3)
        one = torch.ones(N_KV, 1, HEAD_DIM)
        for _ in range(3):
            holder.append(1, 0, one, one)
        with self.assertRaises(RuntimeError):
            holder.append(1, 0, one, one)

    def test_release_frees_every_layer_of_one_request_only(self):
        holder = KVHolder("cpu", max_context=8)
        one = torch.ones(N_KV, 1, HEAD_DIM)
        for layer in (0, 3, 7):
            holder.append(1, layer, one, one)
        holder.append(2, 0, one, one)
        self.assertEqual(holder.release(1), 3)
        self.assertEqual(holder.positions(1, 0), 0)
        self.assertEqual(holder.positions(2, 0), 1, "another request's cache must survive")


class TestTheBatchedSweepMatchesThePerTokenOne(CustomTestCase):
    """The rewrite that removed the token loop and the group materialisation must not have moved
    a single bit.

    The loop sliced the cache once per token to give each its causal boundary; the replacement
    states the boundary as a mask over one contracted score tensor. Those are the same arithmetic
    only if the masked positions contribute exactly nothing -- and a mask applied AFTER the scaling
    or before the logsumexp in the wrong order would still produce plausible attention, so this
    compares against the slicing form rather than against a tolerance.

    It also guards the reason the rewrite happened: the loop plus repeat_interleave materialised
    on the order of two hundred gigabytes a step at 32k context, and the reversed arrangement's
    first end-to-end measurement was a measurement of that.
    """

    def _slicewise(self, q3, k, v, seen, scaling, kv_group):
        from sglang.srt.afd.pool_attention import sweep_cache

        outs, lses = [], []
        for t in range(q3.shape[0]):
            end = int(seen[t])
            o, lse = sweep_cache(q3[t : t + 1], k[:, :end], v[:, :end],
                                 scaling=scaling, kv_group=kv_group)
            outs.append(o[0])
            lses.append(lse[0])
        return torch.stack(outs), torch.stack(lses)

    def test_a_chunk_with_staggered_boundaries_matches_slicing(self):
        from sglang.srt.afd.pool_attention import sweep_cache

        torch.manual_seed(0)
        kv_heads, group, dim, positions, tokens = 2, 3, 8, 11, 4
        heads = kv_heads * group
        q3 = torch.randn(tokens, heads, dim, dtype=torch.float64)
        k = torch.randn(kv_heads, positions, dim, dtype=torch.float64)
        v = torch.randn(kv_heads, positions, dim, dtype=torch.float64)
        seen = torch.arange(positions - tokens + 1, positions + 1, dtype=torch.long)

        o_batched, lse_batched = sweep_cache(q3, k, v, scaling=0.125, kv_group=group, seen=seen)
        o_sliced, lse_sliced = self._slicewise(q3, k, v, seen, 0.125, group)

        # Not bit-identical, and it should not be: a softmax over a masked full-width row adds its
        # terms in a different order from one over a sliced row, which is the same reassociation
        # the split itself performs. What matters is the SCALE. In float64 that reassociation is
        # around 1e-16, while the case above shows a boundary off by one position landing near
        # 1e-1 -- fifteen orders apart, so a tolerance here cannot hide a boundary error.
        o_gap = (o_batched - o_sliced).abs().max().item()
        lse_gap = (lse_batched - lse_sliced).abs().max().item()
        self.assertLess(o_gap, 1e-12, f"outputs differ by {o_gap}, which is not reassociation")
        self.assertLess(lse_gap, 1e-12, f"log partitions differ by {lse_gap}")

    def test_a_token_that_may_read_nothing_gets_an_empty_sum(self):
        """Its output is zero and its partition -inf, so the host's join takes its token whole.

        The masked scores are all -inf for such a token, and softmax over an all -inf row is nan.
        A nan here would reach the merge and poison a request's whole generation, silently, since
        nothing downstream checks.
        """
        from sglang.srt.afd.pool_attention import sweep_cache

        torch.manual_seed(0)
        kv_heads, group, dim, positions = 2, 3, 8, 5
        q3 = torch.randn(2, kv_heads * group, dim, dtype=torch.float64)
        k = torch.randn(kv_heads, positions, dim, dtype=torch.float64)
        v = torch.randn(kv_heads, positions, dim, dtype=torch.float64)

        o, lse = sweep_cache(q3, k, v, scaling=0.125, kv_group=group,
                             seen=torch.tensor([0, 3], dtype=torch.long))
        self.assertFalse(torch.isnan(o).any(), "an empty sum produced nan rather than zero")
        self.assertTrue(torch.all(o[0] == 0.0))
        self.assertTrue(torch.all(torch.isinf(lse[0]) & (lse[0] < 0)))
        self.assertFalse(torch.isnan(o[1]).any())

    def test_the_group_is_not_materialised(self):
        """A key head serves its whole group by contraction, so the peak allocation does not scale
        with the group size.

        Asserted on the result rather than on memory: the shared key head must give every query
        head in its group the same key, which repeat_interleave also did -- so this pins the
        semantics the cheap path has to preserve, and the cost is what the rewrite was for.
        """
        from sglang.srt.afd.pool_attention import sweep_cache

        torch.manual_seed(0)
        kv_heads, group, dim, positions = 1, 4, 6, 7
        q3 = torch.randn(1, kv_heads * group, dim, dtype=torch.float64)
        # every query head in the group gets the same query, so every one must get the same answer
        q3[0, :] = q3[0, 0]
        k = torch.randn(kv_heads, positions, dim, dtype=torch.float64)
        v = torch.randn(kv_heads, positions, dim, dtype=torch.float64)
        o, _ = sweep_cache(q3, k, v, scaling=0.125, kv_group=group)
        for head in range(1, group):
            self.assertTrue(torch.equal(o[0, head], o[0, 0]),
                            "heads sharing a key head and a query disagreed")



if __name__ == "__main__":
    unittest.main()
