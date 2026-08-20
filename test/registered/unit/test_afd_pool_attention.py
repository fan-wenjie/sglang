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
        o_swept, lse_swept = sweep_cache(
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
            o, lse = sweep_cache(q, k[:, :end], v[:, :end], scaling=SCALE, kv_group=GROUP)
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
        o, lse = sweep_cache(q, k, v, scaling=SCALE, kv_group=GROUP)
        self.assertEqual(tuple(o.shape), (N_Q, HEAD_DIM))
        self.assertEqual(tuple(lse.shape), (N_Q,))
        with self.assertRaises(RuntimeError):
            sweep_cache(q, k, v, scaling=SCALE, kv_group=GROUP + 1)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
