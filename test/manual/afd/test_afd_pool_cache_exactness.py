"""The pool's cache and sweep, on the real weights, against attention computed directly.

    python test/manual/afd/test_afd_pool_cache_exactness.py

Needs the checkpoint and a GPU, so it is manual rather than registered. What it establishes is the
one thing the CPU algebra test cannot: that the key and value the POOL projects, through the
model's own fused projection with its qk-norm and its rotation, accumulate in the right order and
sweep to the same answer as attention taken directly over that history.

The failure this guards is not a crash. A projection that dropped the rotation, or a cache that
paired a key with another position's value, gives a fluent model attending to a past it never had.

Note what is NOT guarded, because the first version of this file asserted it and was wrong:
attention is permutation-invariant over cached positions. softmax(qK')V is a sum, so reversing the
cache changes nothing -- 3.8e-16 when measured. Order does not live in the cache; it lives in the
rotation already applied to each key. So the controls below break the two things that DO carry
information: the key-value pairing, and the positions the rotation was applied at.
"""

import os
import sys
import unittest

import torch

MODEL = os.environ.get("AFD_MODEL", "/home/user/experiment/models/Qwen3.8-27B-FP8")
STEPS = 6


def _runner():
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import PortArgs, ServerArgs

    args = ServerArgs(model_path=MODEL, tp_size=1, mem_fraction_static=0.70,
                      disable_cuda_graph=True, attention_backend="triton")
    port_args = PortArgs.init_new(args)
    config = ModelConfig.from_server_args(args)
    return ModelRunner(model_config=config, mem_fraction_static=args.mem_fraction_static,
                       gpu_id=0, ps=ParallelState.trivial(tp_size=1),
                       nccl_port=port_args.nccl_port, server_args=args)


class TestPoolCacheOnRealWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not os.path.isdir(MODEL):
            raise unittest.SkipTest("needs a GPU and the checkpoint")
        cls.runner = _runner()

    def test_a_decode_stream_sweeps_the_history_it_actually_wrote(self):
        from sglang.srt.afd.pool_attention import KVHolder, SweepService, sweep_cache
        from sglang.srt.afd.read_point import full_attention_layers, layer_types_of
        from sglang.srt.afd.remote_attention import join

        model = self.runner.model
        types = layer_types_of(model)
        layer_id = full_attention_layers(types)[2]
        layer = model.model.layers[layer_id]
        attn = layer.attn
        heads, kv_heads = attn.tp_q_head_num, attn.tp_k_head_num
        head_dim, v_head_dim = attn.qk_head_dim, attn.v_head_dim
        hidden_size = model.config.hidden_size

        holder = KVHolder("cuda", max_context=64)
        service = SweepService(model, holder, types)

        torch.manual_seed(0)
        history_k, history_v, worst = [], [], 0.0
        for step in range(STEPS):
            hidden = torch.randn(1, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.05
            q = torch.randn(1, heads * head_dim, device="cuda", dtype=torch.bfloat16) * 0.05
            positions = torch.tensor([step], device="cuda")

            normed = model.model.layers[layer_id].input_layernorm(hidden)
            o_swept, lse, score, v_now = service.serve_attention(1, layer_id, normed, positions)
            got = join(o_swept, lse, score, v_now, heads=heads, kv_heads=kv_heads,
                       v_head_dim=v_head_dim, dtype=torch.bfloat16)
            with torch.no_grad():
                q, k_now, _, _ = model.model.layers[layer_id].forward_prepare_native(
                    positions=positions, hidden_states=normed)

            history_k.append(k_now.view(kv_heads, 1, head_dim).double())
            history_v.append(v_now.view(kv_heads, 1, v_head_dim).double())
            k_all = torch.cat(history_k, dim=1)
            v_all = torch.cat(history_v, dim=1)
            want, _ = sweep_cache(q.view(heads, head_dim).double(), k_all, v_all,
                                  scaling=attn.scaling, kv_group=heads // kv_heads)
            want = want.reshape(1, heads * v_head_dim)

            gap = float((got.double() - want).abs().max() / want.abs().max())
            worst = max(worst, gap)
            print(f"    step {step}: {k_all.shape[1]} cached position(s), relative gap {gap:.2e}")
        print(f"    worst over {STEPS} steps: {worst:.2e}")
        self.assertLess(worst, 5e-2, "the pool's history is not the history it swept")

    def test_reversing_the_history_changes_nothing_and_that_is_the_point(self):
        """Pinned because it is counter-intuitive and it decides what the cache must guarantee.

        A sum does not care about the order of its terms. Whatever the cache must preserve, it is
        not the order it was written in -- so a test that asserted a reversed cache differs would
        have been asserting something false, and passing it would have meant the sweep was wrong.
        """
        from sglang.srt.afd.pool_attention import sweep_cache

        torch.manual_seed(1)
        k = torch.randn(4, 6, 256, dtype=torch.float64)
        v = torch.randn(4, 6, 256, dtype=torch.float64)
        q = torch.randn(24, 256, dtype=torch.float64)
        forward, _ = sweep_cache(q, k, v, scaling=256**-0.5, kv_group=6)
        backward, _ = sweep_cache(q, k.flip(1), v.flip(1), scaling=256**-0.5, kv_group=6)
        gap = float((forward - backward).abs().max() / forward.abs().max())
        print(f"    history reversed (k and v together): {gap:.2e}  -- invariant, as it must be")
        self.assertLess(gap, 1e-12)

    def test_breaking_the_key_value_pairing_is_visible(self):
        """The real ordering risk. Reversing k while leaving v gives every key another position's
        value -- the cache is the same size, every tensor is the right shape, and the attention is
        over a history that never existed."""
        from sglang.srt.afd.pool_attention import sweep_cache

        torch.manual_seed(1)
        k = torch.randn(4, 6, 256, dtype=torch.float64)
        v = torch.randn(4, 6, 256, dtype=torch.float64)
        q = torch.randn(24, 256, dtype=torch.float64)
        right, _ = sweep_cache(q, k, v, scaling=256**-0.5, kv_group=6)
        crossed, _ = sweep_cache(q, k.flip(1), v, scaling=256**-0.5, kv_group=6)
        gap = float((right - crossed).abs().max() / right.abs().max())
        print(f"    key and value crossed: {gap:.2e}")
        self.assertGreater(gap, 1e-2)

    def test_projecting_at_the_wrong_positions_is_visible(self):
        """The other one, on the real weights: the pool applies the rotation, so it needs the
        host's positions. Sending the wrong ones is the bug that makes order matter."""
        from sglang.srt.afd.pool_attention import KVHolder, SweepService, sweep_cache
        from sglang.srt.afd.read_point import full_attention_layers, layer_types_of

        model = self.runner.model
        types = layer_types_of(model)
        layer_id = full_attention_layers(types)[2]
        attn = model.model.layers[layer_id].attn
        hidden_size = model.config.hidden_size

        torch.manual_seed(3)
        hidden = torch.randn(1, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.05
        service = SweepService(model, KVHolder("cuda", 8), types)
        normed = model.model.layers[layer_id].input_layernorm(hidden)
        _, _, score_at_0, _ = service.serve_attention(
            1, layer_id, normed, torch.tensor([0], device="cuda"))
        service_b = SweepService(model, KVHolder("cuda", 8), types)
        _, _, score_at_37, _ = service_b.serve_attention(
            2, layer_id, normed, torch.tensor([37], device="cuda"))
        gap = float((score_at_0.double() - score_at_37.double()).abs().max()
                    / score_at_0.double().abs().max())
        print(f"    same hidden, positions 0 vs 37: {gap:.2e}")
        self.assertGreater(gap, 1e-2, "the rotation is not being applied at the position given")


if __name__ == "__main__":
    unittest.main(verbosity=2)
