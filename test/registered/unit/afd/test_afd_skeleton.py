"""The attention service: a host built from the numbers, tested against no model.

The skeleton is the host's answer to `host_model: skeleton` in the pushed
configuration -- layers from the layer kinds, widths from the config, no family
class anywhere. Pinned here: the tree the routing needs is all there (the kinds
read off the class names, the convolution's storage, the final norm), the
pool's weight push lands in it through the real wire, and a process that is
not an AFD host never builds it by accident.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import types

import torch

from sglang.test.test_utils import CustomTestCase

from sglang.srt.afd import model_files
from sglang.srt.afd.model_files import files_reply, papers_from_reply
from sglang.srt.afd.skeleton import AttentionServiceStack, skeleton_wanted


def _manifest_from(config):
    """What the pool would derive, written from the config for a model-less test."""
    text = config.text_config
    heads = text.num_attention_heads
    head_dim = text.head_dim or text.hidden_size // heads
    layers = []
    for kind in text.layers_block_type:
        if kind in ("attention", "full_attention"):
            layers.append(
                {
                    "kind": "softmax_kv",
                    "heads": heads,
                    "head_dim": head_dim,
                    "kv_heads": text.num_key_value_heads,
                    "v_head_dim": head_dim,
                    "scaling": head_dim**-0.5,
                }
            )
        else:
            layers.append(
                {
                    "kind": "gated_delta",
                    "k_heads": text.linear_num_key_heads,
                    "v_heads": text.linear_num_value_heads,
                    "dk": text.linear_key_head_dim,
                    "dv": text.linear_value_head_dim,
                    "conv_taps": text.linear_conv_kernel_dim,
                }
            )
    return {
        "layers": layers,
        "hidden_size": text.hidden_size,
        "norm_eps": text.rms_norm_eps,
        "vocab_size": text.vocab_size,
        "positions": "mrope",
    }


MP = "/home/user/experiment/models/Qwen3.8-27B"


def _pushed_config():
    import os

    if not os.path.isdir(MP):
        raise unittest.SkipTest("the reference checkpoint is not on this machine")
    model_files._PAPERS["fake:9"] = papers_from_reply(files_reply(MP))
    try:
        from sglang.srt.utils.hf_transformers.config import get_config

        return get_config("pool://fake:9", trust_remote_code=False)
    finally:
        model_files._PAPERS.pop("fake:9", None)


class TestTheAttentionService(CustomTestCase):
    def test_the_tree_is_what_the_routing_reads(self):
        from sglang.srt.afd.layer_kinds import layer_types_of

        config = _pushed_config()
        stack = AttentionServiceStack(_manifest_from(config))
        kinds = layer_types_of(types.SimpleNamespace(model=stack))
        text = config.text_config
        spoken = [
            "full_attention" if k in ("attention", "full_attention") else k
            for k in text.layers_block_type
        ]
        self.assertEqual(kinds, spoken)
        self.assertEqual(len(kinds), 64)
        self.assertEqual(kinds.count("full_attention"), 16)

        conv_dim = 2 * text.linear_num_key_heads * text.linear_key_head_dim + (
            text.linear_num_value_heads * text.linear_value_head_dim
        )
        first_linear = kinds.index("linear_attention")
        conv = stack.layers[first_linear].linear_attn.conv1d
        self.assertEqual(
            tuple(conv.weight.shape),
            (conv_dim, 1, text.linear_conv_kernel_dim),
        )
        self.assertIsNone(conv.bias)

        first_full = kinds.index("full_attention")
        attn = stack.layers[first_full].attn
        self.assertEqual(attn.layer_id, first_full)
        self.assertEqual(stack.norm.weight.shape[0], text.hidden_size)

    def test_the_push_lands_in_the_skeleton(self):
        # the pool's residual weights, through the REAL wire, into a model that
        # never was one
        import socket
        import threading

        from sglang.srt.afd.installer import _pull_residual_weights
        from sglang.srt.afd.pool_client import PoolClient
        from sglang.srt.afd.protocol import OP_WEIGHTS, Frame, decode, send_frame

        config = _pushed_config()
        stack = AttentionServiceStack(_manifest_from(config))
        model = types.SimpleNamespace(model=stack)
        model.parameters = stack.parameters

        linear = [
            i
            for i, layer in enumerate(stack.layers)
            if "Linear" in type(layer).__name__
        ]
        pushed = [
            torch.full((1, stack.layers[i].linear_attn.conv1d.weight.numel()), 2.0)
            for i in linear
            for _ in (0,)
        ]
        reply = []
        for t in pushed:
            reply.append(t)
            reply.append(torch.zeros(1, 0))
        reply.append(torch.full((1, stack.norm.weight.numel()), 3.0))

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)

        def serve():
            sock, _ = server.accept()
            frame = decode(sock)
            send_frame(sock, Frame(frame.request_id, 0, tuple(reply), OP_WEIGHTS))

        threading.Thread(target=serve, daemon=True).start()
        client = PoolClient(
            f"127.0.0.1:{server.getsockname()[1]}", 10.0, reconnect=False
        )
        try:
            _pull_residual_weights(model, client)
        finally:
            client.close()
            server.close()
        self.assertTrue(
            torch.equal(
                stack.layers[linear[0]].linear_attn.conv1d.weight,
                torch.full_like(stack.layers[linear[0]].linear_attn.conv1d.weight, 2.0),
            )
        )
        self.assertTrue(
            torch.equal(stack.norm.weight, torch.full_like(stack.norm.weight, 3.0))
        )

    def test_the_manifest_is_read_off_the_built_modules(self):
        import json

        import torch.nn as nn

        from sglang.srt.afd.manifest import manifest_of

        class ToyLinearLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_attn = nn.Module()
                for name, value in (
                    ("num_k_heads", 2),
                    ("num_v_heads", 4),
                    ("head_k_dim", 8),
                    ("head_v_dim", 8),
                    ("conv_kernel_size", 3),
                ):
                    setattr(self.linear_attn, name, value)

        class ToyAttentionLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = types.SimpleNamespace(
                    tp_q_head_num=6,
                    head_dim=8,
                    tp_k_head_num=2,
                    v_head_dim=8,
                    scaling=8**-0.5,
                )

        model = types.SimpleNamespace()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([ToyLinearLayer(), ToyAttentionLayer()])
        norm = nn.Module()
        norm.weight = nn.Parameter(torch.zeros(16))
        norm.variance_epsilon = 1e-6
        model.model.norm = norm
        model.lm_head = types.SimpleNamespace(weight=torch.zeros(32, 16))
        model.is_mrope_enabled = True

        manifest = manifest_of(model)
        json.dumps(manifest)  # rides the pushed configuration: JSON or nothing
        self.assertEqual(
            [s["kind"] for s in manifest["layers"]], ["gated_delta", "softmax_kv"]
        )
        self.assertEqual(manifest["layers"][0]["conv_taps"], 3)
        self.assertEqual(manifest["layers"][1]["heads"], 6)
        self.assertEqual(manifest["hidden_size"], 16)
        self.assertEqual(manifest["vocab_size"], 32)
        self.assertEqual(manifest["positions"], "mrope")

        stack = AttentionServiceStack(manifest)
        conv = stack.layers[0].linear_attn.conv1d
        self.assertEqual(tuple(conv.weight.shape), (2 * 2 * 8 + 4 * 8, 1, 3))
        self.assertEqual(stack.layers[1].attn.tp_q_head_num, 6)

    def test_a_process_that_is_no_host_never_builds_it(self):
        self.assertFalse(skeleton_wanted())

    def test_the_pool_decides_the_build(self):
        from sglang.srt.afd.pushed_config import pool_config

        base = dict(model_path="/x/Qwen", afd_transfer_backend=None)
        family = pool_config(types.SimpleNamespace(**base, afd_host_skeleton=False))
        skel = pool_config(types.SimpleNamespace(**base, afd_host_skeleton=True))
        self.assertEqual(family["host_model"], "family")
        self.assertEqual(skel["host_model"], "skeleton")


if __name__ == "__main__":
    unittest.main()
