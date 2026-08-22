"""Install Early-Q on the real Qwen3.8-27B stack and show it bites.

Needs the checkpoint and a GPU, so it is a manual test rather than a CI one:

    python test/srt/afd/test_early_q_end_to_end.py

Three things it establishes, in the order a failure would appear:

    coverage   the shift moves exactly the full-attention layers it can reach, and the layer that
               clamps is recorded as clamped rather than counted as converted
    mechanism  the QUERY moves and the key and value do not. Checked at the layer rather than at
               the logits: a logits delta shows only that something changed, which a hook on the
               wrong tensor also produces. This distinguishes the intended wiring from a wiring
               that reads the layer output -- a correct model, no overlap, and no error

What it deliberately does NOT do: claim the shifted model is as good. That is a quality question
the study answers with training runs, not something a forward pass can say.
"""

import json
import os
import unittest

import torch

MODEL = os.environ.get("AFD_MODEL", "/home/user/experiment/models/Qwen3.8-27B-FP8")
RECORD = "/home/user/experiment/sglang/afd_e2e.json"


def _build_runner(shift: int):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import PortArgs, ServerArgs

    server_args = ServerArgs(
        model_path=MODEL,
        tp_size=1,
        mem_fraction_static=0.70,
        disable_cuda_graph=True,
        attention_backend="triton",
        afd_q_shift_layers=shift,
    )
    port_args = PortArgs.init_new(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        ps=ParallelState.trivial(tp_size=1),
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    return runner, model_config


class TestEarlyQOnTheRealStack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs a GPU")
        if not os.path.isdir(MODEL):
            raise unittest.SkipTest(f"no checkpoint at {MODEL}")
        cls.runner, cls.model_config = _build_runner(shift=1)
        from sglang.srt.afd.read_point import layer_types_of

        cls.layer_types = layer_types_of(cls.runner.model)

    def test_the_plan_covers_the_softmax_layers_and_records_the_clamp(self):
        from sglang.srt.afd.read_point import full_attention_layers
        from sglang.srt.afd_query_shift.wiring import install_early_q

        full = full_attention_layers(self.layer_types)
        wiring = install_early_q(self.runner.model, 1, self.layer_types)
        try:
            record = wiring.record()
            # at shift 1 every softmax layer has an h_(l-1) beneath it, so none clamps
            self.assertEqual(record["n_considered"], len(full))
            self.assertEqual(record["n_moved"], len(full))
            self.assertEqual(record["n_clamped"], 0)
            for layer, source in record["sources"].items():
                self.assertEqual(int(layer) - source, 1)
        finally:
            wiring.remove()

    def test_a_group_shift_clamps_the_first_softmax_layer(self):
        from sglang.srt.afd.read_point import full_attention_layers
        from sglang.srt.afd_query_shift.wiring import install_early_q

        full = full_attention_layers(self.layer_types)
        wiring = install_early_q(self.runner.model, 4, self.layer_types)
        try:
            record = wiring.record()
            self.assertEqual(record["n_clamped"], 1, "layer 3 has no h_(-1) to read")
            self.assertEqual(record["clamped"], [full[0]])
            self.assertEqual(record["n_moved"], len(full) - 1)
            self.assertEqual(record["sources"]["7"], 3, "a group shift reads the last softmax layer")
        finally:
            wiring.remove()

    def test_the_query_moves_and_the_key_and_value_do_not(self):
        """The mechanism itself, on the real weights.

        Tested at the layer rather than at the logits because this is the claim: the query is
        projected from an earlier point of the residual stream and the key and value are not. A
        logits delta would show only that SOMETHING changed, which a hook on the wrong tensor
        also produces.
        """
        from sglang.srt.afd.read_point import full_attention_layers
        from sglang.srt.afd_query_shift.wiring import install_early_q

        target = full_attention_layers(self.layer_types)[4]     # a converted layer, mid-stack
        layer = self.runner.model.model.layers[target]
        hidden = self.runner.model.config.hidden_size if hasattr(
            self.runner.model, "config") else self.model_config.hf_text_config.hidden_size

        torch.manual_seed(0)
        x = torch.randn(16, hidden, device="cuda", dtype=torch.bfloat16)
        early = torch.randn(16, hidden, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(16, device="cuda")

        wiring = install_early_q(self.runner.model, 1, self.layer_types)
        try:
            with torch.no_grad():
                layer._afd_q_hidden = None
                q0, k0, v0, _ = layer.forward_prepare_native(
                    positions=positions, hidden_states=x
                )
                layer._afd_q_hidden = early
                q1, k1, v1, _ = layer.forward_prepare_native(
                    positions=positions, hidden_states=x
                )
                layer._afd_q_hidden = None
        finally:
            wiring.remove()

        dq = float((q0.float() - q1.float()).abs().max())
        dk = float((k0.float() - k1.float()).abs().max())
        dv = float((v0.float() - v1.float()).abs().max())
        print(f"    layer {target}: query moves by {dq:.3e}, key by {dk:.3e}, value by {dv:.3e}")
        self.assertGreater(dq, 0.0, "the query did not move: the hook is INERT")
        self.assertEqual(dk, 0.0, "the key moved; only the query's read point may change")
        self.assertEqual(dv, 0.0, "the value moved; only the query's read point may change")
        with open(RECORD, "w") as f:
            json.dump(
                {"model": MODEL, "layer": target, "d_query": dq, "d_key": dk, "d_value": dv},
                f,
                indent=2,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
