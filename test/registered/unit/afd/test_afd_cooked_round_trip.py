"""One token through the push-arrangement span, both halves in one process, no sockets.

The pool's half runs for real -- the tiny stack's linear attention is the model's own
GatedDeltaNet -- and the host's half runs for real too: `issue_host` hands the cooked
coefficient straight to `contract_early` and files the reading it answers, `collect_early`
serves it back, and `apply_host` hands the finished step to `apply_advance`, against a history
cache seeded with distinct states. What the sockets would carry is exactly what crosses the
stubs, so the plumbing this checks is the plumbing that failed on the first live pair: shapes
on the wire, the window mapping, the early-collect-apply order per layer, and the ring
advancing under the layer that convolved against it.
"""

import unittest
from types import SimpleNamespace

import torch
from afd_tiny_stack import build_tiny_stack

from sglang.srt.afd.linear_history import HistoryCache
from sglang.srt.afd.protocol import OP_STATE_APPLY, OP_STATE_EARLY
from sglang.srt.afd_query_shift.early_contraction import apply_advance, contract_early
from sglang.test.test_utils import CustomTestCase

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "the tiny stack's fused norms have no CPU kernel")
class TestOneTokenRoundTrip(CustomTestCase):
    def setUp(self):
        from sglang.srt.afd.linear_state import LinearStates
        from sglang.srt.afd.span import SpanRunner

        self.stack, self.config, kinds = build_tiny_stack(device="cuda")
        self.kinds = kinds
        states = LinearStates(
            slots=4,
            num_v_heads=self.config.linear_num_value_heads,
            head_k_dim=self.config.linear_key_head_dim,
            head_v_dim=self.config.linear_value_head_dim,
            device=torch.device("cuda"),
        )
        self.runner = SpanRunner(self.stack, states, layer_types=kinds, query_shift=1)
        width = (
            2 * self.config.linear_num_key_heads * self.config.linear_key_head_dim
            + self.config.linear_num_value_heads * self.config.linear_value_head_dim
        )
        taps = int(self.config.linear_conv_kernel_dim)
        self.service = SimpleNamespace(
            cache=HistoryCache(
                slots=4,
                layers=len(kinds),
                value_heads=self.config.linear_num_value_heads,
                head_k_dim=self.config.linear_key_head_dim,
                head_v_dim=self.config.linear_value_head_dim,
                device=torch.device("cuda"),
                conv_width=width,
                conv_taps=taps,
            ),
            dims=(
                self.config.linear_num_key_heads,
                self.config.linear_num_value_heads,
                self.config.linear_key_head_dim,
                self.config.linear_value_head_dim,
            ),
            _parked=None,
            reads=0,
            updates=0,
            rows_of=lambda frame: self._rows,
        )
        self.service.cache.state.normal_()
        self.service.cache.conv.normal_().mul_(0.1)
        self._rows = [3]
        self.ops = []

        self.readings = {}

        def issue_host(layer_id, request_ids, q_tilde):
            frame = SimpleNamespace(
                request_id=9,
                layer=layer_id,
                tensors=(q_tilde.reshape(len(request_ids), -1).detach().cpu(),),
                op=OP_STATE_EARLY,
            )
            # the far end answers straight away; the reply table here is a dict
            (reading,) = contract_early(self.service, frame)
            self.readings[int(layer_id)] = reading
            self.ops.append(("early", int(layer_id)))
            return [int(layer_id)]

        def collect_early(pending, rows):
            (lid,) = pending
            self.ops.append(("collect", int(lid)))
            return self.readings.pop(int(lid)).to("cuda").float()

        def apply_host(layer_id, request_ids, packed):
            # the assembly kernel's slab, exactly as the wire would carry it
            frame = SimpleNamespace(
                request_id=9,
                layer=layer_id,
                tensors=(packed.detach().cpu(),),
                op=OP_STATE_APPLY,
            )
            apply_advance(self.service, frame)
            self.ops.append(("apply", int(layer_id)))

        def mix_host(layer_id, request_ids, packed, alpha, beta):
            # the first linear layer of a pass has no previous residual, so it takes the raw
            # op; this stand-in only has to answer with the right shape
            self.ops.append(("raw", int(layer_id)))
            rows = packed.shape[0]
            heads = self.config.linear_num_value_heads
            return torch.zeros(
                rows,
                heads * self.config.linear_value_head_dim,
                device=packed.device,
                dtype=torch.float32,
            )

        self.runner._local.issue_host = issue_host
        self.runner._local.collect_early = collect_early
        self.runner._local.apply_host = apply_host
        self.runner._local.mix_host = mix_host
        self.runner._local.ask_host = None

    def _windows(self, linear_layers):
        slot = self.service.cache.slot_of(3)
        partials = []
        for lid in linear_layers:
            window = self.service.cache.conv[lid][slot][..., 1:]
            weight = (
                self.stack.model.layers[lid]
                .linear_attn.conv1d.weight.squeeze(1)
                .to(window.device)[:, :-1]
            )
            partials.append((window * weight).sum(-1))
        stacked = torch.stack(partials, dim=0).unsqueeze(0)
        # flat, as the wire carries it; `_hold_windows` rebuilds the axes
        return stacked.reshape(1, -1).to("cuda")

    def test_a_prologue_and_a_span_run_cooked(self):
        layers = self.stack.model.layers
        prologue = [
            i for i in range(len(self.kinds)) if self.kinds[i] != "full_attention"
        ]
        first_head = next(
            i for i in range(len(self.kinds)) if self.kinds[i] == "full_attention"
        )
        prologue = [i for i in prologue if i < first_head]
        hidden_size = self.config.hidden_size
        embedded = torch.randn(1, hidden_size, device="cuda", dtype=torch.bfloat16)
        positions = torch.zeros(1, dtype=torch.int64, device="cuda")

        q, k, v = self.runner.run_prologue(
            [3], embedded, positions, windows=self._windows(prologue)
        )
        self.assertEqual(q.shape[0], 1)
        # the first linear layer has no previous residual and takes the raw op; every later
        # one was cooked -- early, then the reading collected, then the advance applied
        self.assertEqual(self.ops[0], ("raw", prologue[0]))
        for lid in prologue[1:]:
            for step in ("early", "collect", "apply"):
                self.assertIn((step, lid), self.ops)
            self.assertLess(
                self.ops.index(("early", lid)), self.ops.index(("collect", lid))
            )
            self.assertLess(
                self.ops.index(("collect", lid)), self.ops.index(("apply", lid))
            )
        self.assertEqual(self.readings, {})
        self.assertIsNone(self.service._parked)


if __name__ == "__main__":
    unittest.main()
