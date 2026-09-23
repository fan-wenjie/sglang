"""The DSA-side backend's lean/top-k dispatch reads the overflow probe.

Regression: the probe's readback was issued with a torch.sum signature
torch rejects, so the first decode step of every server on the optimised
tree died before the client saw a token; no CPU test exercises the probe
(a pinned host scalar and an event need a device). This pins the black-box
contract: with no overflow the next step is lean, after an overflow it runs
the full indexer, and one probe never blocks the stream.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.vestigekv_dsa_backend import VestigeKVDSABackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, suite="base-b-test-1-gpu-small")


def _be(n_layers=3):
    be = VestigeKVDSABackend.__new__(VestigeKVDSABackend)
    be._ovf_count_stack = torch.zeros(n_layers, dtype=torch.int32, device="cuda")
    be._ovf_probe = None
    be._ovf_last = 0
    be.lean_step = False
    be._dsa = SimpleNamespace()  # a resolved sibling
    be._full_arm = lambda: False
    be._stats = {"lean": 0, "topk": 0}
    be.config = SimpleNamespace(overflow_fallback=True)
    return be


class TestOverflowProbe(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_lean_until_a_lane_overflows(self):
        from sglang.srt.layers.attention.graph_variants import VK_LEAN, VK_TOPK

        be = _be()
        be._probe_overflow()
        torch.cuda.synchronize()
        self.assertEqual(be._variant_for_step(None), VK_LEAN)
        be._ovf_count_stack[1] += 1  # a fire past the fetch capacity
        be._probe_overflow()
        torch.cuda.synchronize()
        self.assertEqual(be._variant_for_step(None), VK_TOPK)
        be._probe_overflow()  # count unchanged since the last readback
        torch.cuda.synchronize()
        self.assertEqual(be._variant_for_step(None), VK_LEAN)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_an_unfinished_probe_keeps_the_last_decision(self):
        from sglang.srt.layers.attention.graph_variants import VK_TOPK

        be = _be()
        be.lean_step = False
        dev = torch.zeros((), dtype=torch.int32, device="cuda")
        host = torch.zeros((), dtype=torch.int32, pin_memory=True)
        be._ovf_probe = (dev, host, _Pending())
        self.assertEqual(be._variant_for_step(None), VK_TOPK)


class TestFenceOffIsLeanEverywhere(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_no_fence_no_probe_lean_every_step(self):
        # With the fallback off a lane attends its own rows whatever the fire
        # count, so the indexer's selection is never needed: lean without a
        # probe, even right after an overflow.
        from sglang.srt.layers.attention.graph_variants import VK_LEAN

        be = _be()
        be.config = SimpleNamespace(overflow_fallback=False)
        be._ovf_count_stack[0] += 3
        self.assertEqual(be._variant_for_step(None), VK_LEAN)
        self.assertIsNone(be._ovf_probe)
        self.assertEqual(be._stats["lean"], 1)


class _Pending:
    def query(self):
        return False


if __name__ == "__main__":
    unittest.main()
