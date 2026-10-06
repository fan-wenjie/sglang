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

from sglang.srt.environ import envs

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
    return be


class TestOverflowProbe(CustomTestCase):
    """VK_LEAN is a NON-branch-arm mechanism and these cases must say so.

    `_variant_for_step` refuses lean while the branch rule is active, because the
    branch rule thresholds the indexer's scores every step and lean is precisely the
    variant that skips the indexer. Since BRANCH_Q acquired a served default, an unset
    variable means the branch rule is ON, so a lean case has to disable it explicitly --
    `override(None)` sets the explicit-None state the accessor distinguishes from unset.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_lean_until_a_lane_overflows(self):
        from sglang.srt.layers.attention.graph_variants import VK_LEAN, VK_TOPK

        self.enterContext(envs.SGLANG_VESTIGEKV_BRANCH_Q.override(None))
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


class _Pending:
    def query(self):
        return False


if __name__ == "__main__":
    unittest.main()


class TestSelectForkRouting(CustomTestCase):
    """The flag must reach the KERNEL, not merely be read.

    Two arms on this line were silently neutered by actuators that no-opped
    while every log line looked right, so a flag that routes tier 2 to a
    different kernel gets a case that observes WHICH path executed -- the
    scratch each one allocates -- rather than that the flag parsed.
    """

    def _fetch_be(self, dev):
        import torch as T

        be = VestigeKVDSABackend.__new__(VestigeKVDSABackend)
        be._full_arm = lambda: False
        be._branch_q = lambda: 0.75
        be._kept_rows_bound = lambda: 64
        be._li_map = {7: 0}
        be._qbuf = {7: None}
        be._text_cfg_index_kpool = 4
        be._fetch_w = 2048
        G = 256
        be._kept_pos_stack = [T.zeros(1, 64, dtype=T.int32, device=dev)]
        be._kept_len_stack = [T.zeros(1, dtype=T.int32, device=dev)]
        be._fetch_stack = [T.full((1, be._fetch_w), -1, dtype=T.int32, device=dev)]
        be._fetch_len_stack = [T.zeros(1, dtype=T.int32, device=dev)]
        be._fetch_ovf_stack = [T.zeros(1, dtype=T.int32, device=dev)]
        be.req_to_token_pool = SimpleNamespace(
            req_to_token=T.arange(G * 4, dtype=T.int32, device=dev).repeat(1, 1)
        )
        return be, G

    def test_the_flag_routes_to_the_forked_kernel(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        dev = torch.device("cuda")
        be, G = self._fetch_be(dev)
        logits = torch.randn(1, G, device=dev, dtype=torch.float32)
        pool_lens = torch.full((1,), G, dtype=torch.int32, device=dev)
        fb = SimpleNamespace(req_pool_indices=torch.zeros(1, dtype=torch.int64,
                                                          device=dev))
        with envs.SGLANG_VESTIGEKV_SELECT_FORK.override(True):
            be.group_fetch(layer_id=7, logits=logits, pool_lens=pool_lens,
                           forward_batch=fb)
        # Which path ran, observed through what it had to allocate.
        self.assertIn("_select_scratch_bufs", be.__dict__,
                      "the flag was on and the fork did not run")
        self.assertNotIn("_branch_scratch_bufs", be.__dict__,
                         "the chain ran as well as the fork")

    def test_without_the_flag_the_chain_still_runs(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        dev = torch.device("cuda")
        be, G = self._fetch_be(dev)
        logits = torch.randn(1, G, device=dev, dtype=torch.float32)
        pool_lens = torch.full((1,), G, dtype=torch.int32, device=dev)
        fb = SimpleNamespace(req_pool_indices=torch.zeros(1, dtype=torch.int64,
                                                          device=dev))
        with envs.SGLANG_VESTIGEKV_SELECT_FORK.override(False):
            be.group_fetch(layer_id=7, logits=logits, pool_lens=pool_lens,
                           forward_batch=fb)
        self.assertIn("_branch_scratch_bufs", be.__dict__,
                      "the default path did not run")
        self.assertNotIn("_select_scratch_bufs", be.__dict__)
