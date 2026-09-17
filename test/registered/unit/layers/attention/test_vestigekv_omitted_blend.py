"""The omitted-mass blend: research arm SGLANG_DEBUG_VESTIGEKV_OMITTED_BLEND.

VestigeKV attends ~6% of rows and captures ~57% of the dense softmax mass, so
every retained weight is inflated by Z/Zv (~2.9x on the Kimi dumps). The blend
puts the missing denominator back from the certified bound the scan already
computes, carrying the omitted mass at the archive's mean value. It rewrites
the attention output, so the arithmetic is pinned here rather than trusted.
"""

import unittest

import torch
from sglang.srt.layers.attention.vestigekv.tier_decode import _blend_omitted_mass

class TestBlend(unittest.TestCase):
    def _run(self, logm_vals, lse, nsplit, o, mu, slots):
        logm = torch.full((4, o.shape[1]), float("-inf"))
        for s, v in logm_vals.items():
            logm[s] = torch.tensor(v)
        mub = torch.zeros(4, o.shape[1], o.shape[-1]); mub[slots] = mu
        _blend_omitted_mass(o, lse, nsplit, logm, mub, slots)
        return o

    def test_minus_inf_mass_leaves_the_output_untouched(self):
        """A fenced lane attends everything, so logM is -inf and sigma is 1."""
        o = torch.randn(2, 3, 4); ref = o.clone()
        lse = torch.randn(2, 3, 2); nsp = torch.tensor([2, 2], dtype=torch.int32)
        self._run({}, lse, nsp, o, torch.randn(2, 3, 4), torch.tensor([0, 1]))
        torch.testing.assert_close(o, ref)

    def test_blend_matches_the_closed_form(self):
        o = torch.randn(2, 3, 4); src = o.clone()
        lse = torch.tensor([[[0.5, 0.25]] * 3, [[1.0, -2.0]] * 3])
        nsp = torch.tensor([2, 2], dtype=torch.int32)
        mu = torch.randn(2, 3, 4); slots = torch.tensor([0, 1])
        self._run({0: [0.1] * 3, 1: [-0.3] * 3}, lse, nsp, o, mu, slots)
        merged = torch.logsumexp(lse, -1)
        sig = torch.sigmoid(merged - torch.tensor([[0.1] * 3, [-0.3] * 3]))[..., None]
        torch.testing.assert_close(o, src * sig + mu * (1 - sig))

    def test_unwritten_splits_are_masked_not_merged(self):
        """attn_lse is torch.empty; only the first num_kv_splits are written.

        Folding the tail in would add a garbage term to the denominator, which
        moves sigma silently and in the direction of whatever was in memory.
        """
        o = torch.ones(1, 1, 2); mu = torch.zeros(1, 1, 2); slots = torch.tensor([0])
        lse = torch.tensor([[[0.0, 99.0]]])  # split 1 is stale memory
        self._run({0: [0.0]}, lse, torch.tensor([1], dtype=torch.int32), o, mu, slots)
        # only split 0 counts: lse = 0, logM = 0 -> sigma = 0.5
        torch.testing.assert_close(o, torch.full((1, 1, 2), 0.5))

    def test_the_batch_is_the_slot_slice_not_the_buffer_length(self):
        """The graph's lse buffer is sized for the LARGEST captured batch.

        A smaller capture leaves its tail unwritten, so taking the batch from
        attn_lse.shape[0] mixes a bs-4 buffer with a bs-2 step -- which is
        exactly how this arm aborted cuda-graph capture with "size of tensor a
        (4) must match tensor b (2)". The installed slot slice is the batch.
        """
        o = torch.ones(4, 1, 2)  # buffer for 4, step of 2
        lse = torch.zeros(4, 1, 1)
        nsp = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        slots = torch.tensor([0, 1])
        logm = torch.zeros(4, 1)  # logM = lse = 0 -> sigma = 0.5
        mub = torch.zeros(4, 1, 2)
        _blend_omitted_mass(o, lse, nsp, logm, mub, slots)
        torch.testing.assert_close(o[:2], torch.full((2, 1, 2), 0.5))
        torch.testing.assert_close(o[2:], torch.ones(2, 1, 2))  # tail untouched


if __name__ == "__main__":
    unittest.main(verbosity=2)
