"""VestigeKV model geometry: the derivation from a config, and the two
contracts the geometry refactor changed -- the sigma slice is bounded by the
sidecar width, and a rope-less tier scores recall from the sketch alone.

All cases run on the CPU fallback paths (no fused kernels).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.eviction import (
    blockwise_sigma,
    blockwise_sigma_from_pool,
)
from sglang.srt.layers.attention.vestigekv.geometry import KIMI_LINEAR, Geometry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

KIMI_CFG = SimpleNamespace(kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64)
GLM_CFG = SimpleNamespace(
    kv_lora_rank=512, qk_nope_head_dim=256, qk_rope_head_dim=0, index_head_dim=128
)
ROPE_LESS_GEOM = Geometry.from_hf_config(GLM_CFG)


class TestGeometryDerivation(CustomTestCase):
    def test_kimi_derivation_matches_the_validated_defaults(self):
        # KIMI_LINEAR is the default every kernel entry point falls back to;
        # the config-derived geometry of the validated model must be the same
        # object's values, or the serving path and the tests drift apart.
        g = Geometry.from_hf_config(KIMI_CFG)
        self.assertEqual(g, KIMI_LINEAR)
        self.assertEqual(g.latent_dim, D.LATENT_DIM)
        self.assertEqual(g.attn_scale, D.ATTN_SCALE)
        self.assertEqual(g.sigma_offset, D.KV_LORA_RANK)

    def test_rope_less_config_takes_the_indexer_channel(self):
        g = ROPE_LESS_GEOM
        self.assertEqual(g.side_dim, 0)
        self.assertEqual(g.latent_dim, 512)  # the latent row is content only
        self.assertEqual(g.attn_scale, 256**-0.5)  # nope width alone
        self.assertEqual((g.sigma_dim, g.sigma_in_row, g.sigma_offset), (128, False, 0))

    def test_rope_less_config_without_an_indexer_is_refused(self):
        cfg = SimpleNamespace(kv_lora_rank=512, qk_nope_head_dim=256, qk_rope_head_dim=0)
        with self.assertRaises(ValueError):
            Geometry.from_hf_config(cfg)


class TestSigmaSliceWidth(CustomTestCase):
    def test_sigma_slice_stops_at_the_sidecar_width(self):
        # The pool row may be wider than the latent row (padding, extra
        # fields); sigma must read exactly `dim` columns at `offset`. The old
        # tail-to-end slice would fold the extra columns into the residual.
        T, W = D.CLOSE_BLOCK, D.LATENT_DIM + 64
        g = torch.Generator().manual_seed(0)
        kbuf = torch.randn(T + 32, W, generator=g)
        slots = torch.randperm(T + 32, generator=g)[:T]
        got = blockwise_sigma_from_pool(
            kbuf, slots, T, offset=D.KV_LORA_RANK, dim=D.SIDECAR_DIM
        )
        want = blockwise_sigma(kbuf[slots][:, D.KV_LORA_RANK : D.LATENT_DIM], T)
        tail = blockwise_sigma(kbuf[slots][:, D.KV_LORA_RANK :], T)
        self.assertTrue(torch.equal(got, want))
        self.assertFalse(torch.equal(got, tail))




if __name__ == "__main__":
    unittest.main(verbosity=3)
