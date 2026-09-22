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


class TestRopeLessRecall(CustomTestCase):
    """With side_dim == 0 the recall score is sketch + certificate only; the
    tier must build and query on 512-wide rows without any 576 assumption."""

    def _tier(self, T=512, H=4, n_cal=8, seed=0):
        from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

        g = torch.Generator().manual_seed(seed)
        kbuf = torch.randn(T + 16, 512, generator=g).to(torch.bfloat16)
        slots = torch.randperm(T + 16, generator=g)[:T]
        keep = torch.zeros(T, dtype=torch.bool)
        keep[torch.randperm(T, generator=g)[: T // 8]] = True
        qcal = torch.randn(n_cal, H, 512, generator=g)
        qpos = torch.randint(0, T, (n_cal,), generator=g)
        t = RecallTier(geom=ROPE_LESS_GEOM)
        t.build(kbuf, slots, keep, qcal, qpos, conservative=True)
        return t, torch.randn(H, 512, generator=g)

    def test_build_leaves_an_empty_sidecar(self):
        t, _ = self._tier()
        self.assertEqual(tuple(t.side.shape), (int(t.arch.numel()), 0))
        self.assertEqual(t.scale, 256**-0.5)

    def test_fired_set_is_the_sketch_plus_certificate_rule(self):
        t, qe = self._tier()
        sc = t.scale
        skept = (qe.to(torch.bfloat16) @ t.kept_rows.T).float() * sc
        max1 = skept.max(-1).values
        qsk = qe @ t.V.T
        qres = (qe - qsk @ t.V).norm(dim=-1)
        score = (qsk.half() @ t.csk.T).float() * sc + t.zp * (
            qres[:, None] * t.rho[None, :] * sc / (512 - t.r) ** 0.5
        )
        want = t.arch[(score > max1[:, None]).any(0)]
        got = t.query(qe)
        self.assertTrue(torch.equal(got.sort().values, want.sort().values))
        self.assertGreater(int(got.numel()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=3)
