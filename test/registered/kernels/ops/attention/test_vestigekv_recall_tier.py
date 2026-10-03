"""The sketch basis is a free change on soundness and a large one on cost.

`RecallTier.build` states the contract: the certificate
q.c = <c~, q~> + q_res . c_res is exact for ANY orthonormal basis, and only
its TIGHTNESS varies with the choice. Tightness is what the fire count is made
of, and the fire count against the recall capacity is what the fallback rate
is made of -- so the basis is a cost knob that cannot move correctness, as
long as it really is orthonormal.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class TestSketchBasisSelector(CustomTestCase):
    """Both bases must be orthonormal, because that is what the certificate rests on.

    `build` says it plainly: the certificate q.c = <c~,q~> + q_res.c_res is
    exact for ANY orthonormal basis, and only its TIGHTNESS varies. So a new
    basis is a free change on soundness and a large one on fire count -- but
    only if it really is orthonormal. A basis that drifts off orthonormal keeps
    firing rows and reports nothing wrong, which is the failure these pin.
    """

    def _fit(self, basis, seed=0):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.vestigekv import defaults as D
        from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

        g = torch.Generator(device="cuda").manual_seed(seed)
        POOL, Hh, R = 4096, 8, 64
        kv = D.KV_LORA_RANK
        # content rows with a deliberate MEAN component: that is the direction
        # a query-centered fit discards and a key second moment keeps.
        base = torch.randn(1, kv, device="cuda", generator=g) * 3.0
        # the default geometry is Kimi Linear: rows are kv + sidecar wide, and
        # only the leading kv columns are the content the basis is fitted on.
        kbuf = torch.randn(POOL, D.LATENT_DIM, device="cuda", generator=g)
        kbuf[:, :kv] += base
        kbuf = kbuf.to(torch.bfloat16)
        slots = torch.arange(POOL, device="cuda")
        keep = torch.zeros(POOL, dtype=torch.bool, device="cuda")
        keep[: POOL // 8] = True
        qcal = torch.randn(16, Hh, D.LATENT_DIM, device="cuda", generator=g)
        qpos = torch.randint(0, POOL, (16,), device="cuda", generator=g)
        t = RecallTier(r=R)
        with envs.SGLANG_VESTIGEKV_SKETCH_BASIS.override(basis):
            t.build(kbuf, slots, keep, qcal, qpos)
        return t.V.float(), kbuf, keep

    def test_both_bases_are_orthonormal(self):
        for basis in ("qpca", "kmom"):
            V, _kbuf, _keep = self._fit(basis)
            gram = V @ V.T
            eye = torch.eye(V.shape[0], device=V.device)
            err = (gram - eye).abs().max().item()
            self.assertLess(
                err, 2e-3,
                f"{basis} basis is not orthonormal (max |VV^T - I| = {err:.2e}); "
                "the Cauchy-Schwarz certificate is unsound without it",
            )

    def test_kmom_keeps_more_key_energy_than_qpca(self):
        Vq, kbuf, keep = self._fit("qpca")
        Vk, _, _ = self._fit("kmom")
        rows = kbuf[~keep][:, : Vq.shape[1]].float()
        tot = rows.pow(2).sum()
        eq = (rows @ Vq.T).pow(2).sum() / tot
        ek = (rows @ Vk.T).pow(2).sum() / tot
        self.assertGreater(
            float(ek), float(eq),
            f"kmom captured {ek:.3f} of key energy against qpca's {eq:.3f}; "
            "the whole reason to fit on the keys is that it captures more",
        )

    def test_an_unknown_basis_is_refused_not_ignored(self):
        with self.assertRaises(ValueError):
            self._fit("qcpa")  # transposed, the typo that would silently serve qpca

    def test_the_key_mean_direction_is_in_the_kmom_basis(self):
        """Uncentered is the point, and fitting on keys alone does not pin it.

        Centring the key gram still beats a query-fitted basis on key energy,
        so the energy test above passes either way -- it separates key-fit from
        query-fit, not uncentered from centered. What the offline GLM dumps
        actually show is a large MEAN component (key_mean_energy 0.44 to 0.78),
        and subtracting it spends the basis on the fluctuation instead of the
        direction every row shares. This pins that direction.
        """
        Vk, kbuf, keep = self._fit("kmom")
        kv = Vk.shape[1]
        mu = kbuf[~keep][:, :kv].float().mean(0)
        mu = mu / mu.norm()
        kept = (Vk @ mu).norm().item()  # ||proj_V mu||, 1.0 = fully represented
        self.assertGreater(
            kept, 0.9,
            f"the shared key-mean direction retains only {kept:.3f} of its norm "
            "in the kmom basis; a centered gram would discard it, which is the "
            "component the fire count is most sensitive to",
        )
