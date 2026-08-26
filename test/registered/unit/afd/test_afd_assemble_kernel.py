"""The fused assembly against the eager sequence it replaces, at real shapes.

The kernel also packs the APPLY frame on the way through -- `[k | v | alpha | beta |
raw_q | packed_kv]` in float32 -- so the agreement here covers both outputs: the core the
pool serves from, and the slab the host advances its state with. The slab is held to the
exact `torch.cat` it replaced, because float32 round-trips every part and the packing must
change no arithmetic at all.
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

CUDA = torch.cuda.is_available()
VH, DK, DV = 48, 128, 128
RQ, PK = 2048, 14336


@unittest.skipUnless(CUDA, "triton compiles for the device")
class TestTheFusedAssemblyIsTheEagerOne(CustomTestCase):
    def test_agreement(self):
        from sglang.srt.afd.linear_history import gates
        from sglang.srt.afd_query_shift.cook_kernel import assemble_and_pack

        for rows, seed in ((1, 11), (5, 12)):
            torch.manual_seed(seed)
            a = torch.randn(rows, VH, device="cuda")
            b = torch.randn(rows, VH, device="cuda")
            A_log = torch.randn(VH, device="cuda") * 0.3
            dt_bias = torch.randn(VH, device="cuda") * 0.3
            k = torch.randn(rows, VH, DK, device="cuda")
            q = torch.randn(rows, VH, DK, device="cuda")
            reading = torch.randn(rows, VH, DV, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(rows, VH, DV, device="cuda", dtype=torch.bfloat16)
            raw_q = torch.randn(rows, RQ, device="cuda", dtype=torch.bfloat16)
            packed_kv = torch.randn(rows, PK, device="cuda", dtype=torch.bfloat16)

            want_alpha, want_beta = gates(a, b, A_log, dt_bias)
            want_s = want_beta * (k * q).sum(-1)
            want_core = (
                want_alpha.unsqueeze(-1) * reading.float()
                + want_s.unsqueeze(-1) * v.float()
            ).reshape(rows, -1)
            want_slab = torch.cat(
                [
                    k.reshape(rows, -1).float(),
                    v.reshape(rows, -1).float(),
                    want_alpha,
                    want_beta,
                    raw_q.float(),
                    packed_kv.float(),
                ],
                dim=-1,
            )

            core, slab = assemble_and_pack(
                a, b, A_log, dt_bias, k, q, reading, v, raw_q, packed_kv
            )
            torch.testing.assert_close(core, want_core, rtol=1e-3, atol=1e-3)
            # the segments that are COPIES must be exact to the bit; the gates carry the
            # kernel's own arithmetic and get the same tolerance the core does
            width = VH * DK + VH * DV
            torch.testing.assert_close(
                slab[:, :width], want_slab[:, :width], rtol=0, atol=0
            )
            torch.testing.assert_close(
                slab[:, width : width + 2 * VH],
                want_slab[:, width : width + 2 * VH],
                rtol=1e-4,
                atol=1e-5,
            )
            torch.testing.assert_close(
                slab[:, width + 2 * VH :],
                want_slab[:, width + 2 * VH :],
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
