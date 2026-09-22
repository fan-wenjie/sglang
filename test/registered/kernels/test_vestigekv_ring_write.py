"""The fused ring write files exactly what the torch chain filed.

The chain was act_quant (ue8m0) + three index_copy_ + the address arithmetic;
the fused kernel copies act_quant's math, and the ring, the stamp and the
scale table must come out byte for byte the same, including a position that
wraps the ring and two requests filed in one launch.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, suite="base-b-test-1-gpu-small")

RING, D, R1 = 4096 + 512, 128, 4


def _reference(key, pos, slots, fp8):
    from sglang.srt.layers.attention.vestigekv.salience import quantize_salience

    dt = torch.float8_e4m3fn if fp8 else torch.bfloat16
    ring = torch.zeros(R1, RING, D, dtype=dt, device="cuda")
    stamp = torch.full((R1, RING), -1, dtype=torch.int32, device="cuda")
    scale = torch.zeros(R1, RING, dtype=torch.float32, device="cuda")
    flat = slots * RING + pos % RING
    stamp.view(-1).index_copy_(0, flat, pos.to(torch.int32))
    if fp8:
        q, s = quantize_salience(key)
        ring.view(-1, D).view(torch.uint8).index_copy_(0, flat, q.view(torch.uint8))
        scale.view(-1).index_copy_(0, flat, s)
    else:
        ring.view(-1, D).index_copy_(0, flat, key.to(torch.bfloat16))
    return ring, stamp, scale


class TestRingWrite(CustomTestCase):
    def _case(self, fp8, T, seed):
        from sglang.srt.layers.attention.vestigekv.ring_write import ring_write

        g = torch.Generator(device="cuda").manual_seed(seed)
        key = torch.randn(T, D, device="cuda", generator=g, dtype=torch.bfloat16)
        key[: T // 3] *= 40.0  # scales across several powers of two
        pos = torch.randint(0, 3 * RING, (T,), device="cuda", generator=g, dtype=torch.int64)
        slots = torch.randint(0, R1, (T,), device="cuda", generator=g, dtype=torch.int64)
        # no two rows of one launch may target the same ring row (the chain's
        # index_copy_ order would decide the winner); keep (slot, pos % RING) unique
        flat = slots * RING + pos % RING
        keep = torch.ones(T, dtype=torch.bool, device="cuda")
        seen = set()
        for i, f in enumerate(flat.tolist()):
            if f in seen:
                keep[i] = False
            seen.add(f)
        key, pos, slots = key[keep], pos[keep], slots[keep]
        want = _reference(key, pos, slots, fp8)
        dt = torch.float8_e4m3fn if fp8 else torch.bfloat16
        ring = torch.zeros(R1, RING, D, dtype=dt, device="cuda")
        stamp = torch.full((R1, RING), -1, dtype=torch.int32, device="cuda")
        scale = torch.zeros(R1, RING, dtype=torch.float32, device="cuda") if fp8 else None
        ring_write(key, pos, slots, ring, stamp, scale)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(ring.view(torch.uint8), want[0].view(torch.uint8)), "ring bytes differ")
        self.assertTrue(torch.equal(stamp, want[1]), "stamps differ")
        if fp8:
            self.assertTrue(torch.equal(scale, want[2]), "scales differ")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_fp8_ring_matches_the_torch_chain(self):
        self._case(fp8=True, T=1, seed=1)  # one decode token
        self._case(fp8=True, T=512, seed=2)  # a prefill chunk, several slots

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_bf16_ring_matches_the_torch_chain(self):
        self._case(fp8=False, T=300, seed=3)
