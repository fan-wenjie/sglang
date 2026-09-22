"""Reading a lane's rows from the tiers must equal reading them from a CSR.

The forked stage 1 differs from upstream's only in where a row id comes from:
the kept table then the fetch buffer, or the page table when the lane is
fenced, instead of one index array. Same rows in the same order through the
same schedule, so the outputs must agree bit for bit -- if they do not, the
fork changed something it was not supposed to.
"""

import re
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

POOL, R1, CAP, FW, R2T = 4096, 4, 512, 128, 1024
LK, LV, H = 576, 512, 16
SPLITS = 8


def _rows(gen):
    from sglang.srt.layers.attention.vestigekv.decode_fork import VestigeKVRows

    kept_buf = torch.randint(
        0, POOL, (R1, CAP), dtype=torch.int32, device="cuda", generator=gen
    )
    fetch_buf = torch.randint(
        0, POOL, (R1, FW), dtype=torch.int32, device="cuda", generator=gen
    )
    kept_len = torch.full((R1,), 300, dtype=torch.int32, device="cuda")
    fetch_len = torch.full((R1,), 37, dtype=torch.int32, device="cuda")
    r2t = torch.randint(
        0, POOL, (R1 - 1, R2T), dtype=torch.int32, device="cuda", generator=gen
    )
    return VestigeKVRows(
        slots=torch.tensor([1], dtype=torch.int64, device="cuda"),
        kept_buf=kept_buf,
        kept_len=kept_len,
        fetch_buf=fetch_buf,
        fetch_len=fetch_len,
        fetch_ovf=torch.zeros(R1, dtype=torch.int32, device="cuda"),
        r2t=r2t,
        seq=torch.tensor([777], dtype=torch.int64, device="cuda"),
        loc=torch.tensor([POOL - 1], dtype=torch.int64, device="cuda"),
    )


def _run(q, kb, vb, indptr, indices, vk, tiers, fence=True, affine=False):
    import msgspec

    from sglang.srt.layers.attention.vestigekv.decode_fork import (
        decode_grouped_att_m_fwd,
    )

    bs = q.shape[0]
    out = torch.zeros(bs, H, SPLITS, LV, dtype=torch.float32, device="cuda")
    lse = torch.zeros(bs, H, SPLITS, dtype=torch.float32, device="cuda")
    splits = torch.full((bs,), SPLITS, dtype=torch.int32, device="cuda")
    decode_grouped_att_m_fwd(
        q,
        kb,
        vb,
        out,
        lse,
        indptr,
        indices,
        msgspec.structs.replace(vk, tiers=tiers, fence=fence, affine=affine),
        splits,
        SPLITS,
        1.0 / (LK**0.5),
        0.0,
        has_mla=True,
    )
    return out, lse


class TestDecodeForkRowSource(CustomTestCase):
    def _case(self, fenced):
        gen = torch.Generator(device="cuda").manual_seed(7 + fenced)
        pool = torch.randn(
            POOL, 1, LK, dtype=torch.bfloat16, device="cuda", generator=gen
        )
        q = torch.randn(1, H, LK, dtype=torch.bfloat16, device="cuda", generator=gen)
        vk = _rows(gen)
        slot = int(vk.slots[0])
        if fenced:
            vk.fetch_ovf[slot] = 1
            n = int(vk.seq[0])
            ids = vk.r2t[slot, :n].clone()
            ids[-1] = int(vk.loc[0])
        else:
            nk, nf = int(vk.kept_len[slot]), int(vk.fetch_len[slot])
            ids = torch.cat([vk.kept_buf[slot, :nk], vk.fetch_buf[slot, :nf]])
        indices = ids.to(torch.int64)
        indptr = torch.tensor([0, indices.numel()], dtype=torch.int32, device="cuda")
        a_out, a_lse = _run(q, pool, pool[:, :, :LV], indptr, indices, vk, tiers=False)
        b_out, b_lse = _run(q, pool, pool[:, :, :LV], indptr, indices, vk, tiers=True)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(a_out, b_out), f"att_out differs (fenced={fenced})")
        self.assertTrue(torch.equal(a_lse, b_lse), f"att_lse differs (fenced={fenced})")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_kept_plus_fetched_equals_the_csr(self):
        self._case(fenced=False)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_a_fenced_lane_equals_its_page_table_as_a_csr(self):
        self._case(fenced=True)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_affine_equals_the_page_table_when_it_is_contiguous(self):
        # The affine arm computes a fenced row id as base + offs_n instead of
        # loading it, which is only the same row set when the page table is one
        # contiguous run. pagetable_affine measures that it is; this pins that
        # the two arms then agree bit for bit.
        gen = torch.Generator(device="cuda").manual_seed(13)
        pool = torch.randn(
            POOL, 1, LK, dtype=torch.bfloat16, device="cuda", generator=gen
        )
        q = torch.randn(1, H, LK, dtype=torch.bfloat16, device="cuda", generator=gen)
        vk = _rows(gen)
        slot = int(vk.slots[0])
        vk.fetch_ovf[slot] = 1
        n = int(vk.seq[0])
        base = 7
        vk.r2t[slot, :n] = base + torch.arange(n, dtype=torch.int32, device="cuda")
        vk.loc.fill_(base + n - 1)  # the affine arm has no last-row fixup
        indices = vk.r2t[slot, :n].to(torch.int64)
        indptr = torch.tensor([0, n], dtype=torch.int32, device="cuda")
        a_out, a_lse = _run(q, pool, pool[:, :, :LV], indptr, indices, vk, tiers=True)
        b_out, b_lse = _run(
            q, pool, pool[:, :, :LV], indptr, indices, vk, tiers=True, affine=True
        )
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(a_out, b_out), "att_out differs under the affine arm"
        )
        self.assertTrue(
            torch.equal(a_lse, b_lse), "att_lse differs under the affine arm"
        )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_fence_off_ignores_the_overflow_flag(self):
        # The compaction raises fetch_ovf on any overflow, whatever
        # --disable-vestigekv-recall-overflow-fallback says, so the flag alone
        # must not send the kernel to the page table: with fence off the lane
        # reads kept plus fetched, which is what the CSR path packs there.
        gen = torch.Generator(device="cuda").manual_seed(11)
        pool = torch.randn(
            POOL, 1, LK, dtype=torch.bfloat16, device="cuda", generator=gen
        )
        q = torch.randn(1, H, LK, dtype=torch.bfloat16, device="cuda", generator=gen)
        vk = _rows(gen)
        slot = int(vk.slots[0])
        vk.fetch_ovf[slot] = 1  # the scan overflowed; the fallback is off
        nk, nf = int(vk.kept_len[slot]), int(vk.fetch_len[slot])
        indices = torch.cat([vk.kept_buf[slot, :nk], vk.fetch_buf[slot, :nf]]).to(
            torch.int64
        )
        indptr = torch.tensor([0, indices.numel()], dtype=torch.int32, device="cuda")
        a_out, a_lse = _run(q, pool, pool[:, :, :LV], indptr, indices, vk, tiers=False)
        b_out, b_lse = _run(
            q, pool, pool[:, :, :LV], indptr, indices, vk, tiers=True, fence=False
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(a_out, b_out), "att_out differs with the fence off")
        self.assertTrue(torch.equal(a_lse, b_lse), "att_lse differs with the fence off")


class TestDecodeForkRegisters(CustomTestCase):
    """No build may spill, and none may sit at the register ceiling.

    Register allocation is a property of the build, so whatever a variant costs
    is paid on every step -- including the 99.7% of scans that never fence.
    Written as a branch inside the row loop the fenced arm reached the
    255-register ceiling and spilled 40 bytes to local memory; selecting the
    row value instead of branching on the row source brought it back to 200 and
    none.

    The fenced build now also carries the affine arm, which is deliberately a
    second loop, so it legitimately uses more registers than the unfenced one.
    What must not happen is a spill, or an allocation at the 255 ceiling, which
    is where the allocator is starved and the next edit spills.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_the_fence_costs_no_registers(self):
        import subprocess

        from sglang.srt.layers.attention.vestigekv import decode_fork as df

        use = {}
        for fence in (False, True):
            gen = torch.Generator(device="cuda").manual_seed(3)
            pool = torch.randn(
                POOL, 1, LK, dtype=torch.bfloat16, device="cuda", generator=gen
            )
            q = torch.randn(
                1, H, LK, dtype=torch.bfloat16, device="cuda", generator=gen
            )
            vk = _rows(gen)
            indices = torch.zeros(337, dtype=torch.int64, device="cuda")
            indptr = torch.tensor([0, 337], dtype=torch.int32, device="cuda")
            cache = df._vk_fwd_grouped_kernel_stage1.device_caches[
                torch.cuda.current_device()
            ][0]
            before = set(cache)
            _run(q, pool, pool[:, :, :LV], indptr, indices, vk, tiers=True, fence=fence)
            torch.cuda.synchronize()
            new = set(cache) - before
            if not new:  # already compiled in this process by an earlier case
                self.skipTest("kernel already cached for this signature")
            cubin = cache[new.pop()].asm["cubin"]
            path = f"/tmp/vk_fence_{int(fence)}.cubin"
            open(path, "wb").write(cubin)
            out = subprocess.run(
                ["cuobjdump", "-res-usage", path], capture_output=True, text=True
            ).stdout
            use[fence] = {
                k: int(v)
                for k, v in (
                    m.split(":") for m in re.findall(r"(?:REG|STACK):\d+", out)
                )
            }
        for fence, u in use.items():
            self.assertEqual(u.get("STACK", 0), 0, f"fence={fence} spills: {u}")
            self.assertLess(u["REG"], 255, f"fence={fence} is at the ceiling: {u}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
