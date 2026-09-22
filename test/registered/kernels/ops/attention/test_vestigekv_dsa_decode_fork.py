"""The DSA-side decode fork computes what DSA's own kernel computes on the same rows.

vk_dsa_decode reads a lane's rows from the tiers (kept table then fetch
buffer, or the indexer's selection when fenced) with DSA's split-K kernel.
Given the same rows as a [bs, 1, n] index table with no pads and the same
split count, DSA's kernel and the fork partition the tiles identically and
must agree bit for bit -- the fork changed where a row id comes from, not
what is done with it. A lane far shorter than the split partition (empty
splits) and a fenced lane with a -1 pad inside the count are the two
places the fork's own logic has to hold.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="base-b-test-1-gpu-small")

POOL, R1, CAP, FW = 8192, 4, 4096, 2048
H, DV = 16, 512


def _rows(gen, nk, nf, slot=1, seq=777):
    from sglang.srt.layers.attention.vestigekv.decode_fork import VestigeKVRows

    kept_buf = torch.randint(0, POOL, (R1, CAP), dtype=torch.int32, device="cuda", generator=gen)
    fetch_buf = torch.randint(0, POOL, (R1, FW), dtype=torch.int32, device="cuda", generator=gen)
    kept_len = torch.full((R1,), nk, dtype=torch.int32, device="cuda")
    fetch_len = torch.full((R1,), nf, dtype=torch.int32, device="cuda")
    r2t = torch.randint(0, POOL, (R1 - 1, 2048), dtype=torch.int32, device="cuda", generator=gen)
    return VestigeKVRows(
        slots=torch.tensor([slot], dtype=torch.int64, device="cuda"),
        kept_buf=kept_buf, kept_len=kept_len, fetch_buf=fetch_buf, fetch_len=fetch_len,
        fetch_ovf=torch.zeros(R1, dtype=torch.int32, device="cuda"), r2t=r2t,
        seq=torch.tensor([seq], dtype=torch.int64, device="cuda"),
        loc=torch.tensor([POOL - 1], dtype=torch.int64, device="cuda"),
    )


def _dsa(q, kv, rows, splits, scale):
    from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
        triton_sparse_mla_decode_splitk,
    )

    idx = rows.to(torch.int32)[None, None, :].contiguous()  # [bs, 1, n]
    return triton_sparse_mla_decode_splitk(
        q[:, :, :DV], q[:, :, DV:], kv, idx, scale, d_v=DV, kv_splits=splits
    )[0]


class TestDsaDecodeFork(CustomTestCase):
    def _fork(self, q, kv, vk, splits, scale):
        from sglang.srt.layers.attention.vestigekv.dsa_decode_fork import vk_dsa_decode

        out = torch.zeros(1, H, DV, dtype=torch.bfloat16, device="cuda")
        vk_dsa_decode(q, kv, out, vk, scale, d_v=DV, kv_splits=splits)
        return out

    def _setup(self, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        kv = torch.randn(POOL, 1, DV, dtype=torch.bfloat16, device="cuda", generator=gen)
        q = torch.randn(1, H, DV, dtype=torch.bfloat16, device="cuda", generator=gen)
        return gen, kv, q

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_kept_plus_fetched_equals_dsa_on_the_same_rows(self):
        gen, kv, q = self._setup(3)
        vk = _rows(gen, nk=2018, nf=263)
        slot = int(vk.slots[0])
        rows = torch.cat([vk.kept_buf[slot, :2018], vk.fetch_buf[slot, :263]])
        for splits in (1, 8, 32):
            a = _dsa(q, kv, rows, splits, 0.0442)
            b = self._fork(q, kv, vk, splits, 0.0442)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(a, b), f"differs at kv_splits={splits}")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_a_short_lane_leaves_empty_splits_harmless(self):
        gen, kv, q = self._setup(4)
        vk = _rows(gen, nk=37, nf=5)
        slot = int(vk.slots[0])
        rows = torch.cat([vk.kept_buf[slot, :37], vk.fetch_buf[slot, :5]])
        # poison the shared partial buffers first: an unwritten split must not leak
        from sglang.kernels.ops.attention.dsa import triton_sparse_mla_decode as m

        m._get_splitk_bufs(1, 64, 16, DV, q.device)
        lse_buf, acc_buf = m._splitk_bufs[q.device]
        lse_buf.fill_(float("nan"))
        acc_buf.fill_(float("nan"))
        b = self._fork(q, kv, vk, 64, 0.0442)
        lse_buf.fill_(float("nan"))
        acc_buf.fill_(float("nan"))
        a = _dsa(q, kv, rows, 64, 0.0442)
        torch.cuda.synchronize()
        self.assertFalse(bool(torch.isnan(b).any()), "an empty split leaked a stale partial")
        self.assertTrue(torch.equal(a, b))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_a_fenced_lane_attends_the_selection_like_dsa(self):
        import msgspec

        gen, kv, q = self._setup(5)
        vk = _rows(gen, nk=2018, nf=263, seq=6001)
        slot = int(vk.slots[0])
        vk.fetch_ovf[slot] = 1
        K, kpool = 2048, 4
        tail = 6001 % kpool
        n = min(6001 - tail, K) + tail  # compute_dsa_seqlens: 2048 + 1
        topk = torch.full((1, K + kpool - 1), -1, dtype=torch.int32, device="cuda")
        topk[0, :n] = torch.randperm(POOL, device="cuda", generator=gen)[:n].to(torch.int32)
        topk[0, 7] = -1  # a pad inside the count: masked, exactly as DSA masks it
        vk = msgspec.structs.replace(vk, topk=topk, topk_k=K, kpool=kpool)
        a = _dsa(q, kv, topk[0, :n], 32, 0.0442)  # DSA over the same n entries, pad included
        b = self._fork(q, kv, vk, 32, 0.0442)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(a, b))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_the_fork_files_q_into_qbuf(self):
        import msgspec

        gen, kv, q = self._setup(6)
        vk = _rows(gen, nk=300, nf=37)
        qbuf = torch.zeros(R1, H, DV, dtype=torch.bfloat16, device="cuda")
        vk = msgspec.structs.replace(vk, qbuf=qbuf)
        self._fork(q, kv, vk, 8, 0.0442)
        torch.cuda.synchronize()
        slot = int(vk.slots[0])
        self.assertTrue(torch.equal(qbuf[slot], q[0]))
        others = torch.ones(R1, dtype=torch.bool, device="cuda")
        others[slot] = False
        self.assertEqual(float(qbuf[others].abs().sum()), 0.0)
