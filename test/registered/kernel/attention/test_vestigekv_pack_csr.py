"""The two-kernel CSR pack must match the torch _pack_csr bit-for-bit.

Two implementations of one contract exist on purpose (the torch chain serves
the eager step, the Triton pair serves the captured graphs); this test is the
divergence guard: same inputs -> identical kept-table mutation, identical CSR
indices content, identical indptr. The overflow fence is part of the
contract: a lane whose overflow flag is up packs its full row set.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

L, R1, CAP, FW, MAXBS = 3, 9, 64, 16, 4
TRASH = R1 - 1
R2T = CAP  # req_to_token width; seq_len never exceeds the kept capacity here


def _mk_state(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kept_buf = torch.randint(
        1, 5000, (L, R1, CAP), dtype=torch.int32, device="cuda", generator=g
    )
    kept_len = torch.randint(
        1, CAP - 2, (L, R1), dtype=torch.int32, device="cuda", generator=g
    )
    fetch_len = torch.randint(
        0, FW, (L, R1), dtype=torch.int32, device="cuda", generator=g
    )
    fetch_buf = torch.randint(
        1, 5000, (L, R1, FW), dtype=torch.int32, device="cuda", generator=g
    )
    indices = torch.zeros(L, MAXBS * CAP + 1, dtype=torch.int32, device="cuda")
    indptr = torch.zeros(L, MAXBS + 1, dtype=torch.int32, device="cuda")
    return kept_buf, kept_len, fetch_len, fetch_buf, indices, indptr


def _mk_fence(seed, slots, loc):
    """Per-(layer, slot) overflow flags (never on the trash slot), per-lane
    seq_len, and a req_to_token table with this step's slot NOT yet written
    (the pack must place `loc` at seq - 1 itself)."""
    g = torch.Generator(device="cuda").manual_seed(seed + 1000)
    fetch_ovf = torch.randint(
        0, 2, (L, R1), dtype=torch.int32, device="cuda", generator=g
    )
    fetch_ovf[:, TRASH] = 0
    seq = torch.randint(
        1, CAP, (slots.shape[0],), dtype=torch.int64, device="cuda", generator=g
    )
    r2t = torch.randint(
        20000, 30000, (R1, R2T), dtype=torch.int32, device="cuda", generator=g
    )
    return fetch_ovf, seq, r2t


def _torch_ref(backend_cls, state, slots, loc, bs, fence=None):
    """Drive the torch _pack_csr through a __new__-constructed backend."""
    kept_buf, kept_len, fetch_len, fetch_buf, indices, indptr = state
    be = backend_cls.__new__(backend_cls)
    be._kept_buf = {i: kept_buf[i] for i in range(L)}
    be._kept_len = {i: kept_len[i] for i in range(L)}
    be._fetch_len = {i: fetch_len[i] for i in range(L)}
    be._fetch_buf = {i: fetch_buf[i] for i in range(L)}
    be._graph_bufs = {i: {"indices": indices[i], "indptr": indptr[i]} for i in range(L)}
    for i in range(L):
        dense = None
        if fence is not None:
            fetch_ovf, seq, r2t = fence
            rows = r2t[slots, : int(seq.max())].to(torch.int64)
            rows.scatter_(1, (seq - 1)[:, None], loc[:, None])
            dense = (rows, seq, fetch_ovf[i].gather(0, slots) != 0)
        be._pack_csr(i, slots, loc, bs, bs, CAP, dense=dense)
    return state


class TestPackCsrParity(CustomTestCase):
    def _check(self, seed, lanes, fence):
        from sglang.srt.layers.attention.vestigekv.pack_csr import pack_csr_all_layers
        from sglang.srt.layers.attention.vestigekv_mla_backend import (
            VestigeKVMLABackend,
        )

        bs = len(lanes)
        slots = torch.tensor(lanes, dtype=torch.int64, device="cuda")
        loc = torch.randint(1, 9999, (bs,), dtype=torch.int64, device="cuda")
        fence_ops = _mk_fence(seed, slots, loc) if fence else None

        ref = _mk_state(seed)
        got = tuple(t.clone() for t in ref)
        _torch_ref(VestigeKVMLABackend, ref, slots, loc, bs, fence_ops)
        if fence:
            fetch_ovf, seq, r2t = fence_ops
            pack_csr_all_layers(
                slots, loc, *got, seq=seq, fetch_ovf=fetch_ovf, req_to_token=r2t
            )
        else:
            pack_csr_all_layers(slots, loc, *got)
        torch.cuda.synchronize()

        r_kb, r_kl, _, _, r_ix, r_ip = ref
        g_kb, g_kl, _, _, g_ix, g_ip = got
        self.assertTrue(torch.equal(r_kl, g_kl), f"kept_len seed={seed}")
        # trash-row CONTENT is excluded: colliding lanes write different
        # locs to the same cell and the torch scatter's winner is
        # undefined; nothing reads that row.
        self.assertTrue(
            torch.equal(r_kb[:, :TRASH], g_kb[:, :TRASH]),
            f"kept_buf seed={seed}",
        )
        for i in range(L):
            self.assertTrue(
                torch.equal(r_ip[i, : bs + 1], g_ip[i, : bs + 1]),
                f"indptr layer {i} seed={seed}",
            )
            # The row array itself is no longer written: decode stage 1
            # reads a lane's rows from the tiers, so what ships from this
            # pack is the per-lane COUNT stage 2 reduces over. A fenced
            # lane's count is still its whole row set, which is the part
            # the fence has to get right.
            if fence:
                fetch_ovf, seq, _ = fence_ops
                for lane in range(bs):
                    if lanes[lane] == TRASH or not int(fetch_ovf[i, lanes[lane]]):
                        continue
                    lo, hi = int(g_ip[i, lane]), int(g_ip[i, lane + 1])
                    self.assertEqual(
                        hi - lo,
                        int(seq[lane]),
                        f"fenced lane {lane} layer {i} seed={seed}",
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_matches_torch_chain(self):
        for seed, lanes in ((0, [2, 5, 0, TRASH]), (1, [7, TRASH, TRASH, TRASH])):
            self._check(seed, lanes, fence=False)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_fence_matches_torch_chain(self):
        for seed, lanes in ((2, [2, 5, 0, TRASH]), (3, [1, 3, 6, 7])):
            self._check(seed, lanes, fence=True)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_the_pack_writes_counts_and_the_append_and_nothing_else(self):
        # What the deployed decode still needs from this pack: the per-lane
        # counts stage 2 reduces over, and the append of this step's row into
        # the kept table. The row array must be left exactly as it was --
        # stage 1 reads the tiers, and a half-written array here would be
        # read by nothing and hide a real regression in the counts.
        from sglang.srt.layers.attention.vestigekv.pack_csr import pack_csr_all_layers

        lanes = [2, 5, 0, TRASH]
        slots = torch.tensor(lanes, dtype=torch.int64, device="cuda")
        loc = torch.arange(1, len(lanes) + 1, dtype=torch.int64, device="cuda")
        st = _mk_state(4)
        before_kept_len = st[1].clone()
        st[4].fill_(-7)  # a value nothing in the pack may overwrite
        pack_csr_all_layers(slots, loc, *st[:4], st[4], st[5])
        torch.cuda.synchronize()
        self.assertTrue((st[4] == -7).all(), "the row array must be left untouched")
        # kept_len is [layer, slot]: only the slots this step's lanes name
        # grow, and each of them by exactly one row. A padded lane appends to
        # the reserved trash slot, which is emptied before every pack, so it
        # is not part of the contract being checked here.
        for lay in range(st[1].shape[0]):
            for slot in {l for l in lanes if l != TRASH}:
                self.assertEqual(
                    int(st[1][lay, slot]),
                    int(before_kept_len[lay, slot]) + 1,
                    f"layer {lay} slot {slot} must have appended exactly one row",
                )
        self.assertTrue(
            (st[5][:, 1:] >= st[5][:, :-1]).all(), "indptr must be monotone"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
