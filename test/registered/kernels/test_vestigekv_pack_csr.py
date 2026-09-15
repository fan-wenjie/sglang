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

register_cuda_ci(est_time=30, suite="base-b-test-1-gpu-small")

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
            # compare per-lane segments, REAL lanes only: a duplicated
            # trash lane's segment contains the append cell, whose
            # collision winner the torch scatter leaves undefined (both
            # forms are self-consistent; trash-lane attention output is
            # discarded padding).
            for lane in range(bs):
                if lanes[lane] == TRASH:
                    continue
                lo, hi = int(r_ip[i, lane]), int(r_ip[i, lane + 1])
                self.assertTrue(
                    torch.equal(
                        r_ix[i, lo:hi].to(torch.int64),
                        g_ix[i, lo:hi].to(torch.int64),
                    ),
                    f"indices layer {i} lane {lane} seed={seed}",
                )
                if fence:
                    fetch_ovf, seq, r2t = fence_ops
                    if int(fetch_ovf[i, lanes[lane]]):
                        # the fenced lane IS the dense row set, slot last
                        n = int(seq[lane])
                        want = r2t[lanes[lane], : n - 1].tolist() + [int(loc[lane])]
                        self.assertEqual(g_ix[i, lo:hi].tolist(), want)
                        self.assertEqual(hi - lo, n)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_matches_torch_chain(self):
        for seed, lanes in ((0, [2, 5, 0, TRASH]), (1, [7, TRASH, TRASH, TRASH])):
            self._check(seed, lanes, fence=False)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_fence_matches_torch_chain(self):
        for seed, lanes in ((2, [2, 5, 0, TRASH]), (3, [1, 3, 6, 7])):
            self._check(seed, lanes, fence=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
