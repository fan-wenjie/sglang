"""The two-kernel CSR pack must match the torch _pack_csr bit-for-bit.

Two implementations of one contract exist on purpose (the torch chain serves
the legacy scan-graph path, the Triton pair serves the in-graph path); this
test is the divergence guard: same inputs -> identical kept-table mutation,
identical CSR indices content, identical indptr.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="base-b-test-1-gpu-small")

L, R1, CAP, FW, MAXBS = 3, 9, 64, 16, 4
TRASH = R1 - 1


def _mk_state(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kept_buf = torch.randint(
        1, 5000, (L, R1, CAP), dtype=torch.int32, device="cuda", generator=g
    )
    kept_len = torch.randint(
        1, CAP - 2, (L, R1), dtype=torch.int64, device="cuda", generator=g
    )
    fetch_len = torch.randint(
        0, FW, (L, R1), dtype=torch.int64, device="cuda", generator=g
    )
    fetch_buf = torch.randint(
        1, 5000, (L, R1, FW), dtype=torch.int32, device="cuda", generator=g
    )
    indices = torch.zeros(L, MAXBS * CAP + 1, dtype=torch.int32, device="cuda")
    indptr = torch.zeros(L, MAXBS + 1, dtype=torch.int32, device="cuda")
    return kept_buf, kept_len, fetch_len, fetch_buf, indices, indptr


def _torch_ref(backend_cls, state, slots, loc, bs):
    """Drive the torch _pack_csr through a __new__-constructed backend."""
    kept_buf, kept_len, fetch_len, fetch_buf, indices, indptr = state
    be = backend_cls.__new__(backend_cls)
    be._kept_buf = {i: kept_buf[i] for i in range(L)}
    be._kept_len = {i: kept_len[i] for i in range(L)}
    be._fetch_len = {i: fetch_len[i] for i in range(L)}
    be._fetch_buf = {i: fetch_buf[i] for i in range(L)}
    be._graph_bufs = {i: {"indices": indices[i], "indptr": indptr[i]} for i in range(L)}
    for i in range(L):
        be._pack_csr(i, slots, loc, bs, bs, CAP)
    return state


class TestPackCsrParity(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_matches_torch_chain(self):
        from sglang.srt.layers.attention.vestigekv.pack_csr import pack_csr_all_layers
        from sglang.srt.layers.attention.vestigekv_mla_backend import (
            VestigeKVMLABackend,
        )

        for seed, lanes in ((0, [2, 5, 0, TRASH]), (1, [7, TRASH, TRASH, TRASH])):
            bs = len(lanes)
            slots = torch.tensor(lanes, dtype=torch.int64, device="cuda")
            loc = torch.randint(1, 9999, (bs,), dtype=torch.int64, device="cuda")

            ref = _mk_state(seed)
            got = tuple(t.clone() for t in ref)
            _torch_ref(VestigeKVMLABackend, ref, slots, loc, bs)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
