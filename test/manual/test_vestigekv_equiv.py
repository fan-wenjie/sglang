"""Cross-implementation equivalence: the vendored sglang VestigeKV core must be
bitwise-identical to the validated mini-sglang reference on identical inputs.

This is the audit-rule-6 test (two deliberate copies must be asserted equal, not
each merely "looks right"). The reference (minisgl.kimi.tier2 / policy) is
validated end-to-end to 512k (PREREG19/31/32); passing here transfers that
validation to the port's decision logic, leaving only the sglang I/O plumbing.

Run: python -m sglang.srt.layers.attention.vestigekv.test_vestige_equiv
(needs the mini-sglang source on sys.path; the test adds it.)
"""

import sys

import torch

sys.path.insert(0, "/home/user/fft/nope_kv/mini-sglang/python")

from sglang.srt.layers.attention.vestigekv.eviction import select_kept, sidecar_sigma
from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier as SglRecallTier


def _ref_recall_tier():
    import importlib.util

    p = "/home/user/fft/nope_kv/mini-sglang/python/minisgl/kimi/tier2.py"
    spec = importlib.util.spec_from_file_location("_ref_tier2", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.RecallTier


def _ref_policy_fns():
    # policy.py imports .mla_pool; load it as part of the package instead.
    import minisgl.kimi.policy as pol

    return pol.VestigePolicy


def test_eviction_equiv(dev):
    torch.manual_seed(0)
    B = 8192
    side = torch.randn(B, 64, device=dev)
    Pol = _ref_policy_fns()
    ref = Pol(num_slots=1, rho=1 / 32, block=4096, sinks=4, kappa=16)
    ref_sigma = ref._block_sigma(side)
    sgl_sigma = sidecar_sigma(side, kappa=16)
    assert torch.equal(ref_sigma, sgl_sigma), (
        f"sigma drift max|d|={(ref_sigma - sgl_sigma).abs().max():.3e}"
    )
    # top-m keep (reference inlines it in maybe_close; replicate its exact steps)
    closed = B
    m = max(1, round((1 / 32) * closed))
    ref_keep = torch.zeros(closed, dtype=torch.bool, device=dev)
    ref_keep[ref_sigma.topk(min(m, closed)).indices] = True
    ref_keep[:4] = True
    sgl_keep = select_kept(sgl_sigma, rho=1 / 32, closed=closed, sinks=4)
    assert torch.equal(ref_keep, sgl_keep), "keep-mask drift"
    print(
        f"PASS eviction equiv: sigma bitwise-equal, keep-mask equal (m={m}, kept={int(sgl_keep.sum())})"
    )


def test_recall_equiv(dev):
    torch.manual_seed(1)
    T, H = 4096, 8
    rows = torch.randn(T, 576, device=dev)
    keep = torch.zeros(T, dtype=torch.bool, device=dev)
    keep[torch.randperm(T, device=dev)[: T // 32]] = True
    keep[:4] = True
    n = 64
    q_cal = torch.randn(n, H, 576, device=dev)
    q_pos = torch.randint(T // 2, T, (n,), device=dev)

    Ref = _ref_recall_tier()
    r_ref, r_sgl = Ref(r=64, topj=16), SglRecallTier(r=64, topj=16)
    st_ref = r_ref.build(rows, keep, q_cal, q_pos)
    st_sgl = r_sgl.build(rows, keep, q_cal, q_pos)
    assert st_ref["zp"] == st_sgl["zp"] and st_ref["n_hard"] == st_sgl["n_hard"], (
        f"build stats differ: {st_ref} vs {st_sgl}"
    )
    # query equivalence over several decode steps
    torch.manual_seed(2)
    mism = 0
    for _ in range(16):
        qe = torch.randn(H, 576, device=dev)
        f_ref = r_ref.query(qe.clone())
        f_sgl = r_sgl.query(qe.clone())
        if not torch.equal(f_ref.sort().values, f_sgl.sort().values):
            mism += 1
    assert mism == 0, f"{mism}/16 query steps produced different fetch sets"
    print(
        f"PASS recall equiv: build stats identical (zp={st_sgl['zp']}, "
        f"n_hard={st_sgl['n_hard']}, arch={st_sgl['arch']}); 16/16 query steps identical"
    )


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    test_eviction_equiv(dev)
    test_recall_equiv(dev)
    print("ALL VESTIGEKV CORE EQUIV TESTS PASS")


if __name__ == "__main__":
    main()
