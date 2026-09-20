"""Cross-implementation equivalence, self-contained.

History: this test used to assert the vendored sglang VestigeKV core equal to
the mini-sglang reference (minisgl/kimi/{policy,tier2}.py). That package only
exists on the original development machine, so the check could not run
anywhere else. It now asserts the shipped implementation equal to a NAIVE
in-test reference that re-implements the same math straight-line (audit rule
6: two deliberate copies, asserted equal -- neither trusted on its own).

The naive reference encodes the CURRENT math, not the historical scheme the
port started from. Where the two differ, the present is authoritative (see
defaults.py and the vestigekv-dev git history for the why of each change):

  historical (d08b1e155e)                     current
  -----------------------------------------   -------------------------------------
  gate quantile 0.03, hard-coded n>5          gate_alpha(tau)=0.05, n >= min_hard
  11-rung zp ladder, zp=2.0 when no evidence  closed-form conformal kthvalue over every
                                              query's best archived row, Z_MAX
  fp32 scoring everywhere                     storage-dtype scoring (bf16 side /
                                              fp16 sketch), quantize-then-calibrate
  SVD basis                                   eigh basis (same subspace, 4x faster)
  certificate unclamped                       cert clamped at ENTROPY_EPS

Layers of assertion:
1. eviction: the fused sigma kernel vs the naive rFFT chain -- decision
   equality (keep masks torch.equal) plus value closeness (~1e-6, the
   projection form vs cuFFT); the non-fused fallback is bitwise-checked
   against the naive chain.
2. recall build: shipped build vs naive build -- V bitwise equal (same eigh
   call), n_hard / thr_g / gate / archive size exactly equal; zp equal within
   operand rounding (the fused operand build is ~1 ulp off cuBLAS, the same
   gap the registered suite records under its 1% fire-set bar).
3. recall query: a naive copy of the decision math, reading the tier's OWN
   operands, vs tier.query (eager form) -- bitwise-equal fetch sets; and vs
   tier.query_fixed (fused scan kernel) -- the registered 1% fire-set bar
   (the fused scan sits ~1 ulp off the eager formulation at the fire
   boundary; see DEFECTS.md). Both the uncapped and the top-j capped forms.

Run: pytest engine/test/manual/test_vestigekv_equiv.py
"""

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv import eviction
from sglang.srt.layers.attention.vestigekv.eviction import select_kept
from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

KV = D.KV_LORA_RANK  # 512


# --------------------------------------------------------------------------
# Naive reference: tier-1 eviction (per-block rFFT low-pass residual)
# --------------------------------------------------------------------------


def _naive_sidecar_sigma(side: torch.Tensor, kappa: int) -> torch.Tensor:
    f = torch.fft.rfft(side.float(), dim=0)
    f[kappa:] = 0
    low = torch.fft.irfft(f, n=side.shape[0], dim=0)
    return (side.float() - low).norm(dim=-1)


def _naive_blockwise_sigma(side: torch.Tensor, block: int) -> torch.Tensor:
    n_blocks = side.shape[0] // block
    if n_blocks == 0:
        return side.new_zeros(0)
    return torch.cat(
        [
            _naive_sidecar_sigma(side[i * block : (i + 1) * block], D.LOWPASS_KAPPA)
            for i in range(n_blocks)
        ]
    )


def _naive_select_kept(sigma, rho, closed, sinks):
    m = max(1, round(rho * closed))
    keep = torch.zeros(closed, dtype=torch.bool, device=sigma.device)
    keep[sigma.topk(min(m, closed)).indices] = True
    keep[:sinks] = True
    return keep


# --------------------------------------------------------------------------
# Naive reference: tier-2 recall index (straight-line current math)
# --------------------------------------------------------------------------


def _naive_build(kbuf, row_slots, keep, q_cal, q_pos, r, recall_target, scale):
    """Returns a dict of operands and thresholds, computed with nothing but
    plain whole-tensor torch ops (no chunking, no fused kernels, no caches)."""
    sc_ = scale
    dev = row_slots.device
    with D.ieee_fp32_matmul():
        rows = kbuf[row_slots.to(torch.int64)].float()  # [T, 576]
        T = rows.shape[0]
        H = q_cal.shape[1]
        arch_idx = (~keep).nonzero().flatten()
        qe = q_cal.reshape(-1, D.LATENT_DIM).float()  # [n*H, 576]

        qcal_c = qe[:, :KV] - qe[:, :KV].mean(0, keepdim=True)
        _evals, evecs = torch.linalg.eigh(qcal_c.T @ qcal_c)
        V = evecs[:, -r:].T.flip(0)
        c = rows[:, :KV] @ V.T
        csk_all = c.half()
        rho_all = (rows[:, :KV] - c @ V).norm(dim=-1)

        # calibration labels: full-cache causal argmax, single shot
        qpos_r = q_pos.repeat_interleave(H)
        s = (qe @ rows.T) * sc_
        colk = torch.arange(T, device=dev)[None, :]
        s = s.masked_fill(colk > qpos_r[:, None], torch.finfo(torch.float32).min)
        tgt = s.argmax(-1)  # first max, matching the chunked running-max
        hard = ~keep[tgt]

        kept_rows = kbuf[row_slots[keep].to(torch.int64)].to(torch.bfloat16)
        s_kept = (qe.to(torch.bfloat16) @ kept_rows.T).float() * sc_
        p1 = torch.softmax(s_kept, -1)
        ent = -(p1 * p1.clamp_min(D.ENTROPY_EPS).log()).sum(-1)
        max1 = s_kept.max(-1).values

        n_hard = int(hard.sum())
        alpha = D.gate_alpha(recall_target)
        thr_g = (
            float(ent[hard].quantile(alpha))
            if n_hard >= D.min_hard(recall_target)
            else float("-inf")
        )
        if float((ent > thr_g).float().mean()) > D.GATE_SELF_DISABLE_FRACTION:
            thr_g = float("-inf")

        # zp: every query's best ARCHIVED row must be certified at or above
        # its true score (the engine's calibration target)
        s_arch = s.masked_fill(keep[None, :], torch.finfo(torch.float32).min)
        abest, atgt = s_arch.max(-1)
        has_arch = abest > torch.finfo(torch.float32).min
        n_cal_q = int(has_arch.sum())
        need_more = n_cal_q < D.min_hard(recall_target)
        if need_more:
            zp = D.Z_MAX
        else:
            qh = qe[has_arch]
            qskh = qh[:, :KV] @ V.T
            qresh = (qh[:, :KV] - qskh @ V).norm(dim=-1)
            apos = atgt[has_arch]
            tgt_side = rows[apos, KV:]
            tgt_csk = csk_all[apos]
            tgt_rho = rho_all[apos]
            idxs_t = (qh[:, KV:].to(torch.bfloat16).float() * tgt_side.float()).sum(
                -1
            ) + (qskh.half().float() * tgt_csk.float()).sum(-1)
            idxs_t = idxs_t * sc_
            cert_t = (qresh * tgt_rho * sc_ / (KV - r) ** 0.5).clamp_min(D.ENTROPY_EPS)
            z_req = (abest[has_arch] - idxs_t) / cert_t
            k = D.conformal_k(n_cal_q, recall_target)
            zp = min(float(z_req.kthvalue(k).values), D.Z_MAX)

    return {
        "V": V,
        "csk_all": csk_all,
        "rho_all": rho_all,
        "arch": row_slots.index_select(0, arch_idx),
        "zp": zp,
        "thr_g": thr_g,
        "n_hard": n_hard,
    }


def _naive_query(tier, qe):
    """The decision math, re-typed straight-line, reading the tier's own
    operands (V / kept rows / side / csk / rho / arch / zp / thr_g) so the
    comparison isolates the decision logic from the operand build."""
    sc_ = tier.scale
    qe = qe.float()
    if tier.kept_slots.shape[0] == 0:
        max1 = qe.new_full((qe.shape[0],), float("-inf"))
        gate = qe.new_ones(qe.shape[0], dtype=torch.bool)
    else:
        s_kept = (qe.to(torch.bfloat16) @ tier.kept_rows.T).float() * sc_
        max1 = s_kept.max(-1).values
        p1 = torch.softmax(s_kept, -1)
        ent = -(p1 * p1.clamp_min(D.ENTROPY_EPS).log()).sum(-1)
        gate = ent > tier.thr_g
        if not bool(gate.any()):
            return tier.arch[:0]
    qsk = qe[:, :KV] @ tier.V.T
    qres = (qe[:, :KV] - qsk @ tier.V).norm(dim=-1)
    idxs = (
        (qe[:, KV:].to(torch.bfloat16) @ tier.side.T).float()
        + (qsk.half() @ tier.csk.T).float()
    ) * sc_
    cert = (qres[:, None] * tier.rho[None, :]) * sc_ / (KV - tier.r) ** 0.5
    score = idxs + tier.zp * cert
    fire = (score > max1[:, None]) & gate[:, None]
    return tier.arch[fire.any(0)]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fixture(dev):
    torch.manual_seed(1)
    T, H, n = 4096, 8, 64
    kbuf = torch.randn(T, D.LATENT_DIM, device=dev).bfloat16()
    row_slots = torch.arange(T, device=dev)
    keep = torch.zeros(T, dtype=torch.bool, device=dev)
    keep[torch.randperm(T, device=dev)[: T // 32]] = True
    keep[: D.SINKS] = True
    q_cal = torch.randn(n, H, D.LATENT_DIM, device=dev).bfloat16()
    q_pos = torch.randint(T // 2, T, (n,), device=dev)
    tier = RecallTier(r=64)
    stats = tier.build(kbuf, row_slots, keep, q_cal, q_pos)
    naive = _naive_build(
        kbuf, row_slots, keep, q_cal, q_pos, 64, D.RECALL_TARGET, D.ATTN_SCALE
    )
    return tier, stats, naive


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_eviction_equiv(dev=None):
    dev = dev or _dev()
    torch.manual_seed(0)
    B, block = 8192, D.CLOSE_BLOCK
    side = torch.randn(B, D.SIDECAR_DIM, device=dev).bfloat16()

    naive_sig = _naive_blockwise_sigma(side, block)

    # Non-fused fallback is the naive chain verbatim: bitwise equal.
    old = eviction.SIGMA_FUSED
    try:
        eviction.SIGMA_FUSED = False
        sig_ref = eviction.blockwise_sigma(side, block)
    finally:
        eviction.SIGMA_FUSED = old
    assert torch.equal(sig_ref, naive_sig), "fallback sigma drifted from naive rFFT"

    if dev.type == "cuda":
        # Fused projection form: ~1e-6 off cuFFT by design; the decision
        # (top-m keep mask) is what must not move.
        sig_fused = eviction.blockwise_sigma(side, block)
        d = (sig_fused - naive_sig).abs().max().item()
        assert d < 1e-4, f"fused-vs-naive sigma max|d|={d:.3e}"
        k_fused = select_kept(sig_fused, D.RHO, B)
        k_naive = _naive_select_kept(naive_sig, D.RHO, B, D.SINKS)
        assert torch.equal(k_fused, k_naive), "fused sigma moved the keep mask"

    # select_kept itself vs the naive selection, bitwise.
    k_ship = select_kept(naive_sig, D.RHO, B)
    k_ref = _naive_select_kept(naive_sig, D.RHO, B, D.SINKS)
    assert torch.equal(k_ship, k_ref), "keep-mask drift"

    # Pool-level slice audit: the sidecar is the pool row's LAST 64 channels
    # ([KV_LORA_RANK:576]). A historical bug sliced [128:576] instead; the
    # checks above take a pre-sliced [B, 64] input and cannot catch that
    # class, so assert the pool-addressed path against naive slicing here.
    kbuf = torch.randn(B, D.LATENT_DIM, device=dev).bfloat16()
    slots = torch.arange(B, device=dev)
    naive_pool = _naive_blockwise_sigma(kbuf[:, KV:], block)
    # Non-fused pool fallback gathers then transforms: bitwise vs naive.
    old = eviction.SIGMA_FUSED
    try:
        eviction.SIGMA_FUSED = False
        sig_pool_ref = eviction.blockwise_sigma_from_pool(kbuf, slots, block)
    finally:
        eviction.SIGMA_FUSED = old
    assert torch.equal(sig_pool_ref.float(), naive_pool), (
        "pool-path slice/transform drifted from naive [KV_LORA_RANK:] slicing"
    )
    if dev.type == "cuda":
        sig_pool = eviction.blockwise_sigma_from_pool(kbuf, slots, block)
        kp = select_kept(sig_pool, D.RHO, (B // block) * block)
        kp_ref = _naive_select_kept(naive_pool, D.RHO, (B // block) * block, D.SINKS)
        assert torch.equal(kp, kp_ref), "fused pool sigma moved the keep mask"

    print(
        f"PASS eviction equiv: fallback bitwise, fused decision-equal "
        f"(kept={int(k_ship.sum())}/{B}); pool slice audited"
    )


def test_recall_build_equiv(dev=None):
    dev = dev or _dev()
    if dev.type != "cuda":
        return  # the fused operand build is the interesting path; CPU-only skip
    tier, stats, naive = _fixture(dev)

    assert torch.equal(tier.V, naive["V"]), "sketch basis drifted (same eigh call)"
    assert stats["arch"] == int(naive["arch"].numel()), "archive size differs"
    assert stats["n_hard"] == naive["n_hard"], (
        f"n_hard differs: {stats['n_hard']} vs {naive['n_hard']}"
    )
    assert tier.thr_g == naive["thr_g"], (
        f"gate threshold differs: {tier.thr_g} vs {naive['thr_g']}"
    )
    assert stats["gate_off"] == (naive["thr_g"] == float("-inf"))
    # zp rides on the fused operand build (~1 ulp off cuBLAS): compare within
    # the operand-rounding envelope, not bitwise.
    dz = abs(tier.zp - naive["zp"])
    assert dz <= 0.05, f"zp drift {tier.zp} vs naive {naive['zp']} (|d|={dz:.3e})"
    print(
        f"PASS recall build equiv: V bitwise, n_hard={stats['n_hard']}, "
        f"thr_g={tier.thr_g:.6g}, zp {tier.zp:.6g} vs naive {naive['zp']:.6g}"
    )


def test_recall_query_equiv(dev=None):
    dev = dev or _dev()
    if dev.type != "cuda":
        return
    tier, _stats, _naive = _fixture(dev)
    # W must cover the worst fire: query_fixed keeps the first W fired
    # rows while the eager forms return the full fired set. 4096 (the
    # --vestigekv-recall-capacity default) sits above the observed worst
    # fire; the fixture's archive is ~3968 rows and CAN fire nearly
    # whole under the Z_MAX safety clamp.
    W = 4096
    out = torch.zeros(1, W, dtype=torch.int64, device=dev)
    out_len = torch.zeros(1, dtype=torch.int64, device=dev)
    out_ovf = torch.zeros(1, dtype=torch.int32, device=dev)
    torch.manual_seed(2)
    mism_eager = 0
    worst_fixed = 0.0
    for _ in range(16):
        qe = torch.randn(8, D.LATENT_DIM, device=dev).bfloat16()
        f_naive = _naive_query(tier, qe.clone()).sort().values
        f_eager = tier.query(qe.clone()).sort().values
        tier.query_fixed(qe.clone(), out, out_len, out_ovf, 0)
        f_fixed = out[0, : int(out_len[0])].sort().values.to(f_naive.dtype)
        if not torch.equal(f_naive, f_eager):
            mism_eager += 1
        # query_fixed's fused scan sits ~1 ulp off the eager formulation
        # at the fire boundary (DEFECTS.md; the registered bar is 1% of
        # the fired set, twice the observed worst 0.48%).
        sym = len(set(f_naive.tolist()) ^ set(f_fixed.tolist()))
        worst_fixed = max(worst_fixed, sym / max(1, f_naive.numel()))
    assert mism_eager == 0, f"{mism_eager}/16 eager mismatches"
    assert worst_fixed <= 0.01, (
        f"query_fixed worst fire-set disagreement "
        f"{100 * worst_fixed:.2f}% over 16 steps"
    )
    print(
        f"PASS recall query equiv: eager 16/16 bitwise; "
        f"query_fixed worst fire-set gap {100 * worst_fixed:.3f}% (bar 1%)"
    )


def main():
    dev = _dev()
    print(f"device: {dev}")
    test_eviction_equiv(dev)
    test_recall_build_equiv(dev)
    test_recall_query_equiv(dev)
    print("ALL VESTIGEKV CORE EQUIV TESTS PASS (self-contained naive reference)")


if __name__ == "__main__":
    main()
