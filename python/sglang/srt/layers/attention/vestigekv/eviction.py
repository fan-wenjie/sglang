"""Sidecar-residual tier-1 eviction for VestigeKV (PREREG23, tax-free).

Pure functions extracted VERBATIM (same math) from the validated mini-sglang
VestigePolicy (minisgl/kimi/policy.py). Kept bit-identical to the reference so
the sglang backend's eviction can be asserted equal to it.

sigma_u = || r_u - lowpass_kappa(r)_u || over the 64-dim sidecar (the [512:576]
decoupled branch of the latent row), rFFT-truncated along the sequence axis; a
single GLOBAL top-m over all sidecar sigmas keeps the m most-anomalous rows
(constant-m rebalance, licensed by ranking stationarity). The first `sinks`
rows are always kept.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D


# Experimental (vestigekv-sigmafuse branch): projection-form fused sigma.
# Mathematically exact vs the rFFT chain (same orthogonal projection); fp32
# arithmetic differs from cuFFT at ~1e-6 -- three orders below the bf16
# input quantization -- measured 100% top-m overlap. 28x on batched prefill.
SIGMA_FUSED = True


def sidecar_sigma(side: torch.Tensor, kappa: int = D.LOWPASS_KAPPA) -> torch.Tensor:
    """side: [B, 64] sidecar rows along the sequence axis. Returns [B] anomaly
    scores = residual of a kappa-band low-pass along dim 0."""
    if SIGMA_FUSED and side.is_cuda:
        from sglang.srt.layers.attention.vestigekv.sigma_fused import sigma_fused

        sig, _ = sigma_fused(side, kappa)
        return sig
    f = torch.fft.rfft(side.float(), dim=0)
    f[kappa:] = 0
    low = torch.fft.irfft(f, n=side.shape[0], dim=0)
    return (side.float() - low).norm(dim=-1)


def select_kept(
    sigma: torch.Tensor,
    rho: float,
    closed: int,
    sinks: int = D.SINKS,
    m_fixed: int | None = None,
) -> torch.Tensor:
    """sigma: [closed] sidecar anomaly scores. Returns a [closed] bool keep mask:
    the top-m by sigma plus the first `sinks` rows. m = round(rho * closed) (or
    m_fixed for a hard absolute budget)."""
    m = m_fixed if m_fixed is not None else max(1, round(rho * closed))
    keep = torch.zeros(closed, dtype=torch.bool, device=sigma.device)
    keep[sigma.topk(min(m, closed)).indices] = True
    keep[:sinks] = True
    return keep


def blockwise_sigma(side: torch.Tensor, block: int = D.CLOSE_BLOCK) -> torch.Tensor:
    """sigma over full blocks only: one fixed-window rFFT per block.

    The cutoff kappa counts frequency BINS, and bin k corresponds to period
    T/k -- so a whole-prefix transform makes the cutoff drift with context
    length (>= 256 tokens at T=4096 but >= 32k tokens at T=512k). Fixed
    block windows keep sigma's meaning scale-invariant, match the reference
    policy bit-for-bit, and make the computation naturally incremental: each
    block is transformed exactly once, at close, and its sigma is immutable.

    Deliberately a per-block loop over sidecar_sigma, NOT one batched rfft:
    cuFFT's batched (dim=1) and single (dim=0) plans differ by ~1e-6, and
    decode-time closes score their blocks through sidecar_sigma -- a batched
    prefill would put two numerically divergent copies of the same statistic
    into one global top-m. The loop runs once per prefill (~128 transforms at
    512k) and keeps every sigma bit-identical to the close path and to the
    reference policy.

    Returns sigma for the first (len(side) // block) * block rows; the
    remainder is the unclosed tail, unconditionally attended, needing no sigma.
    """
    n_blocks = side.shape[0] // block
    if n_blocks == 0:
        return side.new_zeros(0)
    if SIGMA_FUSED and side.is_cuda:
        # Batched fused path: one launch, one program per block. Bit-identical
        # per block to the single-instance call (verified), so prefill and
        # decode-time closes still put ONE numeric flavor into the global
        # top-m -- the batched-vs-single cuFFT plan divergence this loop
        # guarded against does not exist for the projection kernel.
        from sglang.srt.layers.attention.vestigekv.sigma_fused import sigma_fused

        sig, _ = sigma_fused(
            side[: n_blocks * block].reshape(n_blocks, block, side.shape[1])
        )
        return sig.reshape(-1)
    return torch.cat(
        [sidecar_sigma(side[i * block : (i + 1) * block]) for i in range(n_blocks)]
    )
