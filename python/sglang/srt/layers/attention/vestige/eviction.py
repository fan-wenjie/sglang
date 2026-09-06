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


def sidecar_sigma(side: torch.Tensor, kappa: int = 16) -> torch.Tensor:
    """side: [B, 64] sidecar rows along the sequence axis. Returns [B] anomaly
    scores = residual of a kappa-band low-pass along dim 0."""
    f = torch.fft.rfft(side.float(), dim=0)
    f[kappa:] = 0
    low = torch.fft.irfft(f, n=side.shape[0], dim=0)
    return (side.float() - low).norm(dim=-1)


def select_kept(
    sigma: torch.Tensor,
    rho: float,
    closed: int,
    sinks: int = 4,
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
