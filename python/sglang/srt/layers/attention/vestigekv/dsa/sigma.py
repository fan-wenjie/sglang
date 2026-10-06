"""Group-space sigma over DSA's index-k: this arm's tier-1 ranking signal.

The projection itself stays shared (`vestigekv.sigma_fused`): it is an orthogonal
projection onto a trigonometric subspace, the same arithmetic whatever is being
projected, and a second copy of a Triton kernel is a drift risk with nothing to
show for it. What is arm-specific -- and what lives here -- is the SIGNAL and the
window. This arm transforms DSA's own index-k, one 128-dim key per `index_kpool`
positions, over a window of `CLOSE_BLOCK / pool` groups; the mla arm transforms
the latent row's 64-dim salience branch over a window of CLOSE_BLOCK rows. They
are different statistics that happen to share a transform.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.vestigekv.dsa import defaults as D

SIGMA_FUSED = True
"""Projection-form sigma instead of the rFFT chain. Mathematically exact (same
orthogonal projection); fp32 arithmetic differs from cuFFT at ~1e-6, three orders
below the bf16 input quantization, measured 100% top-m overlap. 28x on batched
prefill."""


def group_sigma(
    keys: torch.Tensor, window: int, kappa: int = D.LOWPASS_KAPPA
) -> torch.Tensor:
    """sigma per index-k group, over full windows only.

    `keys`: [n_groups, index_head_dim] dequantised group keys in sequence order.
    Returns sigma for the first `(n_groups // window) * window` groups; the
    remainder is the un-closed tail, attended unconditionally, needing no sigma.

    Fixed windows rather than a whole-prefix transform: bin k is period T/k, so a
    whole-prefix cutoff would drift with context length (>= 256 tokens at T=4096
    but >= 32k at T=512k). A fixed window keeps sigma's meaning scale-invariant
    and makes the computation naturally incremental -- each window is transformed
    exactly once, at close, and its sigma is immutable afterwards.
    """
    n = keys.shape[0] // window
    if n == 0:
        return keys.new_zeros(0, dtype=torch.float32)
    if SIGMA_FUSED and keys.is_cuda:
        from sglang.srt.layers.attention.vestigekv.sigma_fused import sigma_fused

        sig, _ = sigma_fused(
            keys[: n * window].reshape(n, window, keys.shape[1]), kappa
        )
        return sig.reshape(-1)
    out = []
    for i in range(n):
        b = keys[i * window : (i + 1) * window].float()
        f = torch.fft.rfft(b, dim=0)
        f[kappa:] = 0
        out.append((b - torch.fft.irfft(f, n=window, dim=0)).norm(dim=-1))
    return torch.cat(out)
