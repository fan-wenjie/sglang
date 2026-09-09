"""Fused sigma operator: projection-form low-pass residual + radix histogram.

Mathematical simplification (exact): the rFFT -> low-pass mask -> irFFT chain
equals orthogonal projection onto the (2*kappa-1)-dim trigonometric subspace,
    sigma_u = || r_u - c_u^T (C^T R) ||,
with C the real Fourier basis over the FIXED close window (T = CLOSE_BLOCK),
so C is a startup constant. One Triton program per block does two passes:
  pass 1: Y = C^T R              (KBxD fp32, ieee accumulate)
  pass 2: sigma_u = ||R_u - c_u^T Y||, plus a 1024-bin histogram of sigma's
          fp32 bit pattern (monotone for sigma >= 0) -- the counting pass of
          a radix select, fused. Block histograms are immutable (each block's
          sigma is computed exactly once), so a per-record global histogram
          is maintained by summation and top-m selection needs only a bin
          threshold plus exact refinement inside the boundary bin.

Numerics: fp32-ieee accumulation from bf16 inputs (multiply-then-add rule);
the fp32 arithmetic differs from cuFFT at ~1e-7 rms, three orders below the
bf16 input quantization (1.2e-3 rms) -- top-m selection is insensitive
except at exact ties, where any tie-break is valid (Property 1).
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.vestigekv import defaults as D

N_BINS = 1024
_KB_PAD = 32  # 2*kappa-1 = 31 padded to a tl.dot-friendly 32 (zero column)

_basis_cache: dict = {}


def _basis(T: int, kappa: int, dev) -> torch.Tensor:
    """Real Fourier basis [T, 32]: DC + (kappa-1) cos/sin pairs + zero pad."""
    key = (T, kappa, str(dev))
    b = _basis_cache.get(key)
    if b is not None:
        return b
    u = torch.arange(T, dtype=torch.float64)
    cols = [torch.full((T,), 1.0 / T**0.5, dtype=torch.float64)]
    for k in range(1, kappa):
        ang = 2.0 * torch.pi * k * u / T
        cols.append(torch.cos(ang) * (2.0 / T) ** 0.5)
        cols.append(torch.sin(ang) * (2.0 / T) ** 0.5)
    C = torch.stack(cols, 1)  # [T, 2k-1]
    pad = torch.zeros(T, _KB_PAD - C.shape[1], dtype=torch.float64)
    b = torch.cat([C, pad], 1).float().to(dev).contiguous()
    _basis_cache[key] = b
    return b


@triton.jit
def _sigma_fused_kernel(
    r_ptr,      # [N, T, 64] bf16 sidecar blocks (contiguous)
    c_ptr,      # [T, 32] fp32 basis
    sig_ptr,    # [N, T] fp32 out
    hist_ptr,   # [N_BINS] int32 SHARED (atomic across instances)
    T,
    NB: tl.constexpr,
    BT: tl.constexpr,
    DD: tl.constexpr,   # 64
    KB: tl.constexpr,   # 32
):
    inst = tl.program_id(0)
    r_ptr = r_ptr + inst.to(tl.int64) * T * DD
    sig_ptr = sig_ptr + inst.to(tl.int64) * T
    # pass 1: Y = C^T R   (KB x DD), fp32 ieee
    y = tl.zeros([KB, DD], dtype=tl.float32)
    for t0 in range(0, T, BT):
        t = t0 + tl.arange(0, BT)
        m = t < T
        r = tl.load(r_ptr + t[:, None] * DD + tl.arange(0, DD)[None, :],
                    mask=m[:, None], other=0.0).to(tl.float32)
        c = tl.load(c_ptr + t[:, None] * KB + tl.arange(0, KB)[None, :],
                    mask=m[:, None], other=0.0)
        y += tl.dot(tl.trans(c), r, input_precision="ieee")
    # pass 2: residual norm + histogram
    for t0 in range(0, T, BT):
        t = t0 + tl.arange(0, BT)
        m = t < T
        r = tl.load(r_ptr + t[:, None] * DD + tl.arange(0, DD)[None, :],
                    mask=m[:, None], other=0.0).to(tl.float32)
        c = tl.load(c_ptr + t[:, None] * KB + tl.arange(0, KB)[None, :],
                    mask=m[:, None], other=0.0)
        recon = tl.dot(c, y, input_precision="ieee")
        d = r - recon
        sig = tl.sqrt(tl.sum(d * d, 1))
        tl.store(sig_ptr + t, sig, mask=m)
        bits = sig.to(tl.int32, bitcast=True)
        bin_ = bits >> 21  # top 11 bits: monotone for sigma >= 0, max 1020 < NB
        tl.atomic_add(hist_ptr + bin_, 1, mask=m)


def sigma_fused(side: torch.Tensor, kappa: int = D.LOWPASS_KAPPA):
    """sigma + histogram. side: [T,64] (one block) or [N,T,64] (batched:
    one program per block, ONE launch; histograms atomically accumulate
    into a single shared table -- correct because per-record selection sums
    block histograms anyway)."""
    single = side.dim() == 2
    if single:
        side = side.unsqueeze(0)
    N, T, DD = side.shape
    dev = side.device
    C = _basis(T, kappa, dev)
    side = side.contiguous()
    sig = torch.empty(N, T, dtype=torch.float32, device=dev)
    hist = torch.zeros(N_BINS, dtype=torch.int32, device=dev)
    _sigma_fused_kernel[(N,)](
        side, C, sig, hist, T, NB=N_BINS, BT=64, DD=DD, KB=_KB_PAD,
        num_warps=4, num_stages=1,
    )
    return (sig[0], hist) if single else (sig, hist)


def topm_from_hist(sig: torch.Tensor, hist: torch.Tensor, m: int):
    """Exact top-m indices of sig using its (summed) histogram: bin threshold
    + exact refinement inside the boundary bin. Ties break arbitrarily,
    which Property 1 licenses. Returns int64 indices (unsorted)."""
    if m >= sig.numel():
        return torch.arange(sig.numel(), device=sig.device)
    csum = torch.flip(torch.cumsum(torch.flip(hist.long(), [0]), 0), [0])
    # csum[b] = #{rows with bin >= b}, non-increasing. Boundary bin b* =
    # LARGEST b with csum[b] >= m: then #{bin > b*} < m <= #{bin >= b*}.
    ok = (csum >= m).nonzero().flatten()
    b = int(ok.max())
    n_above = int(csum[b + 1]) if b + 1 < N_BINS else 0
    bits = sig.view(torch.int32)
    bins = bits >> 21
    idx_above = (bins > b).nonzero().flatten()
    need = m - n_above  # > 0 by choice of b
    idx_bound = (bins == b).nonzero().flatten()
    take = torch.topk(sig[idx_bound], need).indices
    return torch.cat([idx_above, idx_bound[take]])
