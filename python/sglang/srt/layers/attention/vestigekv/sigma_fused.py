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
    r_ptr,  # FROM_POOL=0: [N,T,DD] gathered sidecars. =1: [pool,ROW] pool
    slots_ptr,  # FROM_POOL=1: [N*T] int64 pool row indices (else unused)
    scale_ptr,  # HAS_SCALE=1: [pool] fp32 per-row dequant scale (else unused)
    c_ptr,  # [T, 32] fp32 basis
    sig_ptr,  # [N, T] fp32 out
    hist_ptr,  # [N_BINS] int32 SHARED (atomic across instances)
    y_ptr,  # EMIT_Y or LOAD_Y: [N, KB, DD] fp32 partial/reduced projection
    pos_ptr,  # HAS_POS=1: [N*T] int32 position of each row inside the block
    T,
    NB: tl.constexpr,
    BT: tl.constexpr,
    DD: tl.constexpr,  # branch width
    KB: tl.constexpr,  # 32
    FROM_POOL: tl.constexpr,
    ROW: tl.constexpr,  # pool row width
    KV_OFF: tl.constexpr,  # branch offset inside the row
    HAS_SCALE: tl.constexpr,  # rows are fp8 with a per-row scale
    EMIT_Y: tl.constexpr,  # stop after pass 1 and store Y (this rank's partial)
    LOAD_Y: tl.constexpr,  # skip pass 1 and read Y (the reduced projection)
    HAS_POS: tl.constexpr,  # rows are not the block's 0..T-1 in order
):
    inst = tl.program_id(0)
    sig_ptr = sig_ptr + inst.to(tl.int64) * T
    if FROM_POOL:
        slots_ptr = slots_ptr + inst.to(tl.int64) * T
    else:
        r_ptr = r_ptr + inst.to(tl.int64) * T * DD
    kd = tl.arange(0, KB)[:, None] * DD + tl.arange(0, DD)[None, :]
    # pass 1: Y = C^T R   (KB x DD), fp32 ieee. A sum over rows, so it splits
    # over any partition of them: each holder projects its own rows against
    # their own basis rows and the partials add. LOAD_Y takes the sum back.
    y = tl.zeros([KB, DD], dtype=tl.float32)
    if LOAD_Y:
        y = tl.load(y_ptr + inst.to(tl.int64) * KB * DD + kd)
    for t0 in range(0, T * (0 if LOAD_Y else 1), BT):
        t = t0 + tl.arange(0, BT)
        m = t < T
        if FROM_POOL:
            # Strided read straight out of the 576-dim pool row: the branch is
            # a contiguous 64-dim slice at KV_OFF, so no [T,576] intermediate
            # is materialised (the caller used to gather then slice).
            sl = tl.load(slots_ptr + t, mask=m, other=0).to(tl.int64)
            r = tl.load(
                r_ptr + sl[:, None] * ROW + (KV_OFF + tl.arange(0, DD))[None, :],
                mask=m[:, None],
                other=0.0,
            ).to(tl.float32)
            if HAS_SCALE:
                r = r * tl.load(scale_ptr + sl, mask=m, other=0.0)[:, None]
        else:
            r = tl.load(
                r_ptr + t[:, None] * DD + tl.arange(0, DD)[None, :],
                mask=m[:, None],
                other=0.0,
            ).to(tl.float32)
        if HAS_POS:
            # t numbers this holder's rows; the basis is indexed by where the
            # row sits in the block, and the two coincide only when one holder
            # has all of them in order.
            pos = tl.load(pos_ptr + inst.to(tl.int64) * T + t, mask=m, other=0)
        else:
            pos = t
        c = tl.load(
            c_ptr + pos[:, None] * KB + tl.arange(0, KB)[None, :],
            mask=m[:, None],
            other=0.0,
        )
        y += tl.dot(tl.trans(c), r, input_precision="ieee")
    if EMIT_Y:
        tl.store(y_ptr + inst.to(tl.int64) * KB * DD + kd, y)
    # pass 2: residual norm + histogram
    for t0 in range(0, T * (0 if EMIT_Y else 1), BT):
        t = t0 + tl.arange(0, BT)
        m = t < T
        if FROM_POOL:
            # Strided read straight out of the 576-dim pool row: the branch is
            # a contiguous 64-dim slice at KV_OFF, so no [T,576] intermediate
            # is materialised (the caller used to gather then slice).
            sl = tl.load(slots_ptr + t, mask=m, other=0).to(tl.int64)
            r = tl.load(
                r_ptr + sl[:, None] * ROW + (KV_OFF + tl.arange(0, DD))[None, :],
                mask=m[:, None],
                other=0.0,
            ).to(tl.float32)
            if HAS_SCALE:
                r = r * tl.load(scale_ptr + sl, mask=m, other=0.0)[:, None]
        else:
            r = tl.load(
                r_ptr + t[:, None] * DD + tl.arange(0, DD)[None, :],
                mask=m[:, None],
                other=0.0,
            ).to(tl.float32)
        if HAS_POS:
            # t numbers this holder's rows; the basis is indexed by where the
            # row sits in the block, and the two coincide only when one holder
            # has all of them in order.
            pos = tl.load(pos_ptr + inst.to(tl.int64) * T + t, mask=m, other=0)
        else:
            pos = t
        c = tl.load(
            c_ptr + pos[:, None] * KB + tl.arange(0, KB)[None, :],
            mask=m[:, None],
            other=0.0,
        )
        recon = tl.dot(c, y, input_precision="ieee")
        d = r - recon
        sig = tl.sqrt(tl.sum(d * d, 1))
        tl.store(sig_ptr + t, sig, mask=m)
        bits = sig.to(tl.int32, bitcast=True)
        bin_ = bits >> 21  # top 11 bits: monotone for sigma >= 0, max 1020 < NB
        tl.atomic_add(hist_ptr + bin_, 1, mask=m)


def sigma_fused_from_pool(
    kbuf: torch.Tensor,
    slots: torch.Tensor,
    block: int,
    kappa: int = D.LOWPASS_KAPPA,
    *,
    offset: int = D.KV_LORA_RANK,
    dim: int = D.SIDECAR_DIM,
    scale: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    emit_y: bool = False,
    pos: torch.Tensor | None = None,
    basis_len: int | None = None,
):
    """sigma for the blocks of `slots`, read STRAIGHT from the pool.

    kbuf: [pool, ROW] bf16 rows; slots: [n_blocks*block] int64 pool indices.
    Identical arithmetic to sigma_fused(); the only difference is that the
    `dim`-wide branch at column `offset` is addressed inside the kernel
    instead of being gathered into a [T, ROW] intermediate and then sliced.
    `scale` [pool] fp32 marks an fp8 pool: each row is dequantized as
    row.float() * scale[row] before the projection.

    Split mode, for a sequence whose rows are held by more than one rank.
    `emit_y` stops after the projection and writes this holder's partial into
    `y` [n, 32, dim]; passing a `y` without `emit_y` skips the projection and
    finishes from the one handed in, which is where the summed partials go.
    `pos` [n*block] int32 gives each row's index inside the block, required
    once a holder's rows are not the block's own 0..block-1 in order, and
    `basis_len` is the length of the block those positions index -- `block`
    counts the rows THIS holder has, and the two stop being the same number
    the moment a block is split. Neither argument changes the single-holder
    path: the default is the fused kernel it has always been.
    """
    n = slots.numel() // block
    dev = kbuf.device
    C = _basis(block if basis_len is None else basis_len, kappa, dev)
    sig = torch.empty(max(n, 1), block, dtype=torch.float32, device=dev)
    hist = torch.zeros(N_BINS, dtype=torch.int32, device=dev)
    if n == 0:
        return sig.new_zeros(0), hist
    slots = slots[: n * block].contiguous().to(torch.int64)
    _sigma_fused_kernel[(n,)](
        kbuf,
        slots,
        kbuf if scale is None else scale,
        C,
        sig,
        hist,
        sig if y is None else y,
        slots if pos is None else pos,
        block,
        NB=N_BINS,
        BT=64,
        DD=dim,
        KB=_KB_PAD,
        FROM_POOL=1,
        ROW=kbuf.shape[-1],
        KV_OFF=offset,
        HAS_SCALE=scale is not None,
        EMIT_Y=emit_y,
        LOAD_Y=y is not None and not emit_y,
        HAS_POS=pos is not None,
        num_warps=4,
        num_stages=1,
    )
    return sig.reshape(-1), hist


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
        side,
        side,
        side,
        C,
        sig,
        hist,
        sig,  # y_ptr unused: this launcher has no split mode
        side,  # pos_ptr unused
        T,
        NB=N_BINS,
        BT=64,
        DD=DD,
        KB=_KB_PAD,
        FROM_POOL=0,
        ROW=DD,
        KV_OFF=0,
        HAS_SCALE=False,
        EMIT_Y=False,
        LOAD_Y=False,
        HAS_POS=False,
        num_warps=4,
        num_stages=1,
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
