# SPDX-License-Identifier: Apache-2.0
"""One page record carrying DSA's indexer key and VestigeKV's sketch.

Per decode step the sequence is read twice over by two kernels that want the
same rows: DSA's logits (39.89 us/step at 32k, measured) read the index key,
and the archive scan (37.21 us) reads the sketch and its residual norm. Both
are per token, both cover the layer's whole prefix, and neither needs anything
the other does not already touch. Merging them into one pass is what this
layout is for.

Two layouts, and which one wins is an empirical question this module does not
decide.

APPENDED keeps DSA's block where it is and puts VestigeKV's after it, so the
upstream accessors compute the same offsets and the fork's diff stays near
zero:

    page row = [ page_size x (index_head_dim + scale_elems*4) ]   DSA's, untouched
               [ page_size x (sketch_dim + 4) ]                   appended

INTERLEAVED puts both fields in one record per token, so the merged kernel
issues one contiguous load instead of two strided ones:

    token record = [ index_key ][ key_scale ][ sketch ][ rho ]

Appended is one pass over pages; interleaved is one pass over bytes. The
difference is not the traffic, which is identical, but what the compiler can
do with it -- a single contiguous stream leaves room for vectorisation and
pipelining that two strided regions do not, and that room can pull in further
passes that are not visible from the source. Which is why the choice is
measured rather than argued: every analytic prediction made about this decode
path today has been wrong, including that the archive scan's bytes drove the
slope (they do not; it is flat), that a pooled archive would pay (the sound
bound goes vacuous), and that halving the rank would (it costs more).

The sketch cannot be written when the token is: it is a projection onto a
basis fitted from calibration queries at build time, and the token enters the
pool long before. So the field is reserved at allocation and scattered into at
block close, which is where the archive is built today anyway -- the same
work, landing in a different place.

INVARIANT, and the hazard this layout creates. Both fields are functions of the
basis V: the sketch is content @ V.T and rho is ||content - sketch @ V||. Today
V, the sketch and rho are built together and replaced together, so they cannot
disagree. Putting them in a cache that outlives a build breaks that coupling --
a refit of V leaves every stored rho describing a residual against a basis that
no longer exists, and nothing downstream notices: the scan still runs, zp is
still calibrated, and the certificate's bound is simply wrong.

So the records for a (slot, layer) are valid for exactly that tier's current V,
and any path that refits V must rewrite both fields over the whole prefix
before the next scan. scatter_sketch takes the basis generation and asserts it
against the one the records were written under, because a silent mismatch here
is unrecoverable by any later check -- an incorrect bound is indistinguishable
from a correct one at the point it is used.

rho stays fp32 throughout. The sketch can be fp8 precisely because rho's term
absorbs its error; the term doing the absorbing cannot itself be approximate.
"""

import torch


def sketch_block_bytes(sketch_dim: int) -> int:
    """Bytes VestigeKV adds per token: the fp8 sketch plus an fp32 norm."""
    return sketch_dim + 4


def dsa_token_bytes(index_head_dim: int, quant_block_size: int) -> int:
    """Bytes DSA already stores per token: the fp8 key and its scale."""
    return index_head_dim + index_head_dim // quant_block_size * 4


def page_row_bytes(
    page_size: int, index_head_dim: int, quant_block_size: int, sketch_dim: int
) -> tuple[int, int]:
    """(DSA's bytes per page row, total bytes per page row) for APPENDED.

    The first return is the offset VestigeKV's block starts at, which is also
    exactly what the unmodified accessors address.
    """
    dsa = page_size * dsa_token_bytes(index_head_dim, quant_block_size)
    return dsa, dsa + page_size * sketch_block_bytes(sketch_dim)


def interleaved_token_bytes(
    index_head_dim: int, quant_block_size: int, sketch_dim: int
) -> int:
    """Bytes per token record when both fields ride together."""
    return dsa_token_bytes(index_head_dim, quant_block_size) + sketch_block_bytes(
        sketch_dim
    )


def interleaved_views(
    buf: torch.Tensor, *, page_size: int, index_head_dim: int,
    quant_block_size: int, sketch_dim: int,
):
    """(key, key_scale, sketch, rho) over one record per token, all aliasing.

    Here a flat slot id DOES index directly: every token's record is the same
    width with no gap, so the offset is affine in the slot and the whole buffer
    is one contiguous [n_slots, record] view. That is the property the
    appended layout gives up, and the reason this one is worth measuring.
    """
    rec = interleaved_token_bytes(index_head_dim, quant_block_size, sketch_dim)
    assert buf.element_size() == 1, f"page buffer must be byte-typed, got {buf.dtype}"
    assert buf.shape[-1] == page_size * rec, (
        f"page row is {buf.shape[-1]} B, interleaved layout says {page_size * rec}"
    )
    flat = buf.reshape(-1, rec)
    d = index_head_dim
    key = flat[:, :d].view(torch.float8_e4m3fn)
    f32 = flat.view(torch.float32)
    key_scale = f32[:, d // 4]
    sketch = flat[:, d + 4 : d + 4 + sketch_dim].view(torch.float8_e4m3fn)
    rho = f32[:, (d + 4 + sketch_dim) // 4]
    return key, key_scale, sketch, rho


def slot_index(slots: torch.Tensor, page_size: int):
    """Pool slot ids -> (page, row-within-page), the pair the views take.

    The appended block cannot be addressed by a flat slot id: a token's byte
    offset is page*row_bytes_total + dsa_bytes + r*row_bytes, which is not
    affine in page*page_size + r, so no 2-D strided view spans it. The views
    stay 3-D and the caller splits the id, exactly as DSA's own accessors do.
    """
    s = slots.to(torch.int64)
    return s // page_size, s % page_size


def sketch_views(buf: torch.Tensor, *, page_size: int, index_head_dim: int,
                 quant_block_size: int, sketch_dim: int):
    """(sketch, rho) over the appended block: [pages, page_size, sketch_dim]
    fp8 and [pages, page_size] fp32, both ALIASING `buf`.

    Three dimensions, not a flat slot axis: the gap DSA's block leaves in each
    page row makes the offset non-affine in a flat index, and a reshape that
    ignores it silently copies -- which would defeat the single pass this
    layout exists for. Index with slot_index().
    """
    dsa_bytes, total = page_row_bytes(
        page_size, index_head_dim, quant_block_size, sketch_dim
    )
    assert buf.element_size() == 1, f"page buffer must be byte-typed, got {buf.dtype}"
    assert buf.shape[-1] == total, (
        f"page row is {buf.shape[-1]} B, layout says {total}; the appended block "
        "is missing or a stale allocation is in play"
    )
    row = sketch_block_bytes(sketch_dim)
    pages = buf.shape[0]
    tail = buf.as_strided(
        size=(pages, page_size, row), stride=(total, row, 1), storage_offset=dsa_bytes
    )
    sketch = tail[..., :sketch_dim].view(torch.float8_e4m3fn)
    # the norm is the row's last 4 bytes; fp32 cannot be reached through the
    # fp8 view, so it takes its own
    rho = tail.view(torch.float32)[..., sketch_dim // 4]
    return sketch, rho


def scatter_sketch(
    buf: torch.Tensor,
    slots: torch.Tensor,
    sketch: torch.Tensor,
    rho: torch.Tensor,
    *,
    page_size: int,
    index_head_dim: int,
    quant_block_size: int,
    basis_gen: int,
    record_gen: torch.Tensor,
) -> None:
    """Write a block's sketches into the records of `slots`.

    Runs at block close, not per step. The rows are scattered because the pool
    hands out slots in page order and a request's positions are not contiguous
    in it; that is the same address-space difference the archive's own index
    tables exist to bridge, paid once here instead of on every scan.
    """
    assert rho.dtype == torch.float32, (
        f"rho must stay fp32: it is the term that absorbs the sketch's own "
        f"quantisation error, so it cannot carry any of its own (got {rho.dtype})"
    )
    # Device-side and vectorised: record_gen is [n_slots] int32, 0 meaning
    # never written. Reading it back per slot would sync the stream and walk
    # the prefix on the host, which is the cost this whole layout exists to
    # remove, so the check is an async assert like the pool's own range guards.
    prev = record_gen[slots.to(torch.int64)]
    torch._assert_async(((prev == 0) | (prev == basis_gen)).all())
    dst_s, dst_r = sketch_views(
        buf, page_size=page_size, index_head_dim=index_head_dim,
        quant_block_size=quant_block_size, sketch_dim=sketch.shape[1],
    )
    pg, rw = slot_index(slots, page_size)
    dst_s[pg, rw] = sketch
    dst_r[pg, rw] = rho
    record_gen[slots.to(torch.int64)] = basis_gen
