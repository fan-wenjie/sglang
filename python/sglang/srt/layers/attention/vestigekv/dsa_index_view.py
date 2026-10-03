# SPDX-License-Identifier: Apache-2.0
"""Address DSA's own index-key cache as tier 1's salience branch.

On a rope-less MLA the DSA indexer key IS the salience channel (geometry.py):
never rotated, so a row's key means the same thing wherever the row sits, which
is what sigma needs and what a RoPE model cannot offer. DSA already stores that
key for every token of every layer, quantised by the same act_quant math the
salience ring used to repeat, so the ring was a second copy of bytes already in
the pool.

Layout, shared with the indexer and not visible from here: the buffer is
page-major, (num_pages, page_size * (index_head_dim + scale_elems * 4)), and
WITHIN a page the keys come first as page_size * index_head_dim bytes, then the
scales. That is what kernels/ops/attention/dsa/index_buf_accessor.py addresses
in all three of its canonical variants (GetK slices [: page_size * head_dim],
GetS slices [page_size * head_dim :]), so it is the layout, not a choice.

A flat [n_slots, head_dim + 4] view does NOT give that order, and this module
used to take one. It read key bytes as scales: on a written page the
dequantised key norms came back inf where the correct read gives ~16, and the
"scale" read 6.2e7 against the true constant 0.015625. Tier 1 ranks rows by a
statistic over these keys and its top-k IS the keep decision, so the kept set
was chosen from misaddressed bytes with nothing in any count or latency to show
it. The aiter accessor does use an interleaved view, which is why size alone
cannot tell the two apart -- a page of 8448 bytes is both 64 x 132 and
64 x 128 + 64 x 4.
"""

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D


def index_page_views(buf: torch.Tensor, *, index_head_dim: int,
                     quant_block_size: int, page_size: int):
    """((num_pages, page_size, dim) fp8, (num_pages, page_size) fp32) over `buf`.

    Three dimensions, not two: the key block's rows are contiguous within a
    page but the pages are strided past the scale block, so [n_slots, dim] is
    not expressible as a view. Callers index by (slot // page_size,
    slot % page_size); `index_rows` does that for them.
    """
    assert buf.element_size() == 1, (
        f"index-k buffer must be byte-typed to re-view; got {buf.dtype}"
    )
    scale_elems = index_head_dim // quant_block_size
    assert scale_elems == 1, (
        f"sigma takes one scale per row; this geometry has {scale_elems} "
        f"(index_head_dim {index_head_dim}, quant_block_size {quant_block_size})"
    )
    k_bytes = page_size * index_head_dim
    want = k_bytes + page_size * scale_elems * 4
    assert buf.shape[1] == want, (
        f"index-k page is {buf.shape[1]} bytes, expected {want} for page_size "
        f"{page_size} and index_head_dim {index_head_dim}"
    )
    keys = buf[:, :k_bytes].view(torch.float8_e4m3fn).view(
        buf.shape[0], page_size, index_head_dim
    )
    scale = buf[:, k_bytes:].reshape(-1).view(torch.float32).view(
        buf.shape[0], page_size
    )
    return keys, scale


def index_rows(
    buf: torch.Tensor,
    slots: torch.Tensor,
    *,
    index_head_dim: int,
    quant_block_size: int,
    page_size: int,
) -> torch.Tensor:
    """Dequantised index keys for `slots`, [n, index_head_dim] fp32.

    A gather, which is what the [n_slots, ROW] indexing it replaces also was.
    """
    keys, scale = index_page_views(
        buf,
        index_head_dim=index_head_dim,
        quant_block_size=quant_block_size,
        page_size=page_size,
    )
    p = torch.div(slots, page_size, rounding_mode="floor")
    t = slots - p * page_size
    return keys[p, t].float() * scale[p, t][:, None]


def index_sigma(
    *,
    buf: torch.Tensor,
    slots: torch.Tensor,
    index_head_dim: int,
    quant_block_size: int,
    page_size: int,
    block: int = D.CLOSE_BLOCK,
) -> torch.Tensor:
    """Tier-1 sigma of `slots` (whole blocks) read from the index-k cache.

    Every live slot is present: DSA needs the same keys to attend at all, so
    there is no missing-key case to score +inf and no stamp to check.

    Gathers and reduces unfused on purpose. sigma_fused_from_pool strides a
    row-major [n_slots, ROW] pool and this buffer is paged with a split key and
    scale block, so the fused addressing does not apply; this runs at block
    close, not per step.
    """
    from sglang.srt.layers.attention.vestigekv.eviction import blockwise_sigma

    rows = index_rows(
        buf,
        slots,
        index_head_dim=index_head_dim,
        quant_block_size=quant_block_size,
        page_size=page_size,
    )
    return blockwise_sigma(rows, block)
