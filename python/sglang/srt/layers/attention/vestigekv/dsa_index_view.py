# SPDX-License-Identifier: Apache-2.0
"""Address DSA's own index-key cache as tier 1's salience branch.

On a rope-less MLA the DSA indexer key IS the salience channel (geometry.py):
never rotated, so a row's key means the same thing wherever the row sits, which
is what sigma needs and what a RoPE model cannot offer. DSA already stores that
key for every token of every layer, quantised by the same act_quant math the
salience ring used to repeat, so the ring was a second copy of bytes already in
the pool.

Layout, shared with the indexer and not visible from here: the buffer is
page-major, (num_pages, page_size * (index_head_dim + scale_elems * 4)), so a
global token slot s lives at page s // page_size, row s % page_size -- exactly
the row order a flat [n_slots, row_bytes] view gives, which is why the slot ids
in req_to_token index it directly.

Two views over the one buffer, no copy: the fp8 key at columns [0, dim), and
the fp32 scale at the last float32 column. The scale cannot be reached through
the fp8 view -- it is a 4-byte value inside a 1-byte element type -- which is
why this returns a pair rather than one tensor.
"""

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D


def index_key_views(buf: torch.Tensor, *, index_head_dim: int, quant_block_size: int):
    """(keys, scale) over `buf`: [n_slots, row_bytes] fp8 and [n_slots] fp32.

    Both alias `buf`; neither allocates. Feed them to
    eviction.blockwise_sigma_from_pool with offset=0 and dim=index_head_dim,
    whose fused kernel already takes exactly this pair because the ring it was
    written for mimicked this layout.
    """
    assert buf.element_size() == 1, (
        f"index-k buffer must be byte-typed to re-view; got {buf.dtype}"
    )
    scale_elems = index_head_dim // quant_block_size
    assert scale_elems == 1, (
        f"sigma takes one scale per row; this geometry has {scale_elems} "
        f"(index_head_dim {index_head_dim}, quant_block_size {quant_block_size})"
    )
    row_bytes = index_head_dim + scale_elems * 4
    flat = buf.reshape(-1, row_bytes)
    keys = flat.view(torch.float8_e4m3fn)
    # the scale occupies the row's last 4 bytes, i.e. float32 column dim // 4
    scale = flat.view(torch.float32)[:, index_head_dim // 4]
    return keys, scale


def index_sigma(
    *,
    buf: torch.Tensor,
    slots: torch.Tensor,
    index_head_dim: int,
    quant_block_size: int,
    block: int = D.CLOSE_BLOCK,
) -> torch.Tensor:
    """Tier-1 sigma of `slots` (whole blocks) read from the index-k cache.

    Unlike the ring this replaces, every live slot is present: DSA needs the
    same keys to attend at all, so there is no missing-key case to score +inf
    and no stamp to check.
    """
    from sglang.srt.layers.attention.vestigekv.eviction import (
        blockwise_sigma_from_pool,
    )

    keys, scale = index_key_views(
        buf, index_head_dim=index_head_dim, quant_block_size=quant_block_size
    )
    return blockwise_sigma_from_pool(
        keys, slots, block, offset=0, dim=index_head_dim, scale=scale
    )
