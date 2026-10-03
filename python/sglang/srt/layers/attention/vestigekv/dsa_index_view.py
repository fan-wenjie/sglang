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
                     quant_block_size: int, slots_per_page: int):
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
    # slots_per_page, not the pool's page_size: the write kernel takes them as
    # separate constexprs (PAGE_SIZE and SLOTS_PER_PAGE) and computes the scale
    # block's offset as slots_per_page * index_head_dim. They are equal in this
    # config and need not stay so.
    k_bytes = slots_per_page * index_head_dim
    want = k_bytes + slots_per_page * scale_elems * 4
    assert buf.shape[1] == want, (
        f"index-k page is {buf.shape[1]} bytes, expected {want} for "
        f"slots_per_page {slots_per_page} and index_head_dim {index_head_dim}"
    )
    keys = buf[:, :k_bytes].view(torch.float8_e4m3fn).view(
        buf.shape[0], slots_per_page, index_head_dim
    )
    scale = buf[:, k_bytes:].reshape(-1).view(torch.float32).view(
        buf.shape[0], slots_per_page
    )
    return keys, scale


def index_rows(
    buf: torch.Tensor,
    slots: torch.Tensor,
    *,
    index_head_dim: int,
    quant_block_size: int,
    slots_per_page: int,
) -> torch.Tensor:
    """Dequantised index keys at `slots` of this buffer, [n, dim] fp32.

    `slots` index THIS buffer, which is not the token slot space when the
    indexer pools. With index_kpool > 1 the write kernel stores one entry per
    group -- it runs only where ``pos % POOL_SIZE == POOL_SIZE - 1`` and the
    value is a max over the group's scores plus a positional bias -- so a token
    slot does not name a row here and three quarters of such reads are zero.
    Callers wanting a token's salience must go through the block table at the
    pooled position; this function does not guess which space it was handed.
    """
    keys, scale = index_page_views(
        buf,
        index_head_dim=index_head_dim,
        quant_block_size=quant_block_size,
        slots_per_page=slots_per_page,
    )
    p = torch.div(slots, slots_per_page, rounding_mode="floor")
    t = slots - p * slots_per_page
    return keys[p, t].float() * scale[p, t][:, None]


def index_sigma(
    *,
    buf: torch.Tensor,
    slots: torch.Tensor,
    index_head_dim: int,
    quant_block_size: int,
    slots_per_page: int,
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
        slots_per_page=slots_per_page,
    )
    return blockwise_sigma(rows, block)

def index_group_keys(
    buf: torch.Tensor,
    pool_ids: torch.Tensor,
    block_table_row: torch.Tensor,
    *,
    index_head_dim: int,
    slots_per_page: int,
    pool_size: int,
) -> torch.Tensor:
    """Dequantised POOLED index keys for `pool_ids`, [n, index_head_dim] fp32.

    This is the only correct way to reach a token's salience when the indexer
    pools. The write kernel
    (kpool_fp8_index._kpool_decode_update_and_maybe_write_cache_kernel) stores
    one entry per group of `pool_size` tokens and derives its address as

        pool_id        = pos // pool_size
        token_page_row = (pool_id // slots_per_page) * pool_size
        page           = block_tables[req, token_page_row]
        offset         = pool_id % slots_per_page

    so one index page carries slots_per_page * pool_size tokens' worth of
    entries and three of every four token pages hold none. That is why reading
    this buffer at token slots returns zeros for three quarters of them.

    The stored value is not a raw key: it is the softmax-weighted mean of the
    group's `pool_size` keys, weighted by (slot_score + positional bias), then
    Hadamard-rotated and fp8-quantised. The weighting is the model's own gate,
    which is the whole reason this pooling is cheaper to accept than one we
    would impose -- archive_pool.py's vacuous spread term came from pooling the
    rank-64 sketch by an UNWEIGHTED mean and bounding it by Cauchy-Schwarz.

    `block_table_row` is one request's row of the page table, indexed in token
    pages (what metadata.get_page_table_64() hands the writer).
    """
    keys, scale = index_page_views(
        buf,
        index_head_dim=index_head_dim,
        quant_block_size=index_head_dim,
        slots_per_page=slots_per_page,
    )
    pool_ids = pool_ids.to(torch.int64)
    row = torch.div(pool_ids, slots_per_page, rounding_mode="floor") * pool_size
    row = row.clamp_(0, block_table_row.shape[0] - 1)
    page = block_table_row.index_select(0, row).to(torch.int64)
    off = pool_ids - torch.div(
        pool_ids, slots_per_page, rounding_mode="floor"
    ) * slots_per_page
    return keys[page, off].float() * scale[page, off][:, None]


def group_sigma_window(block: int, pool_size: int) -> int:
    """Sigma's window in GROUPS for a token-space window of `block`.

    kappa counts frequency bins and bin k is period window/k, so sampling once
    per group over the same token span leaves the physical cutoff untouched:
    4096 tokens at kappa 16 cuts below 256 tokens, and 1024 groups at kappa 16
    cuts below 64 groups, which is the same 256 tokens. kappa therefore does
    NOT get rescaled. What does change is Nyquist -- the grouped signal cannot
    see structure finer than 2*pool_size tokens -- and at a 256-token cutoff
    that is not where sigma's information is.
    """
    assert block % pool_size == 0, (
        f"sigma window {block} must be a multiple of pool_size {pool_size}"
    )
    return block // pool_size


def indexer_query(indexer, q_lora: torch.Tensor) -> torch.Tensor:
    """The indexer's own query, [n_tokens, index_n_heads, index_head_dim] bf16.

    Mirrors DSA's _get_q_k_bf16 for the rope-less case: wq_b, reshape to heads,
    then rotate_activation. Not a reimplementation of the scoring path -- only
    the query, because the lean decode path skips the GEMMs that would produce
    it and a selector scoring pooled index keys needs one.

    The rotation matters and is why this is safe to compare against the stored
    keys: the write kernel Hadamard-rotates the pooled key and this rotates the
    query by the same orthogonal transform, so the inner product is the one the
    model computes. Dropping either rotation would leave a score that looks
    plausible and ranks differently.
    """
    from einops import rearrange

    from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation

    assert indexer.rope_head_dim == 0 or not getattr(indexer, "apply_rope", False), (
        "indexer_query handles the rope-less geometry only; a rotated query "
        "needs the positions and the rotary cache"
    )
    q, _ = indexer.wq_b(q_lora)
    q = rearrange(q, "l (h d) -> l h d", d=indexer.head_dim)
    return rotate_activation(q)


def group_scores(
    group_keys: torch.Tensor, q_idx: torch.Tensor, head_weights=None
) -> torch.Tensor:
    """One score per group for the whole layer, [n_groups].

    Best over index heads, which is the granularity DSA selects at and the
    granularity select_recall_telemetry already measures the oracle against:
    a row any head wants is a row the layer wants. With `head_weights` the
    heads are combined by the model's own gate instead, summed rather than
    maxed, because a gate is a weighting and not a tie-break.
    """
    # [n_heads, n_groups]
    logits = q_idx.float() @ group_keys.T.float()
    if head_weights is None:
        return logits.amax(dim=0)
    w = head_weights.float().reshape(-1, 1)
    return (logits * w).sum(dim=0)


def select_groups(scores: torch.Tensor, n_rows: int, pool_size: int) -> torch.Tensor:
    """Top `n_rows // pool_size` group ids by score, [k] int64.

    A FIXED budget, not a threshold: that is the whole difference from the
    certificate, and it is what makes the pooled unit usable at all
    (archive_pool.py: tier 2 "cannot [pool] because it certifies a threshold",
    and pooling "requires tier 2 to change its fire rule to a fixed budget
    first"). A budget cannot overflow, so there is no fallback by
    construction -- and no fallback RATE to read as a quality proxy either,
    which is why the oracle-recall telemetry has to replace it.

    n_rows is in ROWS so the budget is stated in DSA's units: index_topk = 2048
    rows is 512 groups of 4, which is exactly what DSA itself selects.
    """
    assert n_rows % pool_size == 0, (
        f"budget {n_rows} rows must be a multiple of pool_size {pool_size}"
    )
    k = min(n_rows // pool_size, scores.shape[0])
    if k <= 0:
        return scores.new_zeros(0, dtype=torch.int64)
    return scores.topk(k).indices.to(torch.int64)
