"""Attention as two partitions: the sweep over the cache, and the join with this step's token.

    o = merge(sweep(q, cache), join(q, k_t, v_t))

Softmax attention is a mergeable aggregate. Partition the keys and values, attend over each part
separately, and the parts recombine exactly from `(output, log-partition)` -- which is what
`merge_state` does and what sglang already relies on to split attention across DCP ranks.

The partition this module cares about is the one the arrangement is built on:

    sweep   every position already in the KV cache. Reads the cache and the QUERY, and nothing
            else. This is the half that can run before x_l exists.
    join    the token this step is producing. Reads its key and value, which come from x_l, which
            comes back from the pool.

That asymmetry is the whole point, and it is a statement about DEPENDENCIES rather than about any
one arrangement. The sweep reads only the query and the history, so it can be issued the moment a
caller has a query -- and an arrangement whose query is in hand before the feed-forward it wants
to overlap runs the sweep, the expensive half, linear in the context length, while the pool works.
The join is one position per request and cannot start until the pool answers.

## Why this calls forward_decode twice rather than writing two kernels

`forward_decode` is a hundred lines of branching: kv scales, logit capping, sinks, xai temperature,
sliding windows, MLA, the unified pool's loc translation. A hand-written sweep would reimplement
some subset of that and drift from the rest, and the failure mode is not a crash -- it is an
attention output that is slightly wrong on the configurations the reimplementation forgot.

So both partitions go through `forward_decode` itself, with `kv_indptr`/`kv_indices` swapped for
the partition's own. Every branch that applies to the fused call applies identically to both
halves, because it is the same code.

## What is refused, and why refusing is the point

`split_refusal()` returns a sentence when this forward cannot be split. A silent fallback to the
fused path would report the fused path's latency under the split path's name, so the caller is
expected to record the refusal, not swallow it. Refused today:

    extend / prefill    a chunk's join is a causal attention among its own tokens, not one
                        position. `extend_attention_fwd` already takes `skip_prefix` /
                        `skip_extend` / `lse_extend` for DCP, so the same partition is reachable
                        there; it is not wired here, and prefill is not the regime the pool
                        exists for.
    sliding window      the cache the layer sweeps is not the cache `kv_indices` names
    MLA / DCP           forward_decode's own short-circuit returns, which merge on their own terms
    page_size > 1       kv_indices names pages; dropping "the last token" is not dropping the
                        last index
    cuda graph capture  the partition is built per forward from live seq_lens

## The empty prefix

A request in its first decode step after a one-token prompt has a prefix of length 0. The kernel
divides by a zero partition function there and writes NaN, with `lse = -inf` beside it saying the
partition is empty. `merge_state` weights that partition by exp(-inf) = 0, and 0 * NaN is NaN, so
the rows are zeroed before merging rather than after. Not a hypothetical: a prompt of one token
is a legal request.
"""

from __future__ import annotations

import torch

from sglang.srt.environ import envs

from sglang.srt.layers.attention.merge_state import merge_state

NEG_INF = float("-inf")


class SweepResult:
    """One partition's `(output, log-partition)`, held until the other half is ready.

    `q` is kept because the join must attend with the SAME query the sweep used. Recomputing it
    from the layer's own input would attend the cache with one query and this step's token with
    another -- a model that is not the model, and one whose output is fluent.
    """

    def __init__(
        self, layer_id: int, q: torch.Tensor, o: torch.Tensor, lse: torch.Tensor
    ):
        self.layer_id = layer_id
        self.q = q
        self.o = o
        self.lse = lse


class PerPassIndex:
    """The prefix partition's indices, built once per forward pass and reused by every layer.

    Keyed by the ForwardBatch itself, held by reference: `id()` alone would alias a freed batch
    onto a new one at the same address and serve a stale partition, which reads as a correct model
    with wrong attention.
    """

    def __init__(self) -> None:
        self._batch = None
        self._value = None

    def get(self, forward_batch, build):
        if self._batch is not forward_batch:
            self._batch = forward_batch
            self._value = build()
        return self._value

    def clear(self) -> None:
        self._batch = None
        self._value = None


def full_attn_backend(backend):
    """The backend that owns the KV cache this split partitions.

    A hybrid stack wraps two: the softmax layers' backend and the linear-attention one. The wrapper
    dispatches per layer, and every layer this module touches is on the full-attention side.
    """
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    if isinstance(backend, HybridLinearAttnBackend):
        return backend.full_attn_backend
    return backend


def split_refusal(backend, layer, forward_batch) -> str | None:
    """Why this forward cannot be split, or None if it can."""
    mode = forward_batch.forward_mode
    if not mode.is_decode():
        return f"forward mode {mode} is not decode; only decode's join is one position per request"
    backend = full_attn_backend(backend)
    if backend.__class__.__name__ != "TritonAttnBackend":
        return (
            f"{backend.__class__.__name__} is not the triton backend this split reads its "
            f"partition out of"
        )
    if backend.forward_metadata is None:
        return "the backend has no forward metadata; the split is built from it"
    if backend.dcp_size > 1:
        return "decode context parallelism already merges partial attention on its own terms"
    if backend.use_mla and not envs.SGLANG_AFD_SPLIT_MLA_DECODE.get():
        return (
            "MLA decode is not split by default: over a latent cache of one KV head the "
            "second pass and the merge cost more than the overlap they buy at the contexts "
            "measured (SGLANG_AFD_SPLIT_MLA_DECODE=1 turns it on)"
        )
    # MLA is no longer refused BY NATURE, and the sentence that used to refuse it was not true of this
    # backend: `TritonAttnBackend.forward_decode` has no MLA short-circuit. `use_mla` picks which
    # setter writes the KV cache and nothing else -- the sweep still runs `decode_attention_fwd`
    # over `kv_indptr`/`kv_indices`/`num_kv_splits` and still leaves its log-partition in
    # `attn_lse`, which is the whole of what this partition needs. The absorbed decode's widths
    # differ (a query of `kv_lora_rank + qk_rope_head_dim`, a value of `kv_lora_rank`, one KV
    # head) and every one of them is read off the layer rather than assumed.
    #
    # It mattered: `kimi_linear`'s seven full-attention layers are MLA, so refusing them shut the
    # sweep window on every attention that checkpoint has -- `sweeps: 0` against `fused_decodes:
    # 5857` in a live report -- and shift 1 was measured paying for a window that never opened.
    # Held to the fused call token-for-token on a 64-token greedy continuation, which is the
    # comparison that means something here: the two paths are meant to be the same arithmetic.
    if backend.page_size != 1:
        return (
            f"page_size={backend.page_size}: kv_indices names pages, so dropping the current "
            f"token is not dropping the last index"
        )
    if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
        return "a sliding-window layer sweeps window_kv_indices, not the indices this splits"
    if forward_batch.spec_info is not None:
        return "speculative decoding puts more than one candidate token in the join"
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return "cuda graph capture cannot hold a partition built from live seq_lens"
    return None


def _build_prefix(backend, forward_batch):
    """kv_indptr / kv_indices / num_kv_splits over every cached position but this step's.

    Each request's positions sit contiguously in `kv_indices`, its own last. Dropping exactly one
    element per request means the source of prefix element `i` is `i + (requests before it)` --
    no gather over a mask, and no device-to-host sync beyond the one `total` needs.
    """
    metadata = backend.forward_metadata
    kv_indptr, kv_indices = metadata.kv_indptr, metadata.kv_indices
    device = kv_indices.device
    bs = forward_batch.batch_size

    prefix_lens = (forward_batch.seq_lens[:bs] - 1).to(torch.int32)
    prefix_indptr = torch.zeros(bs + 1, dtype=kv_indptr.dtype, device=device)
    torch.cumsum(prefix_lens, dim=0, out=prefix_indptr[1:])

    seq_lens_sum = forward_batch.seq_lens_sum
    total = (
        int(seq_lens_sum) - bs if seq_lens_sum is not None else int(prefix_lens.sum())
    )
    if total <= 0:
        # every request is one token long: there is no cache to sweep, and a partition of nothing
        # is not a partition. The caller falls back rather than dividing by an empty sum.
        return None

    which = torch.repeat_interleave(
        torch.arange(bs, dtype=torch.int32, device=device),
        prefix_lens,
        output_size=total,
    )
    source = torch.arange(total, dtype=torch.int32, device=device) + which
    prefix_indices = kv_indices[source.long()]

    prefix_splits = torch.empty((bs,), dtype=torch.int32, device=device)
    backend.get_num_kv_splits(prefix_splits, prefix_lens)

    # this step's token is the last index of each request's run
    current_indices = kv_indices[(kv_indptr[1 : bs + 1] - 1).long()]
    current_indptr = torch.arange(bs + 1, dtype=kv_indptr.dtype, device=device)
    current_splits = torch.ones((bs,), dtype=torch.int32, device=device)
    return (
        prefix_indptr,
        prefix_indices,
        prefix_splits,
        current_indptr,
        current_indices,
        current_splits,
    )


def _run_partition(
    backend,
    layer,
    forward_batch,
    *,
    q,
    k,
    v,
    indptr,
    indices,
    splits,
    save_kv_cache: bool,
):
    """One partition, through forward_decode itself, with the partition's indices swapped in."""
    metadata = backend.forward_metadata
    saved = (metadata.kv_indptr, metadata.kv_indices, metadata.num_kv_splits)
    metadata.kv_indptr, metadata.kv_indices, metadata.num_kv_splits = (
        indptr,
        indices,
        splits,
    )
    # splits the kernel does not reach keep whatever the previous layer left; -inf is the identity
    # of the reduction, so an unwritten split contributes nothing instead of a stale partition
    metadata.attn_lse.fill_(NEG_INF)
    try:
        o = backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
        )
    finally:
        metadata.kv_indptr, metadata.kv_indices, metadata.num_kv_splits = saved
    tokens, heads = q.shape[0], layer.tp_q_head_num
    lse = torch.logsumexp(metadata.attn_lse[:tokens, :heads, :].float(), dim=-1)
    return o.view(tokens, heads, layer.v_head_dim), lse


def sweep(
    backend, layer, forward_batch, *, q, index: PerPassIndex
) -> SweepResult | None:
    """Attend the cache with the query alone. Returns None when the partition does not exist."""
    backend = full_attn_backend(backend)
    parts = index.get(forward_batch, lambda: _build_prefix(backend, forward_batch))
    if parts is None:
        return None
    prefix_indptr, prefix_indices, prefix_splits = parts[0], parts[1], parts[2]
    o, lse = _run_partition(
        backend,
        layer,
        forward_batch,
        q=q,
        k=None,
        v=None,
        indptr=prefix_indptr,
        indices=prefix_indices,
        splits=prefix_splits,
        save_kv_cache=False,
    )
    # An empty prefix leaves NaN beside an -inf log-partition; zero it before it meets a
    # weight. Applied unconditionally: asking `.any()` first would read a device tensor on the
    # host once per converted layer per token, and a sync inside the window this whole module
    # exists to keep busy is the one place a correctness guard must not cost anything.
    empty = (lse == NEG_INF).unsqueeze(-1)
    o = torch.where(empty, torch.zeros_like(o), o)
    return SweepResult(layer.layer_id, q, o, lse)


def join(
    backend, layer, forward_batch, *, k, v, state: SweepResult, index: PerPassIndex
):
    """Fold this step's token into the swept cache. Writes the KV cache, as the fused call does."""
    backend = full_attn_backend(backend)
    parts = index.get(forward_batch, lambda: _build_prefix(backend, forward_batch))
    current_indptr, current_indices, current_splits = parts[3], parts[4], parts[5]
    # HEAD-SHAPED, here, because this partition goes to `forward_decode` DIRECTLY and so skips
    # the reshape `RadixAttention.forward` does on its way in. The fused path is reached through
    # that method and therefore never noticed; the split is not, and a flat key reaches the KV
    # write as `[rows, dim]` where `[rows, kv heads, dim]` was expected. At one row that
    # broadcasts and is silently right; at four it raises. The numbers are the layer's own.
    k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
    v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
    o_cur, lse_cur = _run_partition(
        backend,
        layer,
        forward_batch,
        q=state.q,
        k=k,
        v=v,
        indptr=current_indptr,
        indices=current_indices,
        splits=current_splits,
        save_kv_cache=True,
    )
    merged, _ = merge_state(state.o, state.lse, o_cur, lse_cur)
    return merged.view(-1, layer.tp_q_head_num * layer.v_head_dim)
