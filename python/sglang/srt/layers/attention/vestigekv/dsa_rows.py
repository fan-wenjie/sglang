# SPDX-License-Identifier: Apache-2.0
"""What the DSA decode fork reads for one layer of one step, and the router
that hands it the base's stage-1 call.

`TritonAttnBackend.decode_attention_fwd` is an instance attribute, so the whole
redirection is one assignment: the base's `forward_decode` keeps the KV write,
the logits buffer selection, stage 2 and the output reshape, and only its
stage-1 call lands here. Nothing upstream is edited. The redirection is per
layer because the arrays are; a layer this backend does not own falls through to
the base's own function unchanged.

This replaces vestigekv/tier_decode.py and the struct that lived in
vestigekv/decode_fork.py, both of which carried the MLA line's router, the
CSR-reading fallback, the omitted-mass blend and the page-1 affine arm. None of
those can run on GLM-5.3: DSA resolves the pool to page 64, so the affine arm's
precondition (page_size == 1) is never met, and the branch rule replaced the
sketch tier that the CSR path served.
"""

import msgspec
import torch


class VestigeKVRows(msgspec.Struct):
    """Where a lane's attended rows are: the kept table and the fetch buffer of
    ONE layer, plus the page table a fenced lane uses. `fence` False is the
    deleted-fallback ablation (SGLANG_DEBUG_VESTIGEKV_NO_OVERFLOW_FALLBACK): the
    compaction raises fetch_ovf whenever a scan overflowed regardless of that
    key, so the kernel has to be told not to act on it."""

    slots: torch.Tensor
    kept_buf: torch.Tensor
    kept_len: torch.Tensor
    fetch_buf: torch.Tensor
    fetch_len: torch.Tensor
    fetch_ovf: torch.Tensor
    r2t: torch.Tensor
    seq: torch.Tensor
    loc: torch.Tensor
    fence: bool = True
    # The indexer's selection for this step, [bs, TOPK] int32 pool rows (-1
    # padded); a fenced lane attends these instead of its page table.
    topk: object = None
    topk_k: int = 0  # index_topk: the pooled budget the count formula clamps to
    kpool: int = 1  # index_kpool
    qbuf: object = None  # [R1, H, Lk]: when set, the kernel files q here


def rows_for_layer(backend, lid, slots, seq, loc) -> VestigeKVRows:
    """Views, not copies: every array here is a fixed-address buffer the graph
    already writes."""
    return VestigeKVRows(
        slots=slots,
        kept_buf=backend._kept_buf[lid],
        kept_len=backend._kept_len[lid],
        fetch_buf=backend._fetch_buf[lid],
        fetch_len=backend._fetch_len[lid],
        fetch_ovf=backend._fetch_ovf[lid],
        r2t=backend.req_to_token_pool.req_to_token,
        seq=seq,
        loc=loc,
        # The compaction raises fetch_ovf on any overflow, so the config is what
        # says whether to act on it.
        fence=backend.config.overflow_fallback,
    )


class DsaTierDecodeRouter(msgspec.Struct, dict=True):
    """Installed on the base as `decode_attention_fwd`. Stage 1 and 2 are DSA's
    own split-K decode over the kept table and the fetch buffer."""

    inner: object  # the base's original function, for layers this backend skips
    # The base's page size. Hard-coding 1 was invisible while every deployment
    # ran at page 1; DSA resolves the pool to page 64.
    page_size: int = 1
    rows: dict = {}  # layer id -> VestigeKVRows, refreshed per step
    current_layer: int = -1

    def __call__(
        self, q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits,
        attn_lse, num_kv_splits, max_kv_splits, sm_scale,
        k_descale=None, v_descale=None, **kw,
    ):
        vk = self.rows.get(self.current_layer)
        if vk is None:
            return self.inner(
                q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits,
                attn_lse, num_kv_splits, max_kv_splits, sm_scale, k_descale,
                v_descale, **kw,
            )
        from sglang.srt.layers.attention.vestigekv.dsa_decode_fork import vk_dsa_decode

        kb = k_buffer.view(k_buffer.shape[0], 1, -1) if k_buffer.dim() != 3 else k_buffer
        vk_dsa_decode(q, kb, o, vk, sm_scale, d_v=o.shape[-1])
        return o
