# SPDX-License-Identifier: Apache-2.0
"""Route the base's decode attention at the tiers instead of a CSR.

`TritonAttnBackend.decode_attention_fwd` is an instance attribute, so the whole
redirection is one assignment: the base's `forward_decode` keeps doing the KV
write, the logits buffer selection, stage 2 and the output reshape, and only the
call it makes to compute stage 1 lands here instead. Nothing upstream is edited.

What this removes is the CSR: `decode_fork` reads a lane's rows from the kept
table and the fetch buffer, or from the page table when the lane is fenced, so
the pack that copied them into one array is not needed, and neither is any of
the per-step work that arming the fence costs.

The redirection is per layer, because the tier arrays are: `current_layer`
carries which one the base is about to compute. A layer this backend does not
own falls through to the base's own function unchanged.
"""

import msgspec
import torch

from sglang.kernels.ops.attention.decode_attention import _decode_softmax_reducev_fwd
from sglang.srt.layers.attention.vestigekv.decode_fork import (
    VestigeKVRows,
    decode_grouped_att_m_fwd,
)


class TierDecodeRouter(msgspec.Struct, dict=True):
    """Installed on the base as `decode_attention_fwd`; holds what the fork
    needs that the base's call site does not pass."""

    inner: object  # the base's original function, for layers this backend skips
    rows: dict = {}  # layer id -> VestigeKVRows, refreshed per step by the backend
    current_layer: int = -1

    def __call__(
        self, q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse,
        num_kv_splits, max_kv_splits, sm_scale, k_descale=None, v_descale=None, **kw,
    ):
        vk = self.rows.get(self.current_layer)
        if vk is None:
            return self.inner(
                q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse,
                num_kv_splits, max_kv_splits, sm_scale, k_descale, v_descale, **kw,
            )
        # k_descale folds into the scale exactly as upstream does before the
        # grouped launch; the tier path serves bf16 latents, where both are None.
        decode_grouped_att_m_fwd(
            q, k_buffer, v_buffer, attn_logits, attn_lse, kv_indptr, kv_indices, vk,
            num_kv_splits, max_kv_splits, sm_scale, kw.get("logit_cap", 0.0),
            has_mla=kw.get("has_mla", True),
        )
        _decode_softmax_reducev_fwd(
            attn_logits, attn_lse, q, o, v_descale, v_buffer, kv_indptr,
            num_kv_splits, max_kv_splits, kw.get("sinks"),
            use_pdl=kw.get("use_pdl", False),
        )
        return o


def rows_for_layer(backend, lid, slots, seq, loc) -> VestigeKVRows:
    """The arrays the fork reads for one layer of this step. Views, not copies:
    every one of them is a fixed-address buffer the graph already writes."""
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
    )
