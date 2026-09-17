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

import logging

import msgspec
import torch

_BLEND_SEEN = [0]

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
    blend: dict = {}  # layer id -> (logm [B,H], mu [B,Lv]); omitted-mass arm only

    def __call__(
        self, q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse,
        num_kv_splits, max_kv_splits, sm_scale, k_descale=None, v_descale=None, **kw,
    ):
        vk = self.rows.get(self.current_layer)
        # The blend is independent of which stage 1 ran: an eager step keeps the
        # CSR and falls through to `inner`, and its output needs the same
        # denominator correction as a routed one.
        bl = self.blend.get(self.current_layer)
        if vk is None:
            out = self.inner(
                q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse,
                num_kv_splits, max_kv_splits, sm_scale, k_descale, v_descale, **kw,
            )
            if bl is not None:
                _blend_omitted_mass(o, attn_lse, num_kv_splits, *bl)
            return out
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
        if bl is not None:
            _blend_omitted_mass(o, attn_lse, num_kv_splits, *bl)
        return o


def _blend_omitted_mass(o, attn_lse, num_kv_splits, logm_buf, mu_buf, slots):
    """Put back the softmax denominator the scan never attended.

    The output is a convex combination over the attended rows; the rows the
    scan skipped carry mass too, and dropping them inflates every retained
    weight by Z/Zv (measured ~2.9x on the Kimi dumps, where the attended set
    captures 57% of the dense mass). With Zv = exp(lse) from the reduce and an
    estimate M of the omitted mass, the corrected output is

        out * sigma + mu * (1 - sigma),   sigma = Zv/(Zv + M) = sigmoid(lse - logM)

    carrying the omitted mass at that set's MASS-WEIGHTED centroid, which is
    what makes the step exact: the update is one turn of the online-softmax
    recurrence O_n = lerp(O_{n-1}, v_n, sigmoid(s_n - lse_{n-1})) against a
    synthetic row, and a synthetic row is only right if it carries the
    weighted centroid (offline: 0.195 against 0.367 for the plain mean). A fenced lane attends
    everything and is handed logM = -inf, so sigma is 1 and nothing moves.

    attn_lse is allocated with torch.empty and only the first num_kv_splits
    entries of each row are written, so the unwritten tail is masked out rather
    than folded into the merge.
    """
    if _BLEND_SEEN[0] == 0:
        # An arm that silently does not run reads exactly like an arm that
        # ran and changed nothing -- which is how two different compensators
        # once returned bit-identical answers over 650 questions. This line is
        # the difference between a null result and a void one.
        _BLEND_SEEN[0] = 1
        logging.getLogger(__name__).info(
            "VKBLEND active: omitted-mass blend is in the decode path"
        )
    # The batch is the INSTALLED slot slice, never attn_lse.shape[0]: the
    # graph's logits and lse are preallocated at the largest captured batch and
    # a smaller capture leaves the tail unwritten, so reading the buffer's
    # first dimension as the batch mixes a bs-4 buffer with a bs-2 step.
    n, H = slots.shape[0], attn_lse.shape[1]
    # Gathered here, not by the caller: under graph capture the gather has to
    # be an op reading the step's fixed slot buffer, not a host-side index.
    logm = logm_buf.index_select(0, slots)  # [n, H]
    mu = mu_buf.index_select(0, slots)  # [n, H, Lv]
    idx = torch.arange(attn_lse.shape[-1], device=attn_lse.device)
    live = idx[None, None, :] < num_kv_splits[:n, None, None]
    lse = torch.logsumexp(
        attn_lse[:n].float().masked_fill(~live, float("-inf")), dim=-1
    )  # [n, H]
    sigma = torch.sigmoid(lse - logm)[..., None]  # [n, H, 1]
    if _BLEND_SEEN[0] == 1:
        # One reading of sigma, which separates the two ways this arm can be
        # wrong. Offline says the attended set holds ~57% of the mass, so a
        # correct implementation sits near 0.5-0.6. A sigma near 0 means logM
        # is being compared against an lse on a different scale and the output
        # is being swamped by the centroid -- an implementation fault, not a
        # verdict on the idea.
        _BLEND_SEEN[0] = 2
        f = sigma.flatten().float()
        logging.getLogger(__name__).info(
            "VKBLEND sigma: min %.4f p50 %.4f max %.4f over %d (q, head)",
            float(f.min()), float(f.median()), float(f.max()), f.numel(),
        )
    view = o.view(-1, H, o.shape[-1])[:n]
    view.copy_(view.float() * sigma + mu.float() * (1.0 - sigma))


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
        # The compaction raises fetch_ovf on any overflow, so the config is what
        # says whether to act on it; without this the tier path would attend the
        # full row set where the CSR path truncates.
        fence=backend.config.overflow_fallback,
        affine=backend._affine_capture,
    )
