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
    # The base's page size, forwarded to the fork so its K/V page and token
    # strides match the pool. Hard-coding 1 here was invisible for as long as
    # every deployment ran at page 1; DSA resolves the pool to page 64.
    page_size: int = 1
    rows: dict = {}  # layer id -> VestigeKVRows, refreshed per step by the backend
    current_layer: int = -1
    blend: dict = {}  # layer id -> (logm [B,H], mu [B,Lv]); omitted-mass arm only

    def __call__(
        self,
        q,
        k_buffer,
        v_buffer,
        o,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        k_descale=None,
        v_descale=None,
        **kw,
    ):
        vk = self.rows.get(self.current_layer)
        # The blend is independent of which stage 1 ran: an eager step keeps the
        # CSR and falls through to `inner`, and its output needs the same
        # denominator correction as a routed one.
        bl = self.blend.get(self.current_layer)
        if vk is None:
            out = self.inner(
                q,
                k_buffer,
                v_buffer,
                o,
                kv_indptr,
                kv_indices,
                attn_logits,
                attn_lse,
                num_kv_splits,
                max_kv_splits,
                sm_scale,
                k_descale,
                v_descale,
                **kw,
            )
            if bl is not None:
                _blend_omitted_mass(o, attn_lse, num_kv_splits, *bl)
            return out
        # k_descale folds into the scale exactly as upstream does before the
        # grouped launch; the tier path serves bf16 latents, where both are None.
        decode_grouped_att_m_fwd(
            q,
            k_buffer,
            v_buffer,
            attn_logits,
            attn_lse,
            kv_indptr,
            kv_indices,
            vk,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            kw.get("logit_cap", 0.0),
            has_mla=kw.get("has_mla", True),
            page_size=self.page_size,
        )
        _decode_softmax_reducev_fwd(
            attn_logits,
            attn_lse,
            q,
            o,
            v_descale,
            v_buffer,
            kv_indptr,
            num_kv_splits,
            max_kv_splits,
            kw.get("sinks"),
            use_pdl=kw.get("use_pdl", False),
        )
        if bl is not None:
            _blend_omitted_mass(o, attn_lse, num_kv_splits, *bl)
        return o


class DsaTierDecodeRouter(TierDecodeRouter):
    """The DSA-model router: stage 1 and 2 are DSA's own split-K decode over
    the tiers (vestigekv/dsa_decode_fork.py) instead of the MLA-decode fork.
    Layers without rows still fall through to the base's function."""

    def __call__(
        self,
        q,
        k_buffer,
        v_buffer,
        o,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        k_descale=None,
        v_descale=None,
        **kw,
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
    _BLEND_SEEN[0] += 1
    if _BLEND_SEEN[0] % 500 == 1:
        # Sampled periodically, not once: the FIRST call happens before any
        # block has closed, so there is no archive, logM is -inf and sigma is
        # exactly 1. A single reading there says nothing and is what made the
        # first attempt at this diagnostic useless. Offline the attended set
        # holds ~57% of the mass, so a correct implementation settles near
        # 0.5-0.6 once an archive exists; a sigma near 0 means logM is being
        # compared against an lse on a different scale and the output is being
        # swamped by the centroid -- an implementation fault, not a verdict on
        # the idea.
        f = sigma.flatten().float()
        logging.getLogger(__name__).info(
            "VKBLEND sigma @call %d: min %.4f p50 %.4f max %.4f over %d (q, head)",
            _BLEND_SEEN[0],
            float(f.min()),
            float(f.median()),
            float(f.max()),
            f.numel(),
        )
    view = o.view(-1, H, o.shape[-1])[:n]
    view.copy_(view.float() * sigma + mu.float() * (1.0 - sigma))


def combined_lse(attn_lse, num_kv_splits, max_kv_splits):
    """This rank's log-sum-exp over the rows it holds, for a cross-rank merge.

    Upstream's stage 1 stores one LSE per split -- `e_max + log(e_sum)` for the
    rows that split covered -- and stage 2 reduces them while writing only
    `acc / e_sum` to the output. So a rank's combined LSE exists nowhere after
    stage 2 and has to be taken from the same buffer stage 2 read: the reduction
    is log-sum-exp over the valid splits, which is what stage 2 does internally.

    Computing it here rather than making stage 2 emit it keeps upstream
    untouched, which is the whole reason the router redirects one call instead
    of forking the launcher.

    attn_lse is [bs, H, max_kv_splits] and only the first num_kv_splits[b] of
    each row are written; the rest are whatever the buffer held, so they are
    masked to -inf rather than trusted to be small.
    """
    idx = torch.arange(max_kv_splits, device=attn_lse.device)
    valid = idx.view(1, 1, -1) < num_kv_splits.view(-1, 1, 1)
    return torch.logsumexp(attn_lse.masked_fill(~valid, float("-inf")), dim=-1)


def merge_across_ranks(parts):
    """Combine per-rank (output, lse) into the attention of the union.

    parts: [(o, lse)] with o [bs, H, Lv] and lse [bs, H]. Softmax's own
    associativity: the union's output is the lse-weighted average of the
    ranks', which is what upstream's merge_state computes pairwise. A rank
    holding no rows carries lse -inf and drops out on its own.
    """
    from sglang.srt.layers.attention.merge_state import merge_state

    o, lse = parts[0]
    for o_next, lse_next in parts[1:]:
        o, lse = merge_state(o, lse, o_next, lse_next)
    return o, lse


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
