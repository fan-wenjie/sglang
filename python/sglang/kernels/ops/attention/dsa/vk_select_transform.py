from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# The same set kpool_topk_transform builds for: the module is specialised on the
# pooled group budget, which is a compile-time constant in the kernel.
SUPPORTED_GROUP_TOPK = (128, 160, 192, 224, 256, 512)


def _cuda_include_paths() -> List[str]:
    """Where cuda_runtime.h actually is on this box.

    The loader builds with CUDA_HOME, which resolves to /usr here while the
    headers live under /usr/local/cuda-12.8. Every other JIT module in the tree
    was cached before that mattered, so a fresh module is the first to notice.
    Resolved rather than hard-coded: a pinned version string would break on the
    next image.
    """
    seen = []
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    cands = ([os.path.join(home, "include")] if home else []) + sorted(
        glob.glob("/usr/local/cuda*/include"), reverse=True
    )
    for c in cands:
        if os.path.exists(os.path.join(c, "cuda_runtime.h")) and c not in seen:
            seen.append(c)
    return seen[:1]


@cache_once
def _jit_vk_select_transform_module(group_topk: int) -> Module:
    assert group_topk in SUPPORTED_GROUP_TOPK, (
        f"vk_select_transform supports pool-level topk {SUPPORTED_GROUP_TOPK}, "
        f"got {group_topk}"
    )
    return load_jit(
        f"vk_select_transform_{group_topk}",
        cuda_files=["dsa/vk_select_transform.cuh"],
        cuda_wrappers=[("vk_select_transform", "VkSelectTransformKernel::transform")],
        extra_cuda_cflags=[f"-DSGL_GROUP_TOPK={group_topk}"],
        extra_include_paths=_cuda_include_paths(),
    )


def vk_select_transform(
    *,
    score: torch.Tensor,
    lengths: torch.Tensor,
    thresholds: torch.Tensor,
    pool_size: int,
    topk: int,
    kept_groups: Optional[torch.Tensor] = None,
    kept_lens: Optional[torch.Tensor] = None,
    kept_mask: Optional[torch.Tensor] = None,
    ovf_counts: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """VestigeKV's selection in DSA's output layout.

    Emits kept u fired u tail as dst_token_indices, so DSA's attention kernel
    reads it unchanged. `thresholds` is per row and comes from the kept groups'
    own scores; rows whose fired count exceeds the budget fall back to DSA's
    exact radix top-k inside the kernel and are bit-identical to it, with
    `ovf_counts` recording that it happened.
    """
    assert score.ndim == 2 and score.dtype == torch.float32
    assert topk % pool_size == 0
    group_topk = topk // pool_size
    assert score.is_cuda, "vk_select_transform is CUDA-only"

    rows = score.shape[0]
    tail_cols = pool_size - 1 if seq_lens is not None else 0
    out = torch.empty(
        (rows, topk + tail_cols), dtype=torch.int32, device=score.device
    )
    mod = _jit_vk_select_transform_module(group_topk)
    mod.vk_select_transform(
        score,
        lengths.to(torch.int32),
        out,
        pool_size,
        page_table,
        topk_indices_offset,
        row_starts,
        seq_lens.to(torch.int32) if seq_lens is not None else None,
        page_table_row_index,
        thresholds,
        kept_groups,
        kept_lens,
        kept_mask,
        ovf_counts,
    )
    return out


@cache_once
def _jit_vk_select_fetch_module(group_topk: int) -> Module:
    return load_jit(
        f"vk_select_fetch_{group_topk}",
        cuda_files=["dsa/vk_select_transform.cuh"],
        cuda_wrappers=[("vk_select_fetch", "VkSelectFetchKernel::select")],
        extra_cuda_cflags=[f"-DSGL_GROUP_TOPK={group_topk}"],
        extra_include_paths=_cuda_include_paths(),
    )


def vk_select_fetch(
    *,
    logits: torch.Tensor,
    pool_lens: torch.Tensor,
    thresholds: torch.Tensor,
    kept_mask: torch.Tensor,
    slots: torch.Tensor,
    r2t: torch.Tensor,
    fetch_buf: torch.Tensor,
    fetch_len: torch.Tensor,
    fetch_ovf: torch.Tensor,
    ovf_counts: torch.Tensor,
    group_budget: int,
    pool: int,
    group_topk: int = 512,
) -> None:
    """One layer-step of the branch rule, in VestigeKV's own row layout.

    Drop-in for branch_fire's three later stages: same destination addresses,
    same `fetch_len = n * pool` contract, `fetch_ovf` always 0 because the
    overflow cut is exact and in budget. The kept rows are not touched -- they
    stay in kept_buf, where tier 1 wrote them.
    """
    mod = _jit_vk_select_fetch_module(group_topk)
    mod.vk_select_fetch(
        logits, pool_lens.to(torch.int32), thresholds, kept_mask,
        slots.to(torch.int32), r2t, fetch_buf, fetch_len, fetch_ovf,
        ovf_counts, group_budget, pool,
    )


@cache_once
def _jit_vk_collect_emit_module() -> Module:
    return load_jit(
        "vk_collect_emit",
        cuda_files=["dsa/vk_select_transform.cuh"],
        cuda_wrappers=[("vk_collect_emit", "VkCollectEmitKernel::run")],
        extra_cuda_cflags=["-DSGL_GROUP_TOPK=512"],
        extra_include_paths=_cuda_include_paths(),
    )


def vk_collect_emit(
    *,
    logits: torch.Tensor,
    pool_lens: torch.Tensor,
    thresholds: torch.Tensor,
    kept_mask: torch.Tensor,
    cand_score: torch.Tensor,
    cand_idx: torch.Tensor,
    ncand: torch.Tensor,
    slots: torch.Tensor,
    r2t: torch.Tensor,
    fetch_buf: torch.Tensor,
    fetch_len: torch.Tensor,
    fetch_ovf: torch.Tensor,
    ovf_counts: torch.Tensor,
    group_budget: int,
    pool: int,
) -> None:
    """The selection as a WIDE compare pass plus a narrow cut-and-emit.

    DSA's rank threshold forces one CTA over every group, because the rank is not
    known until the whole histogram exists. A reference-set threshold is known
    first, so the compare has no cross-group dependency and takes the device.
    """
    _jit_vk_collect_emit_module().vk_collect_emit(
        logits, pool_lens.to(torch.int32), thresholds, kept_mask,
        cand_score, cand_idx, ncand, slots.to(torch.int32), r2t,
        fetch_buf, fetch_len, fetch_ovf, ovf_counts, group_budget, pool,
    )


@cache_once
def _jit_vk_dsa_select_module(group_topk: int) -> Module:
    return load_jit(
        f"vk_dsa_select_{group_topk}",
        cuda_files=["dsa/vk_select_transform.cuh"],
        cuda_wrappers=[("vk_dsa_select", "VkDsaSelectKernel::run")],
        extra_cuda_cflags=[f"-DSGL_GROUP_TOPK={group_topk}"],
        extra_include_paths=_cuda_include_paths(),
    )


def vk_dsa_select(
    *,
    score: torch.Tensor,
    lengths: torch.Tensor,
    thresholds: torch.Tensor,
    state: dict,
    pool_size: int,
    topk: int,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DSA's selection with a reference-set threshold, in DSA's own layout.

    Returns dst_token_indices exactly as `topk_from_pooled_history_logits` does,
    and publishes the selected group ids into `state["sel_groups"]` for the next
    step to threshold against.
    """
    group_topk = topk // pool_size
    rows = score.shape[0]
    tail_cols = pool_size - 1 if seq_lens is not None else 0
    if out is None:
        out = torch.empty((rows, topk + tail_cols), dtype=torch.int32,
                          device=score.device)
    _jit_vk_dsa_select_module(group_topk).vk_dsa_select(
        score, lengths.to(torch.int32), thresholds,
        state["cand_score"], state["cand_idx"], state["ncand"],
        out, state["sel_groups"], state["sel_n"], state["ovf"],
        group_topk, pool_size,
        page_table, topk_indices_offset, row_starts,
        seq_lens.to(torch.int32) if seq_lens is not None else None,
        page_table_row_index,
    )
    return out
