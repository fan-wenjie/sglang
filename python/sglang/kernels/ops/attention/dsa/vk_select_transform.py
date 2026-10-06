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
