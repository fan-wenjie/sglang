from __future__ import annotations

import logging

from sglang.srt.environ import envs
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Protocol

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)

DSA_DENSE = "dense"
DSA_SPARSE = "sparse"
# VestigeKV on a DSA model (vestigekv_dsa): a step whose lanes did not
# overflow last time attends kept + recalled rows and needs no selection, so
# the indexer files its key and skips scoring and top-k ("lean"); a step
# after an overflow runs the full indexer so the fenced lane can attend the
# selection ("topk"). Chosen per step by the backend from a stale-by-one
# host readback of its overflow counter.
VK_LEAN = "vk_lean"
VK_TOPK = "vk_topk"
_vestigekv_variant_source = None


def set_vestigekv_variant_source(fn) -> None:
    """The vestigekv_dsa backend registers `fn(forward_batch) -> label`."""
    global _vestigekv_variant_source
    _vestigekv_variant_source = fn


class AttentionGraphVariants(Protocol):
    # Capture order is significant when variants share a graph memory pool.
    capture_labels: ClassVar[tuple[str, ...]]

    def select(self, forward_batch: ForwardBatch) -> str:
        """Select one of capture_labels for the batch."""
        ...


@dataclass(frozen=True)
class DsaGraphVariants:
    index_topk: int
    # Dense comes first: the sparse capture peak subsumes its shared-pool storage.
    capture_labels: ClassVar[tuple[str, ...]] = (DSA_DENSE, DSA_SPARSE)

    def select(self, forward_batch: ForwardBatch) -> str:
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0:
            # Plain decode maintains this host mirror without a D2H sync.
            max_kv_len = int(seq_lens_cpu.max().item())
        elif forward_batch.seq_lens is not None and forward_batch.seq_lens.numel() > 0:
            # Fallback: a single scalar reduction d2h (cheap, per-step).
            max_kv_len = int(forward_batch.seq_lens.max().item())
        else:
            # No length info: be safe and use the correct-for-all sparse graph.
            return DSA_SPARSE
        return DSA_DENSE if max_kv_len <= self.index_topk else DSA_SPARSE


@dataclass(frozen=True)
class VestigeKVDsaGraphVariants:
    # topk first: its capture peak (the indexer's scoring buffers) subsumes lean's.
    capture_labels: ClassVar[tuple[str, ...]] = (VK_TOPK, VK_LEAN)

    def select(self, forward_batch: ForwardBatch) -> str:
        fn = _vestigekv_variant_source
        # No source yet (before the backend is built) means the full path.
        return VK_TOPK if fn is None else fn(forward_batch)


def create_attention_graph_variants(
    hf_config, decode_backend: Optional[str] = None
) -> Optional[AttentionGraphVariants]:
    from sglang.srt.configs.model_config import get_dsa_index_topk, is_deepseek_dsa
    from sglang.srt.utils import is_hip

    # The registry admits vestigekv_dsa only on a rope-less DSA model, so the
    # backend name is the whole condition (the outer GLM config does not
    # answer is_deepseek_dsa; its text config does).
    if decode_backend == "vestigekv_dsa" and envs.SGLANG_ENABLE_VESTIGEKV_LEAN_GRAPH.get():
        logger.info(
            "[vestigekv_dsa] dual-graph enabled: capturing topk (full indexer) + "
            "lean (key only) decode graphs; dispatch on last step's overflow."
        )
        return VestigeKVDsaGraphVariants()
    if is_hip() and is_deepseek_dsa(hf_config):
        index_topk = get_dsa_index_topk(hf_config)
        logger.info(
            "[dense-decode] DSA dual-graph enabled: capturing "
            "dense (k-only) + sparse (full indexer) decode graphs; "
            "dispatch on max_kv_len vs index_topk=%d.",
            index_topk,
        )
        return DsaGraphVariants(index_topk)
    return None
