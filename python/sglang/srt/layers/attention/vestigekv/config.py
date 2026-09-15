"""Deployment knobs of the vestigekv_mla backend, resolved once at startup.

The values come from the ``--vestigekv-*`` server flags (exec.kernel
namespace); this struct is the one object the backend reads them through, so
the cross-checks between them run in a single place and the effective
configuration is logged as one line.
"""

import msgspec


class VestigeKVConfig(msgspec.Struct, frozen=True, kw_only=True):
    # Fixed width of the per-(layer, request, step) recall fetch buffer, in
    # rows: graph capture bakes it, so it is a capacity, not a per-head cap.
    recall_capacity: int
    # A step whose fired recall set exceeds the capacity attends the request's
    # full row set (dense MLA) instead of a truncated fetch.
    overflow_fallback: bool
    # Requests shorter than this are served dense: nothing closed, archived
    # or recalled.
    activation_min_tokens: int
    # Rank of the tier-2 recall sketch.
    index_rank: int

    @classmethod
    def from_kernel_config(cls, kernel) -> "VestigeKVConfig":
        """Build from the ``exec.kernel`` config bag (``get_exec().kernel``)."""
        cfg = cls(
            recall_capacity=kernel.vestigekv_recall_capacity,
            overflow_fallback=not kernel.disable_vestigekv_recall_overflow_fallback,
            activation_min_tokens=kernel.vestigekv_activation_min_tokens,
            index_rank=kernel.vestigekv_index_rank,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.recall_capacity < 1:
            raise ValueError(
                f"--vestigekv-recall-capacity must be >= 1, got {self.recall_capacity}"
            )
        if self.activation_min_tokens < 0:
            raise ValueError(
                "--vestigekv-activation-min-tokens must be >= 0, got "
                f"{self.activation_min_tokens}"
            )
        if self.index_rank < 8 or self.index_rank % 8:
            raise ValueError(
                f"--vestigekv-index-rank must be a positive multiple of 8, got {self.index_rank}"
            )
    def describe(self) -> str:
        return (
            f"recall_capacity={self.recall_capacity} "
            f"overflow_fallback={self.overflow_fallback} "
            f"activation_min_tokens={self.activation_min_tokens} "
            f"index_rank={self.index_rank}"
        )
