"""Deployment knobs of the vestigekv_mla backend, resolved once at startup.

The values come from the ``--vestigekv-*`` server flags (exec.kernel
namespace); this struct is the one object the backend reads them through, so
the cross-checks between them run in a single place and the effective
configuration is logged as one line.
"""

import msgspec
from sglang.srt.environ import envs

RECALL_THRESHOLDS = ("max", "lse")


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
    # Recall margin in scaled-logit units: an archived row fires when its
    # certified score exceeds the kept max minus this. 0 is exact max-recall;
    # delta bounds the softmax weight of a dropped row at e^-delta of the
    # kept max (ln K covers K-way near ties, e.g. multi-key needles).
    recall_margin: float
    # What the margin is taken from: "max" = the best kept-row score (a dropped
    # row's weight <= e^-margin of the max row); "lse" = the log-sum-exp of the
    # kept scores (a dropped row's weight <= e^-margin of the whole kept mass:
    # self-adapting to how peaked the kept distribution is).
    recall_threshold: str
    # Calibrate the recall index on absorbed prompt queries during prefill
    # (at PREFILL_BUILD_MIN tokens, then at every doubling) so the first decode
    # steps are served by a calibrated index instead of the provisional one.
    prefill_calibration: bool
    # Refit a layer's index when its scan overflowed on more than this fraction
    # of the steps since that layer's last close (0 disables): the fit is
    # otherwise made once, early, and serves the whole request.
    rebuild_overflow_fraction: float
    # Size the decode kernel's KV split count from the attended rows rather than
    # the request's length (changes the accumulation grouping, not the row set).
    attended_splits: bool
    # Read a lane's rows from the tiers (kept table + fetch buffer, or the page
    # table when fenced) instead of from a CSR packed for the step.
    tier_decode: bool
    # Assert the page table is contiguous so a fenced lane computes its row ids.
    affine_page_table: bool

    @classmethod
    def from_kernel_config(cls, kernel) -> "VestigeKVConfig":
        """Build from the ``exec.kernel`` config bag (``get_exec().kernel``)."""
        cfg = cls(
            recall_capacity=kernel.vestigekv_recall_capacity,
            overflow_fallback=not envs.SGLANG_DEBUG_VESTIGEKV_NO_OVERFLOW_FALLBACK.get(),
            activation_min_tokens=kernel.vestigekv_activation_min_tokens,
            index_rank=kernel.vestigekv_index_rank,
            recall_margin=kernel.vestigekv_recall_margin,
            recall_threshold=kernel.vestigekv_recall_threshold,
            prefill_calibration=kernel.enable_vestigekv_prefill_calibration,
            rebuild_overflow_fraction=kernel.vestigekv_rebuild_overflow_fraction,
            attended_splits=kernel.enable_vestigekv_attended_splits,
            tier_decode=kernel.enable_vestigekv_tier_decode,
            affine_page_table=kernel.enable_vestigekv_affine_page_table,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.affine_page_table and not self.tier_decode:
            raise ValueError(
                "--enable-vestigekv-affine-page-table needs "
                "--enable-vestigekv-tier-decode: only the tier path reads the page table"
            )
        if not 0.0 <= self.rebuild_overflow_fraction <= 1.0:
            raise ValueError(
                "--vestigekv-rebuild-overflow-fraction must be in [0, 1], got "
                f"{self.rebuild_overflow_fraction}"
            )
        if self.recall_capacity < 1:
            raise ValueError(
                f"--vestigekv-recall-capacity must be >= 1, got {self.recall_capacity}"
            )
        if self.activation_min_tokens < 0:
            raise ValueError(
                "--vestigekv-activation-min-tokens must be >= 0, got "
                f"{self.activation_min_tokens}"
            )
        if self.recall_margin < 0:
            raise ValueError(
                f"--vestigekv-recall-margin must be >= 0, got {self.recall_margin}"
            )
        if self.recall_threshold not in RECALL_THRESHOLDS:
            raise ValueError(
                f"--vestigekv-recall-threshold must be one of {RECALL_THRESHOLDS}, "
                f"got {self.recall_threshold!r}"
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
            f"index_rank={self.index_rank} recall_margin={self.recall_margin} "
            f"recall_threshold={self.recall_threshold} "
            f"prefill_calibration={self.prefill_calibration} "
            f"rebuild_overflow_fraction={self.rebuild_overflow_fraction} "
            f"attended_splits={self.attended_splits} "
            f"tier_decode={self.tier_decode} "
            f"affine_page_table={self.affine_page_table}"
        )
