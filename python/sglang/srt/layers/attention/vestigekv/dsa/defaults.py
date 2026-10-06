"""The dsa arm's constants: the ONE namespace vestigekv_dsa_backend reads.

Why the arm has its own. The two arms share names and not signals, and a
constant carried across them by default argument has been wrong twice now --
LOWPASS_KAPPA was fitted on the mla arm's sidecar and reached the dsa arm's
index-k projection by nothing but a default, and the RHO/CLOSE_BLOCK pair had to
grow DSA_-prefixed twins for the same reason. Prefixed twins in one module do not
isolate anything: both arms still read both, and which one is authoritative is a
property of the call site rather than of the module. So the dsa arm reads this
module and the mla arm reads `vestigekv.defaults`, and the two policies cannot
reach each other.

What is NOT duplicated: model geometry and the sizes buffers are allocated from.
Those are properties of GLM-5.3-Flash and of the allocator, not of a tier-1
policy, so they are imported through from the shared module -- a second copy of
`KV_LORA_RANK` is a bug waiting for one of them to be edited.

The ceiling/served distinction is explicit in the names here. `RHO` and
`CLOSE_BLOCK` are what the arm SERVES; `RHO_CEILING` and `CLOSE_BLOCK_CEILING`
are what the kept table and the scan arena are SIZED from, which an env override
may rise to but not past. Reading the wrong one of those two was a live source of
confusion while they were called `DSA_RHO` and `RHO` in one namespace.
"""

from __future__ import annotations

# Arm-neutral: model geometry, allocator sizes, calibration cadence, the row
# invariant's thresholds, and the probability content of RECALL_TARGET.
from sglang.srt.layers.attention.vestigekv.defaults import (  # noqa: F401
    ARCHIVE_POOL,
    ATTN_SCALE,
    CHECK_MAX_KEPT_FRACTION,
    CHECK_MIN_SEQ_BLOCKS,
    INDEX_DTYPE,
    INDEX_RANK,
    KV_LORA_RANK,
    LATENT_DIM,
    LOWPASS_KAPPA_MAX,
    N_CAL_MAX,
    N_CAL_START,
    PREFILL_BUILD_MIN,
    PREFILL_CAL_STRIDE,
    RECENT_WINDOW,
    SCAN_CAPTURE_AFTER,
    SCAN_CAPTURE_MAX_FAILS,
    SIDECAR_DIM,
    SINKS,
    min_hard,
)
from sglang.srt.layers.attention.vestigekv.defaults import CLOSE_BLOCK as CLOSE_BLOCK_CEILING
from sglang.srt.layers.attention.vestigekv.defaults import RHO as RHO_CEILING

RHO = 1 / 128
"""Served tier-1 budget coefficient, four times below the mla arm's.

Paired with CLOSE_BLOCK this reads 0.607-0.614x DSA's bytes against 1.348-1.351x
at the ceiling. Superseded in practice by a CONSTANT budget (see KEPT_GROUPS):
rho*closed grows with context, and at 128k it reaches 1035 rows while a constant
32 groups stays at 128 -- which is what turns the byte ratio from 0.849 to 0.706
at that shape. rho remains the fallback when no constant budget is set."""

CLOSE_BLOCK = 1024
"""Decoded tokens per compression event, and sigma's window.

The SMALLEST block that leaves sigma intact: the cutoff period is
CLOSE_BLOCK/LOWPASS_KAPPA, and at 256 the basis reaches 48% of the window and
niah_multikey_3 collapsed to 0.000. It is also the only knob on the un-closed
tail, which averages CLOSE_BLOCK/2 rows attended unconditionally."""

LOWPASS_KAPPA = 16
"""rFFT bin cutoff of the low-pass whose residual is sigma, on THIS arm's signal.

`_index_sigma` projects DSA's index-k in GROUP space, window CLOSE_BLOCK/pool --
not the latent row's salience branch the mla arm transforms. Measured on the
served signal at the served size (88 calibration dumps, 704 real queries, all 11
DSA layers, mexp/glm53/kappa_m_grid.py), captured attention mass at a constant
32-group budget falls monotonically with kappa:

    kappa        1        2        3        4        8       16
    mass    0.1578   0.1551   0.1539   0.1523   0.1480   0.1410
    prec    0.6825   0.6626   0.6612   0.6555   0.6378   0.6264

kappa=1 keeps bin 0 alone, so its residual IS the row minus its block mean: on
this signal the trigonometric basis buys nothing a mean and a norm could not.
Held at 16 until the in-situ ladder (c521kap-*) reports, because offline tier-1
evidence has never decided quality on this line. Capped at LOWPASS_KAPPA_MAX by
the projection kernel's pad."""

KEPT_GROUPS = None
"""Constant hot set, in index-k GROUPS, or None for the proportional rho form.

A constant budget is the arm's long-context argument: the row term stops growing
with context, so the byte ratio stops being dragged toward 1 by it. The sign of
the trade flips with shape -- at 36k constant and proportional read 0.508 vs
0.511, at 128k 0.706 vs 0.849 -- because a smaller kept set also lowers the
tier-2 threshold, which is a quantile of the kept groups' logits, and fetch rises
to meet it. Set per-run through SGLANG_VESTIGEKV_KEPT_GROUPS."""
