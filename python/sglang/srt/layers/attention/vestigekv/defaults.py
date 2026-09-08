"""Every tunable constant VestigeKV uses, in one place, with the reason.

A number that only exists at its use site cannot be audited, cannot be diffed
between two runs, and cannot be found by someone asking "what did this run
actually do?". Anything here that changes what is compared between two runs
belongs in a run record, not only in this file.

Runtime knobs (the fetch cap, the calibration-query count, the debug hooks) are
env vars registered in `sglang.srt.environ`; this module holds the values that
are NOT knobs -- the ones a deployment does not get to vary.
"""

import contextlib
import math

import torch

# ---- model geometry (Kimi Linear MLA; the only architecture in scope) ----

KV_LORA_RANK = 512
"""Content half of an MLA latent row; also the value vector."""

SIDECAR_DIM = 64
"""Un-roped half of the latent row. Under NoPE this branch is what carries the
tier-1 anomaly signal and the tier-2 exact summand."""

LATENT_DIM = KV_LORA_RANK + SIDECAR_DIM  # 576
"""Full latent row width, as stored in the KV pool."""

ATTN_SCALE = 192**-0.5
"""1/sqrt(qk head dim) = 1/sqrt(128 nope + 64 rope). Applied to every score so
the tier-2 comparison lives on the same scale as attention's own logits."""

# ---- tier 1: sidecar-residual eviction (frozen at prefill) ----

RHO = 1 / 32
"""Fraction of the prefix kept by anomaly rank. Sets the compression ratio."""

SINKS = 4
"""Leading rows kept unconditionally: attention sinks are never anomalous by
the sigma criterion yet removing them destabilizes the softmax."""

RECENT_WINDOW = 256
"""Trailing rows kept unconditionally. Decoded tokens join here, which is why
the archive is fixed at prefill and the tier-2 scan is capturable."""

LOWPASS_KAPPA = 16
"""rFFT cutoff for the low-pass whose residual defines sigma.

Derivability was measured and rejected (m8, 3 docs x 7 layers, real queries):
the query-grounded criterion -- attention mass captured by the sigma_kappa
top-(rho*T) set -- peaks exactly at 16, but the curve is flat to +-0.3% over
kappa in [4, 64], so data-driven selection has nothing to win; and the one
cheap query-free proxy (kurtosis of sigma) rises monotonically past 16 and
would pick 32, disagreeing with the ground truth. A fixed 16 it stays."""

# ---- tier 2: certified recall ----

INDEX_RANK = 64
"""Sketch rank r. Trades scan bytes (2*r floats per archived row) against how
often the residual certificate has to inflate a score to stay sound."""

RECALL_TARGET = 0.90
"""tau: the single quality parameter of the recall tier. Everything below that
used to be a separate knob is DERIVED from it -- see the functions at the end
of this file. The total miss budget 1 - tau is split evenly between the two
stages that can lose a target: the entropy gate (skipping a scan the query
needed) and the sketch competition (the certified score losing anyway)."""

Z_MAX = 8.0
"""Safety clamp on the certificate multiplier, and the value the provisional
(uncalibrated) index serves with. Engineering bound, not a tuning knob: firing
more than this implies the sketch explains almost nothing of the query, at
which point more inflation buys noise, not recall."""

GATE_SELF_DISABLE_FRACTION = 0.60
"""If a fitted gate threshold would leave the gate open on more than this
fraction of calibration queries it is discarding nothing, so it is disabled
outright rather than left as a threshold that only looks like a filter.
A utility constant (scan-cost vs benefit), not a probability."""

N_CAL_START = 8
"""Decode steps collected before the first calibrated build. The schedule then
doubles the window until MIN_HARD hard samples exist (or N_CAL_MAX is hit), so
the effective sample size is set by the conformal requirement below, not by a
hand-picked count."""

N_CAL_MAX = 64
"""Upper bound on the adaptive collection window. A prefix whose queries are
still not producing MIN_HARD hard samples by here is genuinely easy (the kept
tier dominates); the index then serves with zp clamped to Z_MAX, which errs
toward over-fetching, never under-recall."""

ENTROPY_EPS = 1e-12
"""Clamp inside log for the entropy gate."""

BUILD_KEY_CHUNK = 16384
"""Key-axis chunk for the calibration argmax. Chunked because the full
[n*H, T] score matrix OOMs at S=512k."""

FETCH_WIDTH_UNCAPPED = 4096
"""Fetch buffer width when no cap is set. Not a cap: it is a fixed-address
buffer size above the worst fire ever observed (3983); overflow is truncated
and must be counted, never silently dropped."""

# ---- tier-2 scan capture (CUDA graph) ----

# ---- decode-time block closing ----

CLOSE_BLOCK = 4096
"""Decoded tokens per compression event. Every CLOSE_BLOCK decode steps a
request's newest block is closed: sigma is computed for its rows, the global
top-(rho * closed) selection rebalances (matching the reference policy), the
evicted rows join the tier-2 archive via the live caches, and the recall index
refreshes by index selection. Without this, rows generated after prefill are
never evicted (the attended set grows 1:1 with generation) and rows evicted by
a close would be unrecallable -- both wrong at long generation."""

SCAN_CAPTURE_AFTER = 8
"""Decode steps one scan shape must hold before it is captured, so a short
generation does not pay for a graph it replays a handful of times."""

SCAN_KMAX_HEADROOM = CLOSE_BLOCK
"""Extra kept-table columns baked into a capture. kept_len grows by one per
decode step until the next block close resets it, so a capture only ever needs
to survive one close interval: headroom = CLOSE_BLOCK ends the forced
recapture-every-512-steps regime (measured: 8 recaptures per 4096-step request
at 256k, ~45 ms each). The cost is a wider baked gather in the in-graph pack,
which the lens mask renders harmless."""

# ---- fused scan kernel ----

SCAN_BLOCK_A = 64
"""Archive rows per Triton block. 128 and 256 measured the same or ran out of
registers on SM120."""

SCAN_NUM_WARPS = 4


# ---- row invariant (SGLANG_DEBUG_VESTIGEKV_ROWS) ----

CHECK_MIN_SEQ_BLOCKS = 4
"""The invariant only asserts compression once the sequence is this many
CLOSE_BLOCKs long. Below that, kept legitimately IS most of the sequence: the
unclosed tail alone can be CLOSE_BLOCK-1 rows, so at 4 blocks the bound
rho*seq + CLOSE_BLOCK + SINKS < 0.5*seq holds with margin and asserting
earlier would fire on correct behavior."""

CHECK_MAX_KEPT_FRACTION = 0.5
"""Above this share of the sequence the compressed arm is not compressing, which
is the silent wrong-row-set failure this check exists to catch."""

SCAN_CAPTURE_MAX_FAILS = 3
"""Consecutive capture refusals before the eager path becomes permanent. One
refusal can be transient (a busy stream, a momentary OOM) and latching on it
costs every later step in the process; three in a row is the deployment."""

BUILD_ROW_CHUNK = 16384
"""Archive rows converted to fp32 at a time when building the index. Bounds the
build's peak scratch to one chunk instead of a full [T, 576] fp32 copy."""


# ---- derivations from RECALL_TARGET (the probability content) ----


def gate_alpha(tau: float = RECALL_TARGET) -> float:
    """Share of the miss budget the entropy gate may spend: (1 - tau) / 2.

    The gate is a binary filter on hard queries; closing on one loses its
    target outright, so its false-close rate is capped at half the budget."""
    return (1.0 - tau) / 2.0


def scan_target(tau: float = RECALL_TARGET) -> float:
    """Per-scan recall level once the gate has spent its share.

    Overall recall >= (1 - gate_alpha) * scan_target = tau, so the scan must be
    calibrated at tau / (1 - gate_alpha)."""
    return tau / (1.0 - gate_alpha(tau))


def min_hard(tau: float = RECALL_TARGET) -> int:
    """Fewest hard samples for which the conformal quantile at scan_target
    exists: ceil((n+1)*t) <= n requires n >= t / (1 - t)."""

    t = scan_target(tau)
    # 1e-9 guard: for rational tau the ratio is often an exact integer that
    # floating point lands a hair ABOVE (0.9 -> t = 18/19, t/(1-t) = 18 but
    # floats give 18.000000000000004), and a raw ceil would then demand one
    # sample more than the guarantee needs.
    return math.ceil(t / (1.0 - t) - 1e-9)


def conformal_k(n: int, tau: float = RECALL_TARGET) -> int:
    """Order-statistic index for the scan-level conformal quantile: with z_(1)
    <= ... <= z_(n) the required inflations of n exchangeable hard samples,
    zp = z_(k) at k = ceil((n+1)*scan_target) gives the distribution-free
    marginal guarantee P(target recovered) >= scan_target."""
    import math

    return math.ceil((n + 1) * scan_target(tau) - 1e-9)  # same float guard


@contextlib.contextmanager
def ieee_fp32_matmul():
    """Pin fp32 matmuls to full ieee precision on the certificate path.

    The fire decision compares scores near a conformal threshold; tf32's
    10-bit mantissa moved 6 of 58900 rows across it in measurement. The
    scan kernels already force input_precision="ieee"; this guard covers
    the cuBLAS bmms (qsk projection, qres residual) that would otherwise
    follow the global --enable-tf32-matmul switch. Wrapping capture/build
    is enough: kernel selection happens there, replay keeps it.
    """
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


def ieee_fp32(fn):
    """Decorator form of ieee_fp32_matmul for certificate-path methods."""

    def wrapped(*a, **k):
        with ieee_fp32_matmul():
            return fn(*a, **k)

    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped
