"""The pool's half of the cooked linear layer: convolve, normalise, form the coefficient, send.

The first build of the early read sent RAW materials and let the host finish them, and a ladder
priced that at 10.69 percentage points: the host's preparation was 1.207 ms against a pool window
of 0.461, so the schedule's max moved to the host and every layer paid the overflow on the
bottleneck card. The rule that came out of it is architectural, not a tuning note: **the host
receives finished materials and contracts, nothing else.** Everything preparatory happens here,
on the card that is idle 83% of the time it spends in a span.

The ring stays the host's -- the pool holds nothing between calls. What crosses is a WINDOW: the
host attaches each linear layer's last three pre-convolution columns to the span call, this side
convolves against them, and the new column rides back on the mix frame for the host to append.
State in flight, owned at rest by exactly one side.

Everything here is a function of its arguments so the old arrangement and this one can be run
against each other in one process and compared to the bit; `test_afd_cooked_is_the_same_arithmetic`
does exactly that.
"""

from __future__ import annotations

import torch


def convolve_partial(partial: torch.Tensor, x: torch.Tensor, last_tap: torch.Tensor):
    """Finish a causal convolution whose history is already summed, one token a row.

    `partial` is `(rows, channels)` -- the history columns weighted by the first K-1 taps and
    summed, the caller's to provide -- and `x` is this step's own projection, weighted by the
    LAST tap here. Against `convolve_with_ring` this reassociates one sum (history-then-newest
    instead of all four together), the same regime as the fused cook; the history's half is
    computed once by the ring's owner and serves the early and the mix cook alike, because the
    two differ only in the newest tap.
    """
    return torch.nn.functional.silu(partial + x * last_tap)


def cook_early(
    early_qk: torch.Tensor,
    beta: torch.Tensor,
    partial_qk: torch.Tensor,
    weight_qk: torch.Tensor,
    *,
    key_heads: int,
    value_heads: int,
    head_k_dim: int,
):
    """The early half, finished: from the shifted `[q | k]` to the contraction's own inputs.

    Returns `(q_tilde, q)` -- the query coefficient the state is contracted with, and the
    normalised expanded query the mix's `s = beta (k . q)` takes. Both are the caller's to send;
    the far end must be able to go from these to a reply without touching a ring, a norm weight,
    or a convolution, because any of those on the far end is the max moving back to the host.

    `beta` here is the EARLY write strength: it forms the coefficient's `P(k)` and nothing else.
    The state's own advance takes the current one, from the mix.
    """
    from sglang.srt.afd.linear_history import (
        expand_to_value_heads,
        normalise,
        query_coefficient,
    )

    mixed = convolve_partial(partial_qk, early_qk, weight_qk[:, -1])
    rows = mixed.shape[0]
    width = key_heads * head_k_dim
    q = mixed[:, :width].reshape(rows, key_heads, head_k_dim)
    k = mixed[:, width:].reshape(rows, key_heads, head_k_dim)
    q, k = normalise(q, k, scale=head_k_dim**-0.5)
    q = expand_to_value_heads(q, value_heads)
    k = expand_to_value_heads(k, value_heads)
    q_tilde, _ = query_coefficient(q, k, beta)
    return q_tilde, q


def cook_mix(
    packed_kv: torch.Tensor,
    partial_kv: torch.Tensor,
    weight_kv: torch.Tensor,
    *,
    key_heads: int,
    value_heads: int,
    head_k_dim: int,
    head_v_dim: int,
):
    """The mix half, finished: this step's convolved key and value, ready for the state.

    `packed_kv` is the pre-convolution `[k | v]` and nothing else, because the operator has ONE
    query and it went early: the current projection does not compute a q at all -- the caller
    slices the fused weight's q rows away, which is an eighth of that weight read the serial
    arrangement pays and this one does not. The key comes back normalised and expanded because
    that is what both its consumers -- `s` and the state advance -- take.
    """
    from sglang.srt.afd.linear_history import expand_to_value_heads, normalise

    rows = packed_kv.shape[0]
    width = key_heads * head_k_dim
    kv = convolve_partial(partial_kv, packed_kv, weight_kv[:, -1])
    k = kv[:, :width].reshape(rows, key_heads, head_k_dim)
    v = kv[:, width:].reshape(rows, value_heads, head_v_dim)
    # `normalise` takes the pair; the q half here is a placeholder the caller never sees. The
    # scale multiplies q alone, so k's arithmetic is identical to the host's two-argument call.
    _, k = normalise(k, k, scale=head_k_dim**-0.5)
    k = expand_to_value_heads(k, value_heads)
    return k, v


def norm_ratio(
    own_weight: torch.Tensor, reused_weight: torch.Tensor, *, floor: float = 1e-3
) -> torch.Tensor:
    """The vector that turns the reused norm's output into this layer's own.

    Both norms read the SAME residual, so they share the same rms and differ only in the
    learned per-channel scale:

        own(x) = s_own . x/rms(x) = (s_own / s_reused) . reused(x)

    -- the whole "extra RMSNorm" collapses to one elementwise multiply by a ratio computed
    once at install. The caller passes each norm's EFFECTIVE scale vector, in the norm's own
    convention: a plain RMSNorm scales by `weight`, a GemmaRMSNorm by `1 + weight` -- and the
    Qwen3.5 layers this serves are GemmaRMSNorm, whose stored weights sit near -1 while the
    scales sit near zero. The first wiring passed the stored weights of that model and served
    a ratio of the wrong sign and the wrong size; the deployment spoke garbage from its first
    token while the identity test, which checked the formula against the same wrong
    convention, stayed green. Refused, by layer and by channel count, when the reused scale
    has channels near zero: the ratio there amplifies whatever noise sits in that channel,
    and a silent fallback would serve two different arithmetics under one name.
    """
    small = (reused_weight.abs() < floor).sum().item()
    if small:
        raise ValueError(
            f"the reused norm's scale has {small} channel(s) below {floor}; the ratio "
            f"through them amplifies noise. Run this layer's own norm instead."
        )
    return own_weight / reused_weight


def renorm_with_ratio(reused_normed: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    """Apply the install-time ratio: the own-norm reading at the cost of one multiply."""
    return reused_normed * ratio
