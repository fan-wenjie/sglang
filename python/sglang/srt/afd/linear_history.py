"""A linear-attention layer, split the way a softmax one already is: a read, then a mix.

The arrangement's rule is that a per-request read belongs with the request's data and a weight read
belongs with the weights. For a softmax layer that split is obvious and already built: the host
holds a KV cache, sweeps it with a query, and returns what it read; the pool holds the projections
and folds this step's key and value in.

A linear layer looked like it could not be split that way. `linear_state.py` said so, and the
Early-Q window was held to exist only at the sixteen layers in sixty-four that sweep a cache. That
was wrong, and `benchmark/afd/gdn_split.py` is the measurement that says so: the recurrence is
AFFINE in the old state, so

    h_q = S q                                    a read, with the query alone
    h_k = S k                                    a read, with the key alone
    o   = alpha h_q + beta (v - alpha h_k)(k.q)  the two readings, mixed with scalars

reproduces the fused kernel elementwise -- 2.2e-3 relative on the output at bfloat16, 9.7e-8 on the
state update. The fusion is the kernel's, not the recurrence's.

## The interface the two kinds of layer now share

    stage        softmax layer                     linear layer
    read         sweep the cache with q            read the state with q and with k
    returns      o_hist, lse                       h_q, h_k
    mix          fold in this step's k, v          alpha h_q + beta (v - alpha h_k)(k.q)
    holds        the KV cache                      the recurrent state and the convolution's

Both sides are then doing one thing each in both kinds of layer. The host holds a history, reads
it, and returns the reading. The pool holds the weights, and mixes.

## Why the state stays on the host

Because it is the request's, and the pool is stateless -- a caller that stalls stops calling and
blocks nobody, which is the property the pool is a separate process for. Holding a request's
recurrent state on the pool would reserve it between that request's calls, and the arrangement's
own docstring says it must not.

It also puts the update where sglang's own prefill path already lives. A pool that held the state
would have to run a chunked delta rule for prefill as well as a recurrent one for decode; a host
that holds it just runs the model.

## What the split costs, and what it buys

It costs a reading that a fused kernel did in one pass: two contractions of the state instead of
one. The state is 1.5 MiB a layer a request and the read is 7.0 us at any context, so the second
contraction is the same 7.0 us again -- against a round trip of 628 us, which is what the split
exists to overlap.

It buys the window at every layer instead of at one in four. On this model that is 64 against 16.
"""

from __future__ import annotations

import torch


def gates(a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor,
          dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The decay and the write strength, in the form the fused sigmoid gating defines them.

    Separated out because they are the only part of the recurrence that is not a state read, and
    they are computed WHERE THE WEIGHTS ARE: `A_log` and `dt_bias` are per-layer parameters, so
    the side that holds the layer holds them, and the host never has to be told about a model.
    """
    alpha = torch.exp(-torch.exp(A_log) * torch.nn.functional.softplus(a + dt_bias))
    return alpha, torch.sigmoid(b)


def normalise(q: torch.Tensor, k: torch.Tensor, *, scale: float
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """What the kernel does to the query and key before either touches the state.

    `use_qk_l2norm_in_kernel=True` is not decoration: the state read is a contraction against
    these vectors, and reading with the unnormalised ones is a different model that still produces
    text. Applied here so both ends of the split apply it once, in one place.
    """
    return (torch.nn.functional.normalize(q.float(), dim=-1) * scale,
            torch.nn.functional.normalize(k.float(), dim=-1))


def read(state: torch.Tensor, q: torch.Tensor, k: torch.Tensor
         ) -> tuple[torch.Tensor, torch.Tensor]:
    """The host's whole job in a linear layer: two contractions of the state it holds.

    `state` is (rows, value heads, value dim, key dim) and both queries are (rows, value heads,
    key dim) -- already expanded from the key heads, because the state is per value head and a
    read that broadcast wrongly here would mix one head's history into another's.

    Both readings are against the state as it stands BEFORE this step. That is what makes `h_q`
    sendable the moment the query arrives, one message ahead of the key and value.
    """
    if state.shape[:2] != q.shape[:2] or q.shape != k.shape:
        raise ValueError(
            f"state {tuple(state.shape)} against q {tuple(q.shape)} and k {tuple(k.shape)}: the "
            f"two readings and the state have to agree on rows and heads, and a broadcast that "
            f"papered over this would read one head's history for another and stay fluent."
        )
    return (torch.einsum("bhvk,bhk->bhv", state, q),
            torch.einsum("bhvk,bhk->bhv", state, k))


def mix(h_q: torch.Tensor, h_k: torch.Tensor, *, v: torch.Tensor, k: torch.Tensor,
        q: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """The layer's output, from the two readings and this step's key and value.

    Runs where the weights are: everything here is either a scalar per head or a vector the
    weight-holding side projected itself, so the history never has to travel.
    """
    al, be = alpha.unsqueeze(-1), beta.unsqueeze(-1)
    kq = (k * q).sum(-1).unsqueeze(-1)
    return al * h_q + be * (v.float() - al * h_k) * kq


def update(state: torch.Tensor, h_k: torch.Tensor, *, v: torch.Tensor, k: torch.Tensor,
           alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """The new state, from the same reading the mix used. Runs where the state lives.

    Takes `h_k` rather than recomputing it: it is the same contraction, and computing it twice
    would be the second read this split was careful to charge for once.
    """
    al, be = alpha.unsqueeze(-1), beta.unsqueeze(-1)
    written = be * (v.float() - al * h_k)
    return al.unsqueeze(-1) * state + written.unsqueeze(-1) * k.unsqueeze(-2)


def expand_to_value_heads(x: torch.Tensor, value_heads: int) -> torch.Tensor:
    """Repeat a key-head tensor across the value heads that share it.

    The state is per value head and the query and key are per key head, so one of them has to be
    expanded before they meet. Done explicitly rather than by broadcasting, because a broadcast
    with the factor wrong is a model that reads a neighbour's history and says nothing.
    """
    heads = x.shape[1]
    if value_heads % heads:
        raise ValueError(
            f"{value_heads} value head(s) do not divide among {heads} key head(s); the expansion "
            f"factor is not an integer and no rounding of it is the model."
        )
    return x.repeat_interleave(value_heads // heads, dim=1)
