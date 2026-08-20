"""The host's half when the cache lives on the pool: send a query, join what comes back.

    host                                    pool
    q_l = W_q(LN1_l(h_(l-1)))               (Early-Q: the query exists before x_l does)
    send SWEEP(q_l, h_(l-1), positions) -->
                                            sweep layer l's cache with q_l        GEMM
                                            ffn(h_(l-1)); x_l = h_(l-1) + ffn
                                            k_t, v_t = W_kv_l(LN1_l(x_l)); append
                                        <-- o_swept, lse, k_t, v_t, x_l
    join(o_swept, lse, q_l, k_t, v_t)       rank-1, this step's token only         GEMV
    W_o(attn_out); carry on with x_l

The host never forms a key or a value and never holds a cache. What it keeps is the query
projection, a rank-1 join and the output projection -- and of those only the two projections are
weight-shared, which is the aggregation this arrangement trades away for per-request independence.

Measured on this model: W_q + W_o is 125.8 MB a layer, and their cost is flat from batch 1 to
batch 24 (39.2 us against 39.8 us), so batching them is worth 23.7x and giving it up costs 57 ms a
step at 24 requests. It is worth giving up only because the host is otherwise idle 97% of a step
waiting on the wire, and 57 ms fits inside 77 ms of waiting. If the wire ever gets fast, that
arithmetic inverts and this is the first thing to reconsider.
"""

from __future__ import annotations

import torch
from sglang.srt.afd.protocol import OP_SWEEP, Frame


def sweep_frame(request_id: int, layer: int, *, q: torch.Tensor, hidden: torch.Tensor,
                positions: torch.Tensor) -> Frame:
    """The call. Positions travel as int64 so the rotation is applied to the right places."""
    return Frame(request_id, layer, (q, hidden, positions.view(-1, 1).to(torch.int64)), OP_SWEEP)


def join(o_swept: torch.Tensor, lse_swept: torch.Tensor, q: torch.Tensor, k_now: torch.Tensor,
         v_now: torch.Tensor, *, heads: int, kv_heads: int, head_dim: int, v_head_dim: int,
         scaling: float) -> torch.Tensor:
    """Fold this step's token into the swept cache. A two-way softmax is a logistic.

    Kept on the host, and kept rank-1, because this is the half that cannot be issued before x_l
    exists. Everything the pool does can be; that asymmetry is the arrangement.
    """
    tokens = q.shape[0]
    group = heads // kv_heads
    q3 = q.view(tokens, heads, head_dim).float()
    k3 = k_now.view(tokens, kv_heads, head_dim).float().repeat_interleave(group, dim=1)
    v3 = v_now.view(tokens, kv_heads, v_head_dim).float().repeat_interleave(group, dim=1)
    o3 = o_swept.view(tokens, heads, v_head_dim).float()

    score = (q3 * k3).sum(-1) * scaling
    weight = torch.sigmoid(lse_swept.float() - score).unsqueeze(-1)
    # an empty cache has lse = -inf, and lerp with weight 0 is this step's own value, which is
    # exactly right -- but -inf reaches here as a NaN weight on some paths, so it is named
    weight = torch.where(torch.isnan(weight), torch.zeros_like(weight), weight)
    merged = torch.lerp(v3, o3, weight)
    return merged.reshape(tokens, heads * v_head_dim).to(q.dtype)
