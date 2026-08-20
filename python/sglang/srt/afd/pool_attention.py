"""The pool's half of attention: the key and value projections, the cache, and the sweep.

Moving the KV cache to the pool inverts what each side holds, and the reason it is worth doing is
arithmetic rather than tidiness:

    host   a query projection, a rank-1 join, an output projection. Per request, no weight sharing
           across requests, memory-bound on nothing -- GEMV work, and its wall clock barely moves
           from 4 concurrent requests to 24 (measured: 4.47s against 4.54s)
    pool   the feed-forward, the key and value projections, the cache, and the sweep over it. One
           weight read serves every caller in the departure -- GEMM work, and its cost per token
           falls 5.6x from a 4-token departure to a 512-token one

## The host never forms a key or a value

The pool computed the feed-forward, so it already holds

    x_l = h_(l-1) + ffn(h_(l-1))

which is exactly the input the key and value projections take. Giving it W_k and W_v means those
tensors are born on the side that stores them and never cross the wire. The host sends its query
and the residual it had to send anyway for the feed-forward.

Under the Early-Q read point the query comes from h_(l-1), so the host can send it in the SAME
frame as the residual -- one round trip per layer, carrying both jobs, and the pool answers with
the swept attention and the feed-forward output together.

## What stays on the host, and why it is not laziness

The JOIN -- folding this step's token into the swept cache -- could run here too, since the pool
has k_t and v_t. It stays on the host because the sweep is the half that can start early and the
join is the half that cannot: the sweep needs only the query, and Q-First puts the query a layer
ahead. Keeping the join on the host is what makes the pool's work purely "sweep the cache", which
is the only shape that can be issued before x_l exists. The pool therefore returns k_t and v_t
with the answer -- 16 KB against the 48 KB of the sweep's own output on this model.

## The cache here is append-only

A decode stream appends one position per step, so this holds a plain growing buffer per
(request, layer) rather than sglang's allocator and radix tree. That is a real limitation and it
is stated rather than hidden: no prefix sharing, no eviction, no fork. It is enough to measure the
arrangement and not enough to serve on, and the integration point -- replacing the host's
`token_to_kv_pool` reads for converted layers -- is where those come back.
"""

from __future__ import annotations

import logging
import threading

import torch

logger = logging.getLogger(__name__)


# Where the positions live in a cached tensor. The cache is (kv_heads, positions, head_dim),
# which is the layout `sweep_cache` reads with einsum "hjd" -- and getting this wrong concatenates
# along the HEAD axis instead, which grows the tensor, passes a shape check, and hands the sweep a
# model with more key heads than the checkpoint has.
POSITION_AXIS = 1


class KVHolder:
    """Per (request, layer) keys and values, appended one decode step at a time.

    Tensors are (kv_heads, positions, head_dim). The wire carries 2D frames, so whoever unpacks a
    frame reshapes into this before appending; the axis is named here because the one bug this
    class can have that nothing downstream catches is appending along the wrong one.
    """

    def __init__(self, device: torch.device | str, max_context: int):
        self.device = device
        self.max_context = max_context
        self._lock = threading.Lock()
        self._k: dict[tuple[int, int], torch.Tensor] = {}
        self._v: dict[tuple[int, int], torch.Tensor] = {}

    def append(self, request_id: int, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Add this step's positions, (kv_heads, new, head_dim), and return the whole cache.

        Returned under the lock and used outside it: a decode stream is the only writer of its own
        key, so the tensors cannot change under the reader, and holding the lock across the sweep
        would serialise every caller behind one GEMM.
        """
        key = (request_id, layer)
        with self._lock:
            have_k, have_v = self._k.get(key), self._v.get(key)
            k_all = k if have_k is None else torch.cat([have_k, k], dim=POSITION_AXIS)
            v_all = v if have_v is None else torch.cat([have_v, v], dim=POSITION_AXIS)
            if k_all.shape[POSITION_AXIS] > self.max_context:
                raise RuntimeError(
                    f"request {request_id} layer {layer} reached "
                    f"{k_all.shape[POSITION_AXIS]} cached "
                    f"positions, past the {self.max_context} this pool was given room for. An "
                    f"append-only cache cannot evict; it refuses instead of overwriting."
                )
            self._k[key], self._v[key] = k_all, v_all
        return k_all, v_all

    def release(self, request_id: int) -> int:
        """Drop every layer of one request. Returns how many were held."""
        with self._lock:
            keys = [key for key in self._k if key[0] == request_id]
            for key in keys:
                self._k.pop(key, None)
                self._v.pop(key, None)
        return len(keys)

    def positions(self, request_id: int, layer: int) -> int:
        with self._lock:
            held = self._k.get((request_id, layer))
        return 0 if held is None else held.shape[POSITION_AXIS]

    def bytes_held(self) -> int:
        with self._lock:
            return sum(t.numel() * t.element_size() for t in self._k.values()) + sum(
                t.numel() * t.element_size() for t in self._v.values()
            )


def sweep_cache(q: torch.Tensor, k_cached: torch.Tensor, v_cached: torch.Tensor,
                *, scaling: float, kv_group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention over the cached positions, as (output, log partition).

    Grouped-query: each key-value head serves `kv_group` query heads, expanded here rather than
    materialised in the cache, so the cache stores what the projection produced.

    Returns the log partition as well as the output because the host has to merge this with its
    own rank-1 join, and a softmax partitioned without its log partition cannot be merged -- the
    weight each half carries is exactly the ratio of the two.
    """
    heads = q.shape[0]
    if k_cached.shape[0] * kv_group != heads:
        raise RuntimeError(
            f"the sweep was given {heads} query head(s) and {k_cached.shape[0]} key head(s) at a "
            f"group size of {kv_group}; the cache and the query disagree about the model"
        )
    # accumulate at least in float32 -- a bfloat16 softmax over thousands of positions loses the
    # tail -- but never DOWN-cast, so a float64 caller checking exactness gets float64 back
    acc = torch.promote_types(q.dtype, torch.float32)
    k = k_cached.repeat_interleave(kv_group, dim=0).to(acc)
    v = v_cached.repeat_interleave(kv_group, dim=0).to(acc)
    scores = torch.einsum("hd,hjd->hj", q.to(acc), k) * scaling
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.einsum("hj,hjd->hd", torch.softmax(scores, dim=-1), v)
    return out, lse
