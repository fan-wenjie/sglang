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


class SweepService:
    """The pool's answer to one SWEEP frame: this layer's cache swept, and the last one's
    feed-forward, in a single round trip.

    The two jobs ride together because the Early-Q read point makes them simultaneous. A frame for
    the boundary between layer l-1 and layer l carries

        h_(l-1)   the residual after layer l-1's attention, which layer l-1's feed-forward takes
        q_l       layer l's query, which the host projected from that same h_(l-1)
        positions for the rotation

    and neither depends on the other. The sweep reads the CACHE, which is already here and has
    nothing to do with this step, so it can start the moment the frame lands; the feed-forward runs
    for x_l, which is what the key and value are then projected from. One round trip per layer
    rather than two, and it is the moved read point that allows it -- at the standard read point
    q_l is a function of x_l and the two jobs are strictly ordered.

    The key and value go back with the answer because the JOIN stays on the host. It could run
    here -- k_t and v_t are right there -- but then the pool's work would depend on this step's
    token and could no longer be issued early, which is the whole property being bought.
    """

    def __init__(self, model, holder: KVHolder, layer_types: list[str]):
        self.layers = model.model.layers
        self.holder = holder
        self.layer_types = layer_types

    def _prepare_kv(self, layer, positions: torch.Tensor, x: torch.Tensor):
        """This layer's key and value, through the model's own projection.

        Calls the layer's `forward_prepare_native` and discards its query. That computes a query
        nobody wants, and it is still the right call: the alternative is reimplementing the fused
        projection's slicing, the qk-norm and the rotation, and a reimplementation is correct on
        the configurations it was written against and wrong on the others without failing.
        """
        with torch.no_grad():
            normed = layer.input_layernorm(x)
            _, k, v, _ = layer.forward_prepare_native(positions=positions, hidden_states=normed)
        return k, v

    def serve_attention(self, request_id: int, layer_id: int, normed: torch.Tensor,
                        positions: torch.Tensor):
        """Everything the host's join needs, from the host's normalised input alone.

        The query does not cross the wire. It is a function of the same normalised input the key
        and value are projected from, and that input has to be sent anyway, so projecting it here
        costs one amortised GEMM and saves 12 KB a layer a token -- 22 us of wire against about
        0.35 us of arithmetic at 64 requests.

        More than bytes: the host's join needs the query ONLY to form the scalar q.k_t per head.
        Computing that scalar here means every term of the merge

            lerp(v_t, o_swept, sigmoid(lse_swept - score))

        comes from ONE query -- this one. Send the query instead and the host merges an lse taken
        with the pool's query against a score taken with its own, and the two differ, because the
        same fp8 matmul picks different kernels on different cards. That gap was measured at 2.5e-4
        on this pair, small and entirely avoidable.

        `forward_prepare_native` produces the query, key and value in one call, which is also the
        model's own qk-norm and rotation rather than a second implementation of them.
        """
        layer = self.layers[layer_id]
        attn = layer.attn
        heads, kv_heads = attn.tp_q_head_num, attn.tp_k_head_num
        head_dim = attn.qk_head_dim
        with torch.no_grad():
            q, k_now, v_now, _ = layer.forward_prepare_native(
                positions=positions, hidden_states=normed
            )
        o_swept, lse = self._sweep_with(request_id, layer_id, q, k_now, v_now)

        tokens = q.shape[0]
        group = heads // kv_heads
        q3 = q.view(tokens, heads, head_dim).float()
        k3 = k_now.view(tokens, kv_heads, head_dim).float().repeat_interleave(group, dim=1)
        score = (q3 * k3).sum(-1) * attn.scaling
        return o_swept, lse, score.to(torch.float32), v_now

    def project_kv(self, layer_id: int, normed: torch.Tensor, positions: torch.Tensor):
        """The key, value and gate alone -- no cache, no sweep, no state.

        This is the other place the line can be drawn. The pool holds W_k, W_v and the gate's
        share of the fused projection; the HOST keeps the cache and does the whole attention. The
        pool never sees a query and never holds a byte of anyone's history, so it stays what it
        was: shareable, freely batched, restartable.

        What it costs is the wire. The key, value and gate go back every layer every token where
        the cache-holding arrangement sends only a swept output, and what it buys is the weights
        of those projections off the host.
        """
        layer = self.layers[layer_id]
        with torch.no_grad():
            _, k, v, gate = layer.forward_prepare_native(
                positions=positions, hidden_states=normed
            )
        if gate is None:
            raise RuntimeError(
                f"layer {layer_id} projects no gate, and this frame's shape assumes one. A "
                f"model without the output gate needs its own reply shape rather than a "
                f"placeholder the host would silently multiply by."
            )
        return k, v, gate

    def _sweep_with(self, request_id: int, layer_id: int, q: torch.Tensor,
                    k_now: torch.Tensor, v_now: torch.Tensor):
        """Append this step's key and value, then sweep everything before each token's own slot."""
        attn = self.layers[layer_id].attn
        heads, kv_heads = attn.tp_q_head_num, attn.tp_k_head_num
        head_dim, v_head_dim = attn.qk_head_dim, attn.v_head_dim

        k_cache = k_now.view(-1, kv_heads, head_dim).transpose(0, 1).contiguous()
        v_cache = v_now.view(-1, kv_heads, v_head_dim).transpose(0, 1).contiguous()
        k_all, v_all = self.holder.append(request_id, layer_id, k_cache, v_cache)

        tokens = q.shape[0]
        q3 = q.view(tokens, heads, head_dim)
        outs, lses = [], []
        for t in range(tokens):
            # everything strictly before this token's own position, which for a chunk includes the
            # earlier tokens of the same chunk -- they are already in the cache
            seen = k_all.shape[POSITION_AXIS] - (tokens - 1 - t) - 1
            o, lse = sweep_cache(q3[t], k_all[:, :seen], v_all[:, :seen],
                                 scaling=attn.scaling, kv_group=heads // kv_heads)
            outs.append(o)
            lses.append(lse)
        return (torch.stack(outs).reshape(tokens, heads * v_head_dim).to(q.dtype),
                torch.stack(lses).to(torch.float32))

    def serve(self, request_id: int, layer_id: int, q: torch.Tensor, hidden: torch.Tensor,
              positions: torch.Tensor):
        """One frame in, the tensors of one reply out."""
        layer = self.layers[layer_id]
        attn = layer.attn
        heads, head_dim = attn.tp_q_head_num, attn.qk_head_dim
        kv_heads = attn.tp_k_head_num

        # the previous layer's feed-forward, and the residual it completes
        with torch.no_grad():
            ffn_out = layer_feed_forward(self.layers[layer_id - 1], hidden)
            x = hidden + ffn_out
            k_now, v_now = self._prepare_kv(layer, positions, x)

        o_swept, lse_swept = self._sweep_with(request_id, layer_id, q, k_now, v_now)
        return o_swept, lse_swept, k_now, v_now, x


def layer_feed_forward(layer, hidden: torch.Tensor) -> torch.Tensor:
    """One layer's feed-forward, unwrapping the tuple some blocks return."""
    out = layer.mlp(hidden)
    return out[0] if isinstance(out, tuple) else out
