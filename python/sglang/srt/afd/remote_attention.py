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
from sglang.srt.afd.protocol import OP_KVPROJ, OP_RELEASE, OP_SWEEP, Frame

# the layer field is unused by a release; naming it keeps the frame readable
RELEASE_LAYER = 0


def sweep_frame(request_id: int, layer: int, *, q: torch.Tensor, hidden: torch.Tensor,
                positions: torch.Tensor) -> Frame:
    """The call. Positions travel as int64 so the rotation is applied to the right places."""
    return Frame(request_id, layer, (q, hidden, positions.view(-1, 1).to(torch.int64)), OP_SWEEP)


def join(o_swept: torch.Tensor, lse_swept: torch.Tensor, score: torch.Tensor,
         v_now: torch.Tensor, *, heads: int, kv_heads: int, v_head_dim: int,
         dtype: torch.dtype) -> torch.Tensor:
    """Fold this step's token into the swept cache. A two-way softmax is a logistic.

    No query here, and none needed: the query enters the merge only through the scalar q.k_t per
    head, which the pool computed with the same query it swept with. What is left is a lerp --
    rank-1, GEMV-shaped, and the only part of attention that cannot be issued before x_l exists.
    """
    tokens = v_now.shape[0]
    group = heads // kv_heads
    v3 = v_now.view(tokens, kv_heads, v_head_dim).float().repeat_interleave(group, dim=1)
    o3 = o_swept.view(tokens, heads, v_head_dim).float()

    weight = torch.sigmoid(lse_swept.float() - score.float()).unsqueeze(-1)
    # an empty cache has lse = -inf, and lerp with weight 0 is this step's own value, which is
    # exactly right -- but -inf reaches here as a NaN weight on some paths, so it is named
    weight = torch.where(torch.isnan(weight), torch.zeros_like(weight), weight)
    merged = torch.lerp(v3, o3, weight)
    return merged.reshape(tokens, heads * v_head_dim).to(dtype)


class RemoteAttention:
    """Route a converted layer's attention to the pool that holds its cache.

    Installed on `layer.attn.forward`, so the host still projects the query and still applies the
    output projection; what leaves is the key and value projection, the cache, and the sweep.

    The host's normalised x_l is captured on the way past `forward_prepare_*` rather than
    recomputed, because the pool applies W_k and W_v to it directly. Re-normalising on the far end
    would be a second implementation of the same norm, and the two agreeing is not something
    anything downstream checks.

    ## Prefill goes over the wire too, and has to

    The first version routed only decode and let prefill fall back to the local path. That writes
    the prompt's keys into the HOST's cache, which nothing then reads: the pool starts its history
    empty and the model answers from nothing. It ran, and it produced "Paris, Paris, Paris" for a
    prompt about prime numbers -- visibly wrong rather than subtly, which is the only good thing
    about it. The sweep already handles a chunk correctly, each token seeing everything before its
    own position including the earlier tokens of the same chunk, so extend goes over the wire.

    ## The request id is the slot, and position zero frees it

    The first version used a constant, so a second request attended the first one's history. The
    id is the request's own pool slot; when a token arrives at position 0 that slot is starting a
    new sequence, which is the moment to drop what was there. sglang reuses slots, so without that
    the cache would be a different request's.

    ## One request at a time, for now

    A decode forward carries one token from each of N requests and this wrapper sends them as one
    frame, so N > 1 would append every request's key to one request's history. It refuses instead.
    Lifting it means a frame per request, or a request id per row.
    """

    def __init__(self, model, client, layers: tuple[int, ...]):
        self.model = model
        self.client = client
        self.layers = layers
        self.calls = 0
        self.refusals = 0
        self._undo = []
        self._install()

    def _install(self) -> None:
        for layer_id in self.layers:
            layer = self.model.model.layers[layer_id]
            layer._afd_normed_input = None
            for name in ("forward_prepare_cuda_fused", "forward_prepare_fused_gate",
                         "forward_prepare_native", "forward_prepare_npu"):
                original = getattr(layer, name)
                setattr(layer, name, self._capture(layer, original))
                self._undo.append(lambda ly=layer, n=name, o=original: setattr(ly, n, o))
            original_attn = layer.attn.forward
            layer.attn.forward = self._remote(layer, layer_id, layer.attn, original_attn)
            self._undo.append(
                lambda ly=layer, o=original_attn: setattr(ly.attn, "forward", o)
            )

    def _capture(self, layer, original):
        def wrapped(positions, hidden_states, **kwargs):
            layer._afd_normed_input = hidden_states
            return original(positions=positions, hidden_states=hidden_states, **kwargs)

        return wrapped

    def _remote(self, layer, layer_id: int, attn, original):
        def forward(q, k, v, forward_batch, save_kv_cache: bool = True, **kwargs):
            normed = layer._afd_normed_input
            mode = forward_batch.forward_mode
            if kwargs or normed is None or not (mode.is_decode() or mode.is_extend()):
                self.refusals += 1
                return original(q, k, v, forward_batch, save_kv_cache=save_kv_cache, **kwargs)
            if forward_batch.batch_size != 1:
                raise RuntimeError(
                    f"the pool keys its cache by request and this frame carries "
                    f"{forward_batch.batch_size} requests' tokens; sending them as one would "
                    f"append every request's key to one request's history"
                )
            positions = forward_batch.positions.reshape(-1)
            request_id = int(forward_batch.req_pool_indices[0]) + 1
            if int(positions[0]) == 0 and layer_id == self.layers[0]:
                # a token at position 0 means this slot is starting a new sequence. sglang reuses
                # slots, so without dropping what was there the sweep would read another
                # request's history -- which is what "Paris, Paris, Paris" was.
                self.client.call(request_id, RELEASE_LAYER, (positions.reshape(-1, 1),),
                                 OP_RELEASE, "cpu")
            o_swept, lse, score, v_now = self.client.call(
                request_id, layer_id,
                (normed, positions.reshape(-1, 1).to(torch.int64)),
                OP_SWEEP, q.device,
            )
            self.calls += 1
            return join(o_swept, lse, score, v_now,
                        heads=attn.tp_q_head_num, kv_heads=attn.tp_k_head_num,
                        v_head_dim=attn.v_head_dim, dtype=q.dtype)

        return forward

    def record(self) -> dict:
        return {"layers": list(self.layers), "calls": self.calls, "refusals": self.refusals}

    def remove(self) -> None:
        for fn in self._undo:
            fn()
        self._undo.clear()


def install_remote_attention(model, client, layer_types: list[str]) -> RemoteAttention:
    """Send every softmax layer's attention to the pool that holds its cache."""
    from sglang.srt.afd.read_point import full_attention_layers

    layers = tuple(full_attention_layers(layer_types))
    if not layers:
        raise RuntimeError("no layer sweeps a cache; there is nothing to move to the pool")
    return RemoteAttention(model, client, layers)


class RemoteKVProjection:
    """Arrangement C: the pool projects the key, value and gate; the host keeps the cache.

    The line moves one step back from `RemoteAttention`. The pool holds W_k, W_v and the gate --
    weight-shared work, which is the only kind a departure can amortise -- and holds no state at
    all. The host keeps the cache, the sweep and the join, which is the work that amortises across
    nothing because each request reads its own history.

    That division is the one the measurements point at. With the cache on the pool the sweep is
    88% of what the pool costs per token at 16k context and does not amortise, and the host has
    nothing left to run inside a round trip, so the Early-Q window closes. Here both come back.

    What it costs is the wire: k, v and the gate return every layer every token, 16 KB against the
    swept output's shape, in exchange for W_k and W_v -- 1.34 GB over 64 layers -- off the host.

    The host still runs its own fused projection to get the query, and throws away the key, value
    and gate it produced. That is waste, and it is the honest first cut: slicing the query's rows
    out of a fused FP8 weight means carrying its scales too, and a comparison between arrangements
    should not also be a comparison between two ways of slicing a weight.
    """

    def __init__(self, model, client, layers: tuple[int, ...]):
        self.model = model
        self.client = client
        self.layers = layers
        self.calls = 0
        self.refusals = 0
        self._undo = []
        for layer_id in layers:
            layer = model.model.layers[layer_id]
            for name in ("forward_prepare_cuda_fused", "forward_prepare_fused_gate",
                         "forward_prepare_native", "forward_prepare_npu"):
                original = getattr(layer, name)
                setattr(layer, name, self._wrapped(layer_id, original))
                self._undo.append(lambda ly=layer, n=name, o=original: setattr(ly, n, o))

    def _wrapped(self, layer_id: int, original):
        def prepare(positions, hidden_states, **kwargs):
            q, k, v, gate = original(positions=positions, hidden_states=hidden_states, **kwargs)
            if kwargs:
                self.refusals += 1
                return q, k, v, gate
            k_pool, v_pool, gate_pool = self.client.call(
                0, layer_id,
                (hidden_states, positions.reshape(-1, 1).to(torch.int64)),
                OP_KVPROJ, q.device,
            )
            self.calls += 1
            return (q, k_pool.reshape(k.shape), v_pool.reshape(v.shape),
                    gate_pool.reshape(gate.shape))

        return prepare

    def record(self) -> dict:
        return {"layers": list(self.layers), "calls": self.calls, "refusals": self.refusals}

    def remove(self) -> None:
        for fn in self._undo:
            fn()
        self._undo.clear()


def install_kv_projection(model, client, layer_types: list[str]) -> RemoteKVProjection:
    from sglang.srt.afd.read_point import full_attention_layers

    layers = tuple(full_attention_layers(layer_types))
    if not layers:
        raise RuntimeError("no softmax layer to move a projection off")
    return RemoteKVProjection(model, client, layers)


def join_scored(o_swept, lse_swept, q, k_now, v_now, *, attn):
    """Fold this step's token in, using the host's own query, key and value.

    In the two-pool split the host sends its query to the cache pool, so the sweep and the score
    are taken with THE SAME query -- the one this side projected. That is why the score is
    computed here rather than returned: there is only one query in the system, and it is here.
    """
    heads, kv_heads = attn.tp_q_head_num, attn.tp_k_head_num
    head_dim, v_head_dim = attn.qk_head_dim, attn.v_head_dim
    tokens = q.shape[0]
    group = heads // kv_heads
    q3 = q.view(tokens, heads, head_dim).float()
    k3 = k_now.view(tokens, kv_heads, head_dim).float().repeat_interleave(group, dim=1)
    v3 = v_now.view(tokens, kv_heads, v_head_dim).float().repeat_interleave(group, dim=1)
    o3 = o_swept.view(tokens, heads, v_head_dim).float()
    score = (q3 * k3).sum(-1) * attn.scaling
    weight = torch.sigmoid(lse_swept.float() - score).unsqueeze(-1)
    weight = torch.where(torch.isnan(weight), torch.zeros_like(weight), weight)
    return torch.lerp(v3, o3, weight).reshape(tokens, heads * v_head_dim).to(q.dtype)
