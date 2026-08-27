"""One linear-attention layer, split at the recurrence: weights here, history over there.

This is standard AFD's, not the derived line's, and the distinction is worth stating because the
code was written on the group cut's runner and looked derived for it. Splitting a linear layer at
the recurrence needs NO moved read point:

    here        input projection, gates, normalisation, the convolution, and the query
                coefficient q~ = q - beta (k.q) k, which folds the key's correction into the query
                so the far end contracts the state ONCE
    over there  r = S q~, one contraction of a state this side does not hold
    here        core = alpha r + beta (k.q) v, and the value never crossed the wire
    over there  the key, the value and the gates, deferred, to advance the state

`linear_history` proves that identity and `benchmark/afd/gdn_split.py` measures it against the
fused kernel. Both were already under `srt/afd`; only the class this sat on was not.

## What it means for the pool

With this, a pool can hold EVERY weight -- the feed-forwards and the linear projections -- while
the host holds every piece of state. That is the arrangement taken to its conclusion, and it is
what makes a pool interchangeable between two calls of one request.

One qualification, and it is not cosmetic: the CONVOLUTION RING stays here. It is this side's own
last K projections, 80 KiB a layer a request against the recurrent state's 3.00 MiB, and while it
is here a request is sticky to the pool that served its previous call. The recurrent state is
pulled back per call and the ring is not, so "holds nothing between calls" is true of the state
and not yet of the ring. Moving it is its own piece of work.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _dump(side: str, tag: str, value) -> None:
    """Save a tensor so the two sides can be diffed ELEMENTWISE, not by magnitude.

    Norms have carried this search as far as they can. They cannot separate "the right vector,
    scaled" from "a different vector of the same length" -- and the last reading said the
    difference is the second kind: layer 1's feed-forward input is `post_attention_layernorm`'s
    output, whose scale is removed by construction, and it still differed by 4.1%.

    So both sides write row 0 of each boundary to SGLANG_AFD_DUMP and the comparison happens
    offline, per element. Same prompt through the colocated model and through the arrangement
    gives the same embedding, so the tensors line up index for index.
    """
    where = envs.SGLANG_DEBUG_AFD_DUMP.get()
    if not where:
        return
    import torch as _t

    path = os.path.join(where, f"{side}__{tag}.pt")
    if os.path.exists(path):
        return
    _t.save(value.detach()[0].float().cpu(), path)


def slots_wanted() -> int:
    """How many requests the pool can hold recurrent state for, taken from the server's own limit.

    From `--max-running-requests` rather than guessed. A slot table that ran out would refuse a
    request mid-generation, and a number invented here would put that cliff somewhere the operator
    never chose. If sglang has not resolved the limit yet there is nothing to derive from, and
    refusing at startup beats picking one.
    """
    from sglang.srt.runtime_context import get_schedule

    limit = get_schedule().max_running_requests
    if not limit:
        raise ValueError(
            "the pool was asked to run linear-attention layers before --max-running-requests "
            "resolved, so there is nothing to size the slot table from. Set it explicitly."
        )
    return int(limit)


def build(model, *, device):
    """A runner over this model's linear layers, sized from the server's own request limit."""
    from sglang.srt.afd.linear_state import LinearStates

    config = model.config
    return LinearRunner(
        model,
        LinearStates(
            slots=slots_wanted(),
            num_v_heads=config.linear_num_value_heads,
            head_k_dim=config.linear_key_head_dim,
            head_v_dim=config.linear_value_head_dim,
            device=device,
        ),
    )


_STAGES = [0, 0.0, 0.0, 0.0, 0.0]  # count, before, callback, after, convolution
# The convolution is inside "before"; it is timed separately because moving it to the
# host is the one change that would make this pool stateless, and whether that is worth
# doing turns on how much of the 0.856 ms it actually is.


def _stage_parts(before_s: float, call_s: float, after_s: float) -> None:
    """A linear layer's three stages on the pool. `call` is the host round trip; the other two
    are this side's own arithmetic, and the budget says they should total ~200 us."""
    _STAGES[0] += 1
    _STAGES[1] += before_s
    _STAGES[2] += call_s
    _STAGES[3] += after_s
    if _STAGES[0] % 600:
        return
    n = _STAGES[0]
    logger.info(
        "afd linear stages: %s calls -- projections+coefficient %.3f ms (convolution %.3f of "
        "it), callback %.3f ms, core+output %.3f ms",
        n,
        1e3 * _STAGES[1] / n,
        1e3 * _STAGES[4] / n,
        1e3 * _STAGES[2] / n,
        1e3 * _STAGES[3] / n,
    )


class LinearRunner:
    """The linear layers' weights, and the arithmetic around a history it does not hold.

    Holds the model because a departure names a layer and this resolves it, and the slot table
    because the convolution ring is indexed by request. Holds no recurrent state: that is the
    caller's, read back across the wire through the callbacks the pool installs on `_local`.
    """

    # What this adds to the pool's HELLO word. Its own bit: a pool that serves whole spans (8) and
    # a pool that serves one linear layer a call are different offers, and a host that asks for
    # the second must not be satisfied by a pool advertising only the first.
    capability_bit = 16

    def __init__(self, model, states) -> None:
        self.model = model
        self.states = states
        # `ask_host` and `defer_update` are installed here by whoever serves a departure, per
        # thread, because two departures run at once and each answers a different caller.
        self._local = threading.local()
        # `_residual` is the SPAN runner's table of per-request residuals, and this class does
        # not own one. The two trace gates below ask whether a request is new by looking for it,
        # which is why they read it through `getattr`: a pool serving OP_LAYER has no span and
        # no table, and on a multi-row (prefill) call an attribute error would take the
        # departure thread down. Single-row decode never reached the gate, which is why this
        # only ever fired the first time a span ran a prefill.

    def linear_attention(self, attn, request_ids, layer_id: int, hidden: torch.Tensor):
        """One linear-attention layer: everything but the history, which is the host's.

        The weights are here and the recurrent state is not, so the layer is split at the one
        place the recurrence allows. `linear_history` proves the identity and
        `benchmark/afd/gdn_split.py` measures it against the fused kernel; what happens here is
        the arrangement of it across two machines:

            here        input projection, gates, normalisation, the convolution -- and the QUERY
                        COEFFICIENT q~ = q - beta (k.q) k, which folds the key's correction into
                        the query so the far end contracts the state ONCE
            over there  r = S q~, one contraction of a state this end does not hold
            here        core = alpha r + beta (k.q) v, and the value never crossed the wire
            over there  the key, the value and the gates, deferred, to advance the state

        The decay is applied HERE, to what comes back. Sending it would be sending a per-head
        scalar across so it could be multiplied and sent back, and the reading is the same shape
        either way.
        """
        _entered = time.perf_counter()
        ask = getattr(self._local, "ask_host", None)
        if ask is None:
            raise RuntimeError(
                f"layer {layer_id} has no way to reach the history. The recurrent state lives on "
                f"the caller's side under this cut, so a span with no callback would have to "
                f"either invent a state or hold one here -- the first is wrong and the second is "
                f"the arrangement this cut replaced."
            )
        # sglang's own names, not transformers'. The two libraries split this projection
        # differently -- one fused `in_proj_qkvz` here against four separate ones there -- and
        # writing the other library's names produced an AttributeError at the first token, which
        # is the same shape of mistake as the state layout being transposed between them.
        qkvz, _ = attn.in_proj_qkvz(hidden)
        ba, _ = attn.in_proj_ba(hidden)
        query, key, value, z, b, a = attn.fix_query_key_value_ordering(qkvz, ba)
        rows = hidden.shape[0]
        flat = [t.reshape(rows, -1) for t in (query, key, value)]
        packed = torch.cat(flat, dim=-1)
        # BEFORE the convolution. The model's `mixed_qkv`, which is what reaches
        # `self.attn(forward_batch, mixed_qkv=...)`, is pre-convolution too -- the convolution runs
        # inside the backend. Publishing the post-convolution tensor here compared one side's
        # filtered channels against the other's unfiltered ones and read 100% apart on every key
        # head while the layer's output agreed to 2%, which is impossible and is how it was caught.
        self._local.last_packed = packed.detach()
        # The only path. Before the convolution, deliberately: the ring is the CALLER's, so what
        # crosses is the PRE-convolution projection and this side never touches a ring.
        #
        # There used to be a second path that convolved here against a ring this side kept, and
        # `--afd-rings-on-host` chose between them. It is gone rather than defaulted off: the ring
        # was the only per-request thing a pool held, and holding it made a request STICKY to the
        # pool that answered its previous call. A pool that can only be asked again is not a pool.
        # A choice between "poolable" and "not" is not a choice worth carrying, and a flag that
        # nothing should ever set is a path nothing ever tests.
        return self._through_the_caller(
            attn, request_ids, layer_id, packed, a, b, z, rows, hidden.dtype
        )

    def _through_the_caller(
        self, attn, request_ids, layer_id, packed, a, b, z, rows, dtype
    ):
        """The layer when the caller holds the ring, which is what leaves this side stateless.

        One crossing, the same one that already happened. It used to carry `q~` down and `S q~`
        back; now it carries the PRE-convolution projection down and `core` back, and the caller
        does the convolution, the gates' effect on the state, and the update.

        `z` stays here. It is projected here and consumed here by the norm, so it never travels --
        which is why the traffic table in AFD_STATELESS_POOL.md was pessimistic by 12 KiB a layer
        a row.

        Nothing per request is written on this side in this path. That is the whole point: a
        request served this way can be answered by ANY pool, and its next call need not come back
        here.
        """
        mix = getattr(self._local, "mix_host", None)
        if mix is None:
            raise RuntimeError(
                f"layer {layer_id} was told the caller holds the convolution ring, and the caller "
                f"installed no way to reach it. The two ends were started for different "
                f"arrangements; refused here rather than falling back to a ring this side no "
                f"longer keeps."
            )
        from sglang.srt.afd.linear_history import gates

        alpha, beta = gates(a, b, attn.A_log, attn.dt_bias)
        core = mix(layer_id, request_ids, packed, alpha, beta)
        core = core.reshape(rows, -1).to(dtype)
        gated = attn.norm(
            core.reshape(-1, attn.head_v_dim), z.reshape(-1, attn.head_v_dim)
        )
        out, _ = attn.out_proj(gated.reshape(rows, -1))
        return out
