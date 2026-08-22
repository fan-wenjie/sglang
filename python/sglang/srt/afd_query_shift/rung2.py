"""rung 2 of the ladder: the linear attention on the pool, one round trip a layer.

The ladder exists because standard AFD is token-identical to colocated and the group cut is not,
with four changes landed between them as one. This rung turns on exactly one of them -- the linear
attention moving to the pool -- and leaves the grouping, the residual held there, and the shifted
read point off. Its verdict against colocated says which of the two the fault is in:

    identical      the fault is the grouping and the residual held on the pool
    not identical  the fault is the move itself, and rung 1 comes next

The state stays on the HOST. An older `OP_LINEAR` path exists, unused, that puts it on the pool
and would have been cheaper to revive; a pool holding per-request state stops being stateless and
cannot be released between one request's calls, which is the property the whole arrangement rests
on. The cheaper path was available and is not the one taken.

So the residual never travels either. The host keeps it, does both norms and the feed-forward, and
sends only the layer's normalised input; the pool answers with that layer's attention output and
calls back for the recurrent state exactly as a span does.

Selected by SGLANG_AFD_RUNG=2 rather than by a server flag. The ladder is an experiment and its
rungs are not settings a deployment chooses -- and a flag in `server_args` would be the derived
arm's configuration living in shared code, which `test_afd_stands_alone.py` exists to prevent.
"""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.afd.arms import register
from sglang.srt.afd.layer_kinds import layer_types_of
from sglang.srt.afd.protocol import OP_LAYER

logger = logging.getLogger(__name__)


def wanted() -> bool:
    """Read where it is used, not at import. The scheduler is a spawned process and an answer
    cached in the parent is an answer about the wrong one."""
    return os.environ.get("SGLANG_AFD_RUNG") == "2"


class LinearOnPool:
    """Replaces each linear layer's attention with a call to the pool. Softmax layers untouched."""

    def __init__(self, model, client) -> None:
        self.model = model
        self.client = client
        self._undo: list = []
        self._rows: list[int] = []
        self._install()

    def _install(self) -> None:
        kinds = layer_types_of(self.model)
        moved = 0
        for index, kind in enumerate(kinds):
            if kind == "full_attention":
                continue
            layer = self.model.model.layers[index]
            self._replace(layer.linear_attn, index)
            moved += 1
        logger.info(
            "afd host: the per-layer linear cut is installed. %s linear layer(s) answer from the "
            "pool, one round trip each; the residual and both norms stay here, and so does the "
            "recurrent state.", moved,
        )

    def _replace(self, attn, layer_id: int) -> None:
        original = attn.forward

        def forward(hidden_states, forward_batch=None, **kwargs):
            rows = self._row_ids(forward_batch)
            self._rows = rows
            handle = self.client.issue_frame(
                int(rows[0]), layer_id, (hidden_states,), OP_LAYER)
            return self.client.collect_frame(handle, hidden_states.device)[0]

        attn.forward = forward
        self._undo.append(lambda a=attn, o=original: setattr(a, "forward", o))

    def current_rows(self):
        """The row ids of the call in flight, for the state reading the pool asks back for.

        Read through the routing rather than captured when the frame was sent: a captured list
        would be the FIRST layer's rows for every layer after it, and on a decode batch where row,
        request and token coincide that is invisible.
        """
        return list(self._rows)

    def _row_ids(self, forward_batch):
        """One id a ROW, expanded by the extend lengths.

        A prefill is many rows of one request, and a request id per REQUEST rather than per row is
        the mistake this tree has made four times -- decode makes row, request and token coincide,
        so it only shows on a prompt.
        """
        ids = [int(r) for r in forward_batch.req_pool_indices]
        lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if lens is None:
            return ids
        out = []
        for rid, n in zip(ids, lens):
            out.extend([rid] * int(n))
        return out

    def remove(self) -> None:
        for undo in self._undo:
            undo()


class Rung2Arm:
    name = "linear-on-pool"

    def wanted(self) -> bool:
        return wanted()

    def install_on_host(self, model, client, *, sweep_ahead):
        """rung 0's feed-forward offload AND this rung's linear attention. Both, or it is not a rung.

        A ladder's rung is the one below it plus one change. Installing only the linear-attention
        move would leave the feed-forward on the host -- a third arrangement that is neither rung 1
        nor rung 2, and whose verdict would answer a question nobody asked. The first version of
        this did exactly that, and would have measured it as rung 2.
        """
        from sglang.srt.afd.pool_client import PoolClient
        from sglang.srt.afd.roles import install_pool_routing, routable_layers
        from sglang.srt.afd_query_shift.installer import _span_slots

        client.require(PoolClient.NEEDS_FEED_FORWARD)
        feed_forward = install_pool_routing(
            model, client, routable_layers(model), sweep_ahead=sweep_ahead)
        routing = LinearOnPool(model, client)

        # The history the pool calls back for. Without it the pool asks for a state reading, this
        # end has no handler, and the departure dies thirty seconds later with "no state reading
        # for request 4 layer 0" -- which reads as a wire problem and is a missing service. Moving
        # the linear attention means moving the CALLS to its state, and whoever moves them owes
        # the answer.
        from sglang.srt.afd.history_service import HistoryService
        from sglang.srt.afd.linear_history import HistoryCache

        config = model.config
        cache = HistoryCache(
            slots=_span_slots(),
            layers=len(layer_types_of(model)),
            value_heads=config.linear_num_value_heads,
            head_k_dim=config.linear_key_head_dim,
            head_v_dim=config.linear_value_head_dim,
            conv_width=1, conv_taps=1,     # the ring stays on the pool; this holds no convolution
            device=next(model.parameters()).device,
        )
        routing.history = HistoryService(cache, rows_of=lambda frame: routing.current_rows())
        client.serve = routing.history
        return (feed_forward, routing)

    def make_pool_runner(self, model, *, device):
        """The runner built directly, not through `make_span_runner`.

        That helper gates on `span_cut_wanted()`, which is false for this rung -- it would hand the
        pool None and every OP_LAYER frame would meet "a LAYER frame reached a pool with no
        linear-attention runner". The runner itself is what this rung needs; the group cut's
        wanting of it is not.
        """
        from sglang.srt.afd.linear_state import LinearStates
        from sglang.srt.afd_query_shift.installer import _span_slots
        from sglang.srt.afd_query_shift.span import SpanRunner

        config = model.config
        states = LinearStates(
            slots=_span_slots(),
            num_v_heads=config.linear_num_value_heads,
            head_k_dim=config.linear_key_head_dim,
            head_v_dim=config.linear_value_head_dim,
            device=device,
        )
        return SpanRunner(model, states, layer_types=layer_types_of(model), query_shift=0)


register(Rung2Arm.name, Rung2Arm)
