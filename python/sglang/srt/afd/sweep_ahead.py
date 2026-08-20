"""Run the cache sweep while the pool runs the feed-forward.

This is the module the arrangement is for. Everything else -- the read point, the wire, the
departure policy -- exists so that these three lines can happen in this order:

    issue(layer j's feed-forward to the pool)      host sends, does not wait
    sweep(layer j+N's query over its KV cache)     GPU works; the query came from h_j
    collect(layer j's feed-forward)                host waits, GPU is busy

The middle line is only possible because layer j+N's query reads h_j, which is on hand BEFORE
layer j's feed-forward runs. At the standard read point the query needs x_(j+N), which needs
every feed-forward between here and there, and the host has nothing to do but wait.

## Where each piece is triggered

    prepare_mlp wrapper    h_j is stashed and the forward batch is noted. A new batch object
                           means a new pass, and any sweep left unconsumed by the last one is
                           dropped here rather than merged into this one's attention.
    PoolRouting            calls `sweep_after_issue(j)` between its issue and its collect. The
                           order is not cosmetic: `issue` copies the hidden states to the host,
                           which synchronises the stream, so a sweep launched first would be
                           waited on by the send and the window would close before it opened.
    no pool                the same call at the end of the prepare_mlp wrapper. There is no
                           window to fill, and the split still runs, so that a quality number and
                           a latency number are measured on one wiring rather than two.
    attn.forward wrapper   joins this step's token onto the swept cache.

## The query is computed here, not at the layer that uses it

A converted layer runs its query projection on a different input from its key and value, and on a
fused qkv projection that means running the projection twice. Doing the early one HERE puts that
second projection inside the window too: the layer that uses it then projects once, for its key
and value, and reads the query out of this stash. Same arithmetic, more of it hidden.

`_prepare_method_name` mirrors `self_attention`'s dispatch rather than reusing it, because
`self_attention` computes q, k and v together and the whole point is to compute q alone. A mirror
drifts, and a drifted one here would hand the layer a query projected by a variant it did not
call -- fluent output, wrong model. So the name travels with the query, and the wrapper on the
variant that is actually called refuses a query that came from another.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable

import torch
from sglang.srt.afd.split_attention import PerPassIndex, join, split_refusal, sweep
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_npu, is_xpu

logger = logging.getLogger(__name__)

_IS_CUDA, _IS_HIP, _IS_NPU, _IS_XPU, _IS_CPU = is_cuda(), is_hip(), is_npu(), is_xpu(), is_cpu()


def _prepare_method_name(layer) -> str:
    """Which forward_prepare_* variant this layer's `self_attention` will call.

    A mirror of the dispatch in Qwen3_5AttentionDecoderLayer.self_attention, restricted to the
    decode path -- the only forward mode this split is installed for, so the branch that turns on
    `is_extend_or_draft_extend_or_mixed` cannot be taken.
    """
    if _IS_CUDA and layer.attn_output_gate:
        return "forward_prepare_cuda_fused"
    if (_IS_HIP or _IS_XPU or _IS_CPU) and layer.attn_output_gate:
        return "forward_prepare_fused_gate"
    if not _IS_NPU or not layer.attn_output_gate:
        return "forward_prepare_native"
    return "forward_prepare_npu"


class SweepAhead:
    """The sweep-ahead schedule, installed on a stack that already has the Early-Q read point."""

    def __init__(self, model, source_of: dict[int, int], stash):
        self.layers = model.model.layers
        # the same stash the read point already fills; a second copy of h_l would be a second
        # answer to "what did layer j produce", and only one of them would be wrong
        self.stash = stash
        # Every converted layer has SOMETHING that reads only h_(l-N): its query projection,
        # which a converted layer runs on a different input from its key and value and therefore
        # runs twice. That projection goes in the window on all of them.
        #
        # Only a softmax layer also has a cache to sweep. A linear-attention layer's query
        # multiplies a recurrent state, and the state is read by the update as well, so
        # partitioning that recurrence would read the biggest tensor in the layer twice to hide
        # one of the two reads. Measured on this model at batch 4: the state is 12.6 MB, q^T S
        # takes 21 us and the update re-reads it for another 21 us, against a 6500 us pool round
        # trip. There is nothing there. The projection is 47 layers x its own cost and is free,
        # because it is already being run.
        self.sweep_at: dict[int, int] = {}
        self.softmax_targets: set[int] = set()
        for target, source in sorted(source_of.items()):
            if source < 0:
                continue
            self.sweep_at[source] = target
            if hasattr(self.layers[target], "attn"):
                self.softmax_targets.add(target)
        self.routed_layers: set[int] = set()

        self.index = PerPassIndex()
        self.pending: dict[int, object] = {}
        self._batch = None
        self._forward_batch = None

        self.n_sweeps = 0
        self.n_projections = 0
        self.n_joins = 0
        self.n_fallbacks = 0
        self.refusals: Counter = Counter()
        # --afd-verify-split: recompute each join the fused way and keep the worst disagreement
        self.verify_path: str | None = None
        self.verify: dict[int, dict] = {}
        self._undo: list[Callable] = []
        self._install_joins()

    # -- pass bookkeeping ---------------------------------------------------

    def on_prepare_mlp(self, layer_id: int, forward_batch) -> None:
        """Note the batch h_l belongs to, and open the window if this layer is a source."""
        if self._batch is not forward_batch:
            # A sweep that outlives its pass would be joined onto a different batch's token: the
            # shapes can even agree, and the output would be an attention over another request's
            # cache. Dropped at the boundary rather than checked at the join.
            self._batch = forward_batch
            self.pending.clear()
            # and the queries that went with them. A stashed query outliving its pass is worse
            # than a stashed sweep: `pending` is consumed by the join, which would notice a
            # missing partner, but `_afd_q_precomputed` is consumed by the projection wrapper,
            # which would simply use it -- last pass's query over this pass's cache.
            for target in self.sweep_at.values():
                self.layers[target]._afd_q_precomputed = None
                self.layers[target]._afd_qkvz_precomputed = None
        self._forward_batch = forward_batch
        if layer_id in self.sweep_at and layer_id not in self.routed_layers:
            self.sweep_after_issue(layer_id)

    # -- the window ---------------------------------------------------------

    def sweep_after_issue(self, source_layer: int) -> None:
        """Project the reading layer's query from h_source and sweep its cache with it."""
        target = self.sweep_at.get(source_layer)
        if target is None or target in self.pending:
            return
        forward_batch = self._forward_batch
        hidden = self.stash.get(source_layer)
        if forward_batch is None or hidden is None:
            return

        layer = self.layers[target]
        # cleared before the refusals below, not after them: a pass that refuses to sweep must
        # leave no query behind for the layer to find
        layer._afd_q_precomputed = None
        layer._afd_qkvz_precomputed = None

        if target not in self.softmax_targets:
            self._project_linear_ahead(layer, hidden)
            return

        backend = get_attn_backend()
        reason = split_refusal(backend, layer.attn, forward_batch)
        if reason is not None:
            self._refuse(reason)
            return

        method = _prepare_method_name(layer)
        with torch.no_grad():
            normed = layer.input_layernorm(hidden)
            q, _, _, gate = getattr(layer, method)(
                positions=forward_batch.positions, hidden_states=normed
            )
            state = sweep(backend, layer.attn, q, forward_batch, self.index)
        if state is None:
            self._refuse("every request is one token long; there is no cache to sweep")
            return
        self.pending[target] = state
        # the method name travels with the query so the layer can refuse a query projected by a
        # variant it is not about to call. `_prepare_method_name` is a mirror of a dispatch that
        # lives in the model file, and a mirror drifts silently -- this is what makes it loud.
        layer._afd_q_precomputed = (q, gate, method)
        self.n_sweeps += 1

    def _project_linear_ahead(self, layer, hidden) -> None:
        """A linear-attention layer's fused input projection, on the early stream, in the window.

        This is the whole of what such a layer can do before x_l exists. The conv1d that follows
        mixes over time using a state this must not disturb, and the gates, the key and the value
        all come from x_l. So the projection runs here and the layer splices its query slice out
        of the result instead of projecting a second time -- the same arithmetic the read point
        already costs, moved inside the pool round trip.
        """
        with torch.no_grad():
            normed = layer.input_layernorm(hidden)
            early_qkvz, _ = layer.linear_attn._forward_input_proj(normed)
        layer._afd_qkvz_precomputed = early_qkvz
        self.n_projections += 1

    def _refuse(self, reason: str) -> None:
        """Count it, and say it once. A refusal that only lands in a counter is a schedule that
        quietly ran synchronously under the overlapped arm's name."""
        if reason not in self.refusals:
            logger.info("afd sweep-ahead: not splitting -- %s", reason)
        self.refusals[reason] += 1

    # -- the join -----------------------------------------------------------

    def _install_joins(self) -> None:
        for target in sorted(self.sweep_at.values()):
            layer = self.layers[target]
            layer._afd_q_precomputed = None
            layer._afd_qkvz_precomputed = None
            if target not in self.softmax_targets:
                continue
            original = layer.attn.forward
            layer.attn.forward = self._joined(target, layer.attn, original)
            self._undo.append(
                lambda ly=layer, o=original: setattr(ly.attn, "forward", o)
            )

    def _joined(self, target: int, attn, original: Callable) -> Callable:
        def forward(q, k, v, forward_batch, save_kv_cache: bool = True, **kwargs):
            state = self.pending.pop(target, None)
            self.layers[target]._afd_q_precomputed = None
            if state is None or kwargs or not save_kv_cache or k is None or v is None:
                # no sweep ran, or the call carries something the partition does not speak for
                # (rope-split keys, a cross-attention layer, a caller that suppresses the KV
                # write). The fused path is correct; it is only counted, so a run that fell back
                # every step cannot report the split path's name.
                self.n_fallbacks += 1
                return original(q, k, v, forward_batch, save_kv_cache=save_kv_cache, **kwargs)
            if q is not state.q:
                raise RuntimeError(
                    f"layer {target} swept the cache with one query and is joining with another. "
                    f"The two halves must share a query: attending the cache with q_a and this "
                    f"step's token with q_b is not attention with either."
                )
            k = k.view(-1, attn.tp_k_head_num, attn.qk_head_dim)
            v = v.view(-1, attn.tp_v_head_num, attn.v_head_dim)
            out = join(get_attn_backend(), attn, k, v, forward_batch, state, self.index)
            self.n_joins += 1
            if self.verify_path is not None:
                self._verify(target, out, original, q, k, v, forward_batch)
            return out

        return forward

    def _verify(self, target: int, out, original: Callable, q, k, v, forward_batch) -> None:
        """The same attention, fused, on the same real traffic, and how far apart the two are.

        Run AFTER the join, which has already written this step's key and value to the cache, so
        the fused call sees exactly the positions the two partitions covered between them and
        `save_kv_cache=False` is not a different question. The two differ only in the order of the
        additions inside one softmax, so the number to watch is a bfloat16 rounding, and anything
        larger is the partition covering the wrong positions rather than covering them in a
        different order.
        """
        reference = original(q, k, v, forward_batch, save_kv_cache=False)
        scale = reference.abs().max()
        gap = (out.float() - reference.float()).abs().max()
        record = self.verify.setdefault(
            target, {"max_abs": 0.0, "max_rel": 0.0, "scale": 0.0, "joins": 0}
        )
        abs_gap = float(gap)
        magnitude = float(scale)
        record["joins"] += 1
        record["max_abs"] = max(record["max_abs"], abs_gap)
        record["scale"] = max(record["scale"], magnitude)
        if magnitude > 0.0:
            record["max_rel"] = max(record["max_rel"], abs_gap / magnitude)
        self._write_verify()

    def _write_verify(self) -> None:
        import json

        with open(self.verify_path, "w") as f:
            json.dump(
                {
                    "layers": {str(k): v for k, v in sorted(self.verify.items())},
                    "worst_rel": max((v["max_rel"] for v in self.verify.values()), default=0.0),
                    "record": self.record(),
                },
                f,
                indent=2,
            )

    # -- reporting ----------------------------------------------------------

    def record(self) -> dict:
        return {
            "windows": len(self.sweep_at),
            "sweep_windows": len(self.softmax_targets),
            "projection_windows": len(self.sweep_at) - len(self.softmax_targets),
            "projections": self.n_projections,
            "window_layers": {str(s): t for s, t in sorted(self.sweep_at.items())},
            "sweeps": self.n_sweeps,
            "joins": self.n_joins,
            "fallbacks": self.n_fallbacks,
            "refusals": dict(self.refusals),
        }

    def remove(self) -> None:
        for fn in self._undo:
            fn()
        self._undo.clear()
        self.pending.clear()
        self.index.clear()


def resolve_split(requested) -> bool:
    """Whether to partition attention, from the flag alone.

    Unset means on. The split changes the order of the additions inside softmax, so a bits-per-byte
    number measured without it and a latency number measured with it describe two models that
    differ by more than the thing being reported. Serving one wiring in both makes the pair a pair.

    Off is a real setting -- it is how the fused reference in `test_split_exactness_on_the_stack`
    is obtained -- and it is not a fallback: `split_refusal` handles the forwards that cannot be
    partitioned, and counts them.
    """
    if requested is None:
        return True
    if not isinstance(requested, bool):
        raise TypeError(f"--afd-split-attention is a flag, got {requested!r}")
    return requested


def install_sweep_ahead(model, wiring) -> SweepAhead | None:
    """Install the schedule on a stack whose read point has already been moved.

    Returns None when the wiring moved nothing, or moved nothing that sweeps a cache: a schedule
    with no window is not a degraded schedule, it is the synchronous one, and returning an object
    that reports zero windows invites a benchmark to call it the overlapped arm.
    """
    source_of = {p.layer: p.source for p in wiring.plan.points if not p.clamped}
    ahead = SweepAhead(model, source_of, wiring.stash)
    if not ahead.sweep_at:
        ahead.remove()
        logger.info(
            "afd sweep-ahead: no layer sweeps a cache at this read point, so there is no half "
            "that can run before x_l exists. Running synchronously."
        )
        return None
    logger.info(
        "afd sweep-ahead: %s window(s) -- %s carrying a cache sweep, %s carrying a query "
        "projection alone (linear attention, whose state the update reads again)",
        len(ahead.sweep_at), len(ahead.softmax_targets),
        len(ahead.sweep_at) - len(ahead.softmax_targets),
    )
    return ahead
