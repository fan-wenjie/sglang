"""Install the Early-Q read point on a loaded Qwen3.5 stack, without forking the model file.

Three hooks, each at a point the model already names:

    prepare_mlp        its second return value IS h_l -- the residual after this layer's attention
                       and before its feed-forward. Wrapped on the layers some other layer reads,
                       so the stack does not hold activations nobody wants.
    forward_prepare_*  the four variants all take `hidden_states` and return (q, k, v, gate). The
                       wrapper runs the same one twice when a source is set: once on this layer's
                       own normalised input for the key and value, once on the source's for the
                       query. Wrapping all four uniformly avoids re-implementing the dispatch,
                       which would drift from upstream the first time a branch is added.
    forward            sets the source before the attention runs and clears it after, so a layer
                       that is not converted cannot inherit the previous one's.

The query and the key/value therefore come from different points of the residual stream, and
everything else -- the norms, the rotation, the qk-norm, the gate, the parameter count -- is the
model's own.

## What this costs, and why it is written this way first

`qkv_proj` is fused: one projection emits q, gate, k and v together. Reading the query from a
different stream therefore runs that projection twice on a converted layer. Slicing the q rows out
of the fused weight would avoid it, and under FP8 that means carrying the scales too -- worth doing,
and not worth doing before the arrangement is known to run at all. The extra projection is
arithmetic on the compute side, which is the side this arrangement is trying to keep busy.

## The stash is keyed by layer, not by request

One forward pass is one batch: `h_l` is a single tensor covering every token in it. There is no
per-request loop here to confuse, and the (request, layer) keying that `pool_client` needs is a
property of the WIRE, where two requests really are in flight at once, not of this.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from sglang.srt.afd_query_shift.checkpoint import resolve_coverage, resolve_shift
from sglang.srt.afd.supported import check_supported
from sglang.srt.afd.sweep_ahead import install_sweep_ahead, resolve_split
from sglang.srt.afd_query_shift.read_point import (
    ReadPlan,
    convertible_layers,
    full_attention_layers,
    is_full_attention,
    layer_types_of,
    plan_read_points,
)

logger = logging.getLogger(__name__)

PREPARE_METHODS = (
    "forward_prepare_cuda_fused",
    "forward_prepare_fused_gate",
    "forward_prepare_native",
    "forward_prepare_npu",
)


class PassHooks:
    """What the per-pass wrappers call, filled in after they are installed.

    The prepare_mlp wrapper is installed with the read point; the sweep-ahead schedule is
    installed on top of it and needs to be called from inside it. A mutable holder is the join
    between them, so neither has to be built twice.
    """

    def __init__(self) -> None:
        self.sweep_ahead = None


class LayerStash:
    """`h_l` for the layers some converted layer reads, for the duration of one forward pass."""

    def __init__(self) -> None:
        self._h: dict[int, torch.Tensor] = {}

    def put(self, layer: int, h: torch.Tensor) -> None:
        self._h[layer] = h

    def get(self, layer: int) -> torch.Tensor | None:
        return self._h.get(layer)

    def clear(self) -> None:
        self._h.clear()

    def __len__(self) -> int:
        return len(self._h)


def _wrap_prepare_mlp(layer, layer_id: int, stash: LayerStash, hooks: PassHooks) -> Callable:
    original = layer.layer_communicator.prepare_mlp

    def wrapped(hidden_states, residual, forward_batch, *args, **kwargs):
        hidden_states, residual = original(
            hidden_states, residual, forward_batch, *args, **kwargs
        )
        # residual here is h_l. The layer's OUTPUT would be x_(l+1), which is the standard read
        # point with extra steps: a correct model, no overlap, and nothing to notice it by.
        stash.put(layer_id, residual)
        if hooks.sweep_ahead is not None:
            hooks.sweep_ahead.on_prepare_mlp(layer_id, forward_batch)
        return hidden_states, residual

    layer.layer_communicator.prepare_mlp = wrapped
    return original


def _wrap_prepare(layer, name: str) -> Callable:
    """Run the same prepare twice when a source is set: q from it, k and v from this layer.

    `self_attention` calls every variant by keyword, so the wrapper does too rather than
    guessing at positions.
    """
    original = getattr(layer, name)

    def wrapped(positions, hidden_states, **kwargs):
        q, k, v, gate = original(positions=positions, hidden_states=hidden_states, **kwargs)
        early = layer._afd_q_precomputed
        if early is not None:
            if early[2] != name:
                raise RuntimeError(
                    f"the sweep window projected this layer's query with {early[2]} and the "
                    f"layer called {name}. The two variants fuse the norm and the rotation "
                    f"differently, so this is a query the model did not ask for -- and the only "
                    f"symptom would be fluent output from an attention nobody wrote."
                )
            # the window already ran this projection on the early stream, inside the pool round
            # trip. Running it again here would be the same arithmetic outside the window.
            return early[0], k, v, early[1]
        source = layer._afd_q_hidden
        if source is None:
            return q, k, v, gate
        q_early, _, _, gate_early = original(
            positions=positions, hidden_states=source, **kwargs
        )
        return q_early, k, v, gate_early

    setattr(layer, name, wrapped)
    return original


def _query_rows(attn):
    """The rows of the fused input projection that produce the query, as a view.

    Returns None when the weight is not a plain contiguous `[out, in]` tensor -- a quantised or
    sharded parameter is not sliceable this way, and falling back to the full projection there is
    correct and merely slower. Refusing instead would make a speed optimisation into a load
    failure on checkpoints this has never been run against.
    """
    weight = getattr(getattr(attn, "in_proj_qkvz", None), "weight", None)
    if weight is None or weight.dim() != 2 or not weight.is_contiguous():
        return None
    rows = attn.key_dim // attn.attn_tp_size
    if rows <= 0 or rows > weight.shape[0]:
        return None
    view = weight[:rows]
    if view.data_ptr() != weight.data_ptr():
        return None                     # not a view; a copy here would double a 160 MiB weight
    return view


def _wrap_linear_input_proj(layer) -> Callable:
    """A linear-attention layer's query slice, read from the earlier stream.

    `in_proj_qkvz` is fused and splits `[key, key, value, value]`, so the query is the FIRST
    slice. The projection is run twice when a source is set -- once on this layer's own normalised
    input for the key, value and gate, once on the source's for the query -- and the two are
    spliced at the projection's output. That splice is safe here because the conv1d that follows
    is depthwise (`groups=conv_dim`): each channel is filtered independently, so replacing a
    contiguous channel range does not mix it with the ones beside it.

    The state stays here. A linear-attention layer holds a recurrent state that belongs to the
    request, so it cannot move to a stateless pool -- only the feed-forward leaves.

    ## The early projection reads only the rows it uses

    `in_proj_qkvz` is 16384 wide on this model and the early stream takes the first 2048 of it --
    the query. Running the whole projection to keep an eighth of it reads 87.5% of a 160 MiB weight
    for nothing, twice a layer, on 48 of 64 layers.

    So the query's rows are sliced out ONCE, at install. A row slice of a `[out, in]` weight is
    contiguous, so it is a view: no copy, no second copy of the weight resident, and the saving is
    the bytes that are never read. Measured at 12.3 us a layer at batch 4, 0.59 ms a decode step,
    which is the same size as what a second key projection costs -- so this pays for Early-K
    outright, and it is worth having whether or not Early-K is ever switched on.
    """
    original = layer.linear_attn._forward_input_proj
    attn = layer.linear_attn
    query_rows = _query_rows(attn)

    def wrapped(hidden_states, *args, **kwargs):
        qkvz, ba = original(hidden_states, *args, **kwargs)
        early_qkvz = layer._afd_qkvz_precomputed
        # consumed, so a later pass that opens no window cannot splice this one's query in --
        # the pass boundary clears it too, and this is the tighter of the two
        layer._afd_qkvz_precomputed = None
        source = layer._afd_q_hidden
        k_tp = attn.key_dim // attn.attn_tp_size
        if early_qkvz is None:
            if source is None:
                return qkvz, ba
            if query_rows is not None:
                # only the query's rows, and none of `in_proj_ba` -- the early stream reads
                # neither the key, the value, the gate nor the two per-head scalars
                early_qkvz = torch.nn.functional.linear(source, query_rows)
            else:
                early_qkvz, _ = original(source, *args, **kwargs)
        if qkvz.shape[-1] < k_tp:
            raise RuntimeError(
                f"the fused projection is {qkvz.shape[-1]} wide and the query slice is {k_tp}; "
                f"the split this splices at is not the split the model uses"
            )
        spliced = torch.cat([early_qkvz[..., :k_tp], qkvz[..., k_tp:]], dim=-1)
        return spliced, ba

    layer.linear_attn._forward_input_proj = wrapped
    return original


def _wrap_layer_forward(layer, layer_id: int, source_of: dict[int, int],
                        stash: LayerStash, norm) -> Callable:
    original = layer.forward

    def wrapped(*args, **kwargs):
        src = source_of.get(layer_id)
        h = stash.get(src) if src is not None and src >= 0 else None
        # LN1 belongs with the attention sub-layer, so the early stream gets the same
        # normalisation this layer's own input would have got.
        layer._afd_q_hidden = norm(h) if h is not None else None
        try:
            return original(*args, **kwargs)
        finally:
            layer._afd_q_hidden = None

    layer.forward = wrapped
    return original


class InstalledWiring:
    """What was changed, so a run can record it and a test can put it back."""

    def __init__(self, plan: ReadPlan, stash: LayerStash, undo: list[Callable],
                 hooks: PassHooks | None = None):
        self.plan = plan
        self.stash = stash
        self.hooks = hooks if hooks is not None else PassHooks()
        self._undo = undo

    def record(self) -> dict:
        return self.plan.as_record()

    def remove(self) -> None:
        if self.hooks.sweep_ahead is not None:
            self.hooks.sweep_ahead.remove()
            self.hooks.sweep_ahead = None
        for fn in self._undo:
            fn()
        self._undo.clear()
        self.stash.clear()


def _resolve_settings(shift_layers, coverage, hf_config):
    """What to serve at, from the flag and the checkpoint.

    A caller contradicting the checkpoint is warned rather than silently obeyed: serving a
    repaired checkpoint at the wrong read point feeds its query projection an input it was never
    trained on.
    """
    if hf_config is not None:
        shift_layers = resolve_shift(shift_layers, hf_config)
        coverage = resolve_coverage(coverage, hf_config)
    return (0 if shift_layers is None else shift_layers,
            "all" if coverage is None else coverage)


def _plan_for(model, *, shift_layers: int, coverage: str, layer_types) -> ReadPlan:
    """Which layer reads which, refusing a conversion that would hook nothing."""
    if layer_types is None:
        layer_types = layer_types_of(model)
    if coverage == "all":
        convertible = convertible_layers(layer_types)
    elif coverage == "softmax":
        convertible = full_attention_layers(layer_types)
    else:
        raise ValueError(f'coverage is "all" or "softmax", got {coverage!r}')
    n_layers = len(model.model.layers)
    plan = plan_read_points(shift_layers, n_layers, convertible=convertible)
    if not plan.moved:
        raise RuntimeError(
            f"--afd-query-shift-layers={shift_layers} moves no layer of this {n_layers}-layer "
            f"stack. A conversion that hooks nothing costs nothing, and a cost of zero reads as "
            f"tolerance rather than as a wiring that never installed."
        )
    return plan


def _install_hooks(layers, plan: ReadPlan, stash: LayerStash, hooks: PassHooks):
    """Wrap the layers the plan names. Returns the undo list and the layers that were stashed."""
    undo, source_of = [], {p.layer: p.source for p in plan.points if not p.clamped}
    needed = sorted({s for s in source_of.values()})

    for layer_id in needed:
        layer = layers[layer_id]
        original = _wrap_prepare_mlp(layer, layer_id, stash, hooks)
        undo.append(lambda ly=layer, o=original: setattr(ly.layer_communicator, "prepare_mlp", o))

    for layer_id in sorted(source_of):
        layer = layers[layer_id]
        layer._afd_q_hidden = None
        layer._afd_q_precomputed = None
        # the window's product for a linear-attention layer: its fused input projection, already
        # run on the early stream, so this layer splices instead of projecting a second time
        layer._afd_qkvz_precomputed = None
        if not is_full_attention(layer):
            # a linear-attention layer: one fused projection, splice the query slice
            original = _wrap_linear_input_proj(layer)
            undo.append(
                lambda ly=layer, o=original: setattr(ly.linear_attn, "_forward_input_proj", o)
            )
        else:
            for name in PREPARE_METHODS:
                # Accessed directly, not guarded: all four exist on this class, and a missing one
                # means the model file changed under this patch, which should be loud.
                original = _wrap_prepare(layer, name)
                undo.append(lambda ly=layer, n=name, o=original: setattr(ly, n, o))
        original = _wrap_layer_forward(layer, layer_id, source_of, stash, layer.input_layernorm)
        undo.append(lambda ly=layer, o=original: setattr(ly, "forward", o))
    return undo, needed


def _report(plan: ReadPlan, coverage: str, n_stashed: int) -> None:
    """The record goes in the log, not only into a return value a caller may drop.

    A run that converted fewer layers than asked would otherwise report a shallower shift's cost
    under a deeper shift's name, with nothing in the log to notice it by.
    """
    logger.info(
        "afd early-q installed: coverage=%s, shift=%s (%s half-layers, offset %.1f), "
        "%s layer(s) moved, %s clamped, %s layer(s) stashed",
        coverage, plan.shift_layers, plan.half_layers, plan.offset_layers,
        len(plan.moved), len(plan.clamped), n_stashed,
    )
    logger.info("afd early-q record: %s", plan.as_record())


def install_early_q(model, shift_layers, layer_types: list[str] | None = None,
                    coverage=None, span_cut=False, hf_config=None, split_attention=None,
                    verify_split=None) -> InstalledWiring:
    """Move the query's read point, on every layer or on the softmax ones alone.

        coverage="all"     every layer that has a query, including linear attention. 63 of 64 on
                           this model, and the coverage a deployment converts.
        coverage="softmax" only the layers whose attention is a sweep over a cache. 16 of 64.
                           Useful as an ablation; NOT the same number, and the study measured the
                           gap: +0.0181 bits per byte against +0.0211, a factor of 1.17.

    `layer_types` is derived from the built stack when not given, so a caller in a frozen
    orchestrator can ask for the wiring without computing its inputs.
    """
    if span_cut:
        # The GROUP cut implements the read point itself, in `SpanRunner._finish`: it projects the
        # next group's query from the residual between the last linear attention and the last
        # feed-forward. Installing this wiring as well applies the shift TWICE to one model, and
        # not harmlessly -- this converts by wrapping `layer.forward` and hooking
        # `post_attention_layernorm`, which is the module the span calls to advance its own
        # residual stream.
        #
        # It WAS installed, on the pool, alongside the span. The log read "coverage=all, shift=1,
        # 63 layer(s) moved" -- every layer except layer 0, which is exempt because nothing sits
        # beneath it to read. And layer 0 was the one layer whose output matched the colocated
        # model exactly, while every other layer sat a few percent out in a direction that was not
        # a rescaling. The span was measured for days as though it were the only transformation on
        # the model it was running on.
        # Shift ZERO, not None. The host's role wiring reads `afd_early_q.hooks.sweep_ahead`, so
        # the hooks object still has to be built -- returning None took the host down with
        # `'NoneType' object has no attribute 'hooks'`. Zero is the standard wiring and the only
        # value that means off, so nothing is converted and everything downstream still has what
        # it asks for.
        logger.info(
            "afd: --afd-span-cut implements its own read point, so the per-layer early-q wiring "
            "converts nothing. --afd-query-shift-layers=%s is honoured by the span itself.",
            shift_layers,
        )
        shift_layers = 0

    shift_layers, coverage = _resolve_settings(shift_layers, coverage, hf_config)
    if shift_layers == 0:
        # nothing to install. Returning an empty wiring rather than None keeps the caller from
        # having to branch, and its record still says what was resolved.
        return InstalledWiring(plan_read_points(0, len(model.model.layers)), LayerStash(), [])

    # Checked before anything is wrapped, and reported whole. Wrapping first and discovering the
    # gaps one AttributeError at a time -- from inside a wrapper, during a forward -- is how a
    # family that spells these differently would learn the contract, and it would learn it in
    # production.
    check_supported(model, coverage=coverage)
    plan = _plan_for(model, shift_layers=shift_layers, coverage=coverage, layer_types=layer_types)
    stash, hooks = LayerStash(), PassHooks()
    undo, stashed = _install_hooks(model.model.layers, plan, stash, hooks)
    _report(plan, coverage, len(stashed))

    wiring = InstalledWiring(plan, stash, undo, hooks)
    # The schedule goes on top of the read point, not beside it: it needs the plan to know which
    # layer reads which, and the stash to find h_l when the window opens.
    if resolve_split(split_attention):
        hooks.sweep_ahead = install_sweep_ahead(model, wiring)
        if hooks.sweep_ahead is not None and verify_split:
            hooks.sweep_ahead.verify_path = verify_split
            logger.warning(
                "afd: --afd-verify-split recomputes every join the fused way and writes the "
                "worst disagreement to %s. Attention runs twice; this is not a latency "
                "configuration.",
                verify_split,
            )
    return wiring


# ---------------------------------------------------------------------------
# The same read plan, on a HuggingFace stack.
#
# sglang's layer fuses q, k and v into one projection, so moving the query
# there costs a second projection. A HuggingFace Qwen3.5 layer keeps q_proj,
# k_proj and v_proj apart, so the query's input can simply be substituted --
# cheaper, and a cleaner statement of the intervention.
#
# This exists so the SAME plan drives both. A quality number measured on one
# and a schedule measured on the other are then about one rewiring; two
# installers driven by two notions of the read point would be two.
# ---------------------------------------------------------------------------


def _hf_layers(model):
    """The decoder stack, wherever this family keeps it.

    A vision-language wrapper keeps the text stack one level deeper. Hardcoding `model.model.layers`
    finds nothing there and a conversion that hooks nothing costs nothing, which reads as tolerance.
    """
    for path in (
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "model", "layers"),
        ("layers",),
    ):
        node = model
        for name in path:
                # a SEARCH over candidate paths, not defensive access: absence is the
            # answer this loop is looking for
            node = getattr(node, name, None)
            if node is None:
                break
        if node is not None and len(node) > 0:
            return node
    raise RuntimeError(
        f"no decoder stack found on a {type(model).__name__}; name the attribute rather than "
        f"letting the search return an empty list"
    )


class HFEarlyQ:
    """h_l captured at post_attention_layernorm's input; q_proj's input substituted."""

    def __init__(self, model, plan: ReadPlan):
        self.plan = plan
        self.layers = _hf_layers(model)
        self._h: dict[int, torch.Tensor] = {}
        self._handles = []
        self._source_of = {p.layer: p.source for p in plan.points if not p.clamped}
        needed = {s for s in self._source_of.values()}

        for layer_id in sorted(needed):
            layer = self.layers[layer_id]
            self._handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    self._stash(layer_id)
                )
            )
        for layer_id in sorted(self._source_of):
            layer = self.layers[layer_id]
            self._handles.append(
                layer.self_attn.q_proj.register_forward_pre_hook(
                    self._substitute(layer_id, layer.input_layernorm)
                )
            )

    def _stash(self, layer_id: int):
        def hook(_module, args):
            # post_attention_layernorm's input IS h_l: the stream after this layer's attention
            # and before its feed-forward.
            self._h[layer_id] = args[0]

        return hook

    def _substitute(self, layer_id: int, norm):
        source = self._source_of[layer_id]

        def hook(_module, args):
            h = self._h.get(source)
            if h is None:
                raise RuntimeError(
                    f"layer {layer_id} wants h_{source} and it is not stashed; the layer that "
                    f"produces it has not run, which is a plan error rather than a race"
                )
            return (norm(h),) + tuple(args[1:])

        return hook

    def record(self) -> dict:
        return self.plan.as_record()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._h.clear()


def install_early_q_hf(model, shift_layers: int, layer_types: list[str]) -> HFEarlyQ:
    """The same plan as `install_early_q`, on a HuggingFace stack."""
    layers = _hf_layers(model)
    convertible = full_attention_layers(layer_types)
    plan = plan_read_points(shift_layers, len(layers), convertible=convertible)
    if shift_layers > 0 and not plan.moved:
        raise RuntimeError(
            f"--afd-query-shift-layers={shift_layers} moves no layer of this {len(layers)}-layer stack"
        )
    logger.info(
        "afd early-q (hf): shift=%s, %s moved, %s clamped",
        plan.shift_layers, len(plan.moved), len(plan.clamped),
    )
    return HFEarlyQ(model, plan)
