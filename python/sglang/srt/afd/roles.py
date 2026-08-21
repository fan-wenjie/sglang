"""The two roles, as processes: a host that owns the cache and a pool that owns the weights.

    pool   loads the stack and answers `mlp(hidden)` for whichever layer a caller names
    host   loads the stack, installs the Early-Q read point, and sends every converted layer's
           feed-forward to the pool instead of running it locally

## What this buys, and where

The feed-forward genuinely leaves the host: the pool runs it, batches across callers, and the
host's own MLP weights go unused on the converted layers. That is the arrangement's plumbing, and
it is what a second machine needs.

The OVERLAP is `sweep_ahead`, passed in here as `between`. `self.attn(q, k, v, forward_batch)`
used to be one call -- the sweep over the cache and the fold-in of this step's token inside one
kernel -- so there was nothing the host could do between issuing a feed-forward and needing its
answer. `split_attention` partitions that call, and the half that needs only the query is launched
here, between the issue and the collect.

The window exists at a layer j only when layer j+N sweeps a cache. On Qwen3.8-27B that is 16 of
64 layers, because three of every four are linear attention, whose query multiplies a recurrent
state this does not know how to partition. The other 48 feed-forwards are still issued and waited
for. That is a property of the model's layer mix, and `SweepAhead.record()` reports the count so
a speedup is read against the number of windows that actually existed.

## Why the pool loads the whole model

It uses one layer's MLP at a time and could hold only those weights. Loading the stack is wasteful
and correct, and correctness first: a pool that loaded a subset would need its own weight-mapping
code, and a mapping bug there produces plausible tokens from the wrong weights.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
from collections.abc import Callable

import torch
from sglang.srt.afd.pool_client import PoolClient, PoolClosed
from sglang.srt.afd.pool_server import serve

logger = logging.getLogger(__name__)

# how often the host says whether its windows are opening. Often enough to see a schedule that
# stopped overlapping mid-run, rarely enough that the line is not the workload.
REPORT_EVERY = 512


def make_pool_forward(model) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """The pool's service: one layer's feed-forward, for a batch of tokens from any caller.

    Each layer's feed-forward is bound HERE, at construction, and never looked up again. When the
    two roles share one process -- which they do in the single-machine test -- the host's router
    replaces `layer.mlp.forward`, and a pool that looked the method up per call would find the
    router, send the work back out over the socket, and wait for itself. The symptom is a hang
    with an idle GPU and no error, and it is invisible on two machines, which is exactly when it
    would be found late.
    """
    layers = model.model.layers
    bound = [layer.mlp.forward for layer in layers]

    def forward(batch: torch.Tensor, layer: int) -> torch.Tensor:
        if not 0 <= layer < len(bound):
            raise RuntimeError(
                f"a caller asked for layer {layer} of a {len(bound)}-layer stack; the two sides "
                f"are not running the same model"
            )
        with torch.no_grad():
            out = bound[layer](batch)
        return out[0] if isinstance(out, tuple) else out

    return forward


def routable_layers(model) -> tuple[int, ...]:
    """The layers whose feed-forward this router can speak for.

    A sparse block takes a forward batch the wire does not carry, so it stays on the host. Chosen
    here rather than refused at the first token, because a stack that turns out to be half
    routable is a different arrangement from the one a benchmark was launched to measure.
    """
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    return tuple(
        i for i, layer in enumerate(model.model.layers)
        if not isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)
    )


def make_cache_pool(model, *, enabled, max_context: int, device):
    """The cache half of a two-pool split: a history, a sweep, an append. No weights."""
    if not enabled:
        return None
    from sglang.srt.afd.pool_attention import CachePool, KVHolder

    logger.info(
        "afd cache pool: histories only, up to %s positions a request. It holds no weight and "
        "answers no feed-forward: the sweep shares nothing between callers, so it is kept off "
        "the service whose whole economy is one weight read serving many.",
        max_context,
    )
    return CachePool(KVHolder(device, max_context), model.model.layers)


def attach_cache_pool(sweep_ahead, *, addr, connect_timeout_s: float = 30.0):
    """Point the Early-Q window at a cache pool, so its sweep goes out over the wire.

    This is what makes the two pools concurrent. The window already opens between the issue and
    the collect of the feed-forward -- that is the whole point of the moved read point -- so a
    sweep issued there is in flight to one machine while the feed-forward is in flight to another.
    """
    if sweep_ahead is None or not addr:
        return None
    from sglang.srt.afd.rendezvous import AppendLedger

    client = PoolClient(addr, connect_timeout_s)
    client.require(PoolClient.NEEDS_CACHE)
    sweep_ahead.cache_client = client
    sweep_ahead.ledger = AppendLedger()
    logger.info("afd host: sweeps go to the cache pool at %s, issued inside the window", addr)
    return client


def span_cut_wanted() -> bool:
    """Whether this process was launched for the group cut.

    Read from the global server args rather than passed down, for the reason `absent_ffn` reads it
    the same way: the pool role is set up inside the scheduler process, which is spawned, and a
    module-level flag set in the parent does not cross that boundary. That failure was silent once
    already here -- the feed-forward weights were built after the flag said not to -- so the value
    is fetched where it is used.
    """
    from sglang.srt.server_args import get_global_server_args

    return bool(get_global_server_args().afd_span_cut)


def _span_slots() -> int:
    """How many requests the pool can hold recurrent state for.

    Taken from `--max-running-requests` rather than guessed. A slot table that ran out would refuse
    a request mid-generation, and a number invented here would put that cliff somewhere the
    operator never chose. If sglang has not resolved the limit yet there is nothing to derive from,
    and refusing at startup beats picking one.
    """
    from sglang.srt.server_args import get_global_server_args

    limit = get_global_server_args().max_running_requests
    if limit is None:
        raise ValueError(
            "--afd-span-cut needs --max-running-requests to size the pool's recurrent state "
            "table. A recurrent state is the whole history compressed and cannot be rebuilt from "
            "a prefix, so the table refuses rather than evicts -- and where that refusal falls "
            "has to be a number somebody chose."
        )
    return int(limit)


def _span_query_shift() -> int:
    """How far back this pool reads the next group's query. Read HERE, on the pool.

    The knob is `--afd-query-shift-layers` and it looks like a host setting, because on the
    per-layer cut it was one. Under the group cut the query projection moved to the pool with
    `W_q`, so the pool is the side that decides where the query is read from and the host's copy
    of the flag decides nothing at all.

    That was not wired, and the span read the shifted point unconditionally. The cost was not a
    wrong model -- shift 1 is the operating point either way -- it was a lost control: a run
    launched with `--afd-query-shift-layers 0` to ask what the shift costs got a shifted read
    anyway, reported no difference, and the difference had never been asked for. Both ends must be
    given the same value until the protocol carries it.
    """
    from sglang.srt.server_args import get_global_server_args

    shift = get_global_server_args().afd_query_shift_layers
    if shift not in (0, 1):
        raise ValueError(
            f"--afd-query-shift-layers={shift} under the group cut. This cut reads the query from "
            f"between the last linear attention and the last feed-forward of a group, which is "
            f"shift 1, or from the group's output, which is shift 0. A deeper shift is a "
            f"different cut and would have to move the read point across a group boundary."
        )
    return int(shift)


def watch_colocated_residual(model) -> None:
    """Log the residual leaving every layer of the model's OWN forward, under SGLANG_AFD_SELFCHECK.

    The reference the span's residual trace has been missing. The pool holds the whole model and
    also runs the spans, so both sequences come from ONE process and ONE set of weights: serve a
    prompt on this server's own port and the model's per-layer residuals are logged; serve it
    through the arrangement and `_trace_residual` logs the span boundaries. A span boundary IS a
    layer boundary, so the two line up and the first index where they part is the answer.

    Without this the span sequence says only that the residual does not grow -- rms 1.07 at the
    first boundary falling to about 0.25 and staying there -- and "a residual stream should grow"
    is an assumption about this model, not a measurement of it. Every earlier round of this search
    that reasoned instead of measuring was wrong.
    """
    import os

    if not os.environ.get("SGLANG_AFD_SELFCHECK"):
        return
    seen = {"n": 0}
    limit = int(os.environ.get("SGLANG_AFD_SELFCHECK", "20"))

    def watch(index):
        def hook(_module, _args, output):
            if seen["n"] >= limit or not isinstance(output, tuple) or len(output) != 2:
                return
            residual = output[1]
            if residual is None:
                return
            seen["n"] += 1
            row = residual[0].float()
            logger.info(
                "afd colocated: layer %s -- rows %s wide %s |%.5g| rms %.5g max %.5g",
                index, residual.shape[0], row.numel(), float(row.norm()),
                float(row.pow(2).mean().sqrt()), float(row.abs().max()),
            )
        return hook

    def watch_gate(index):
        def hook(_module, args):
            if seen["n"] >= limit * 2:
                return
            seen["n"] += 1
            row = args[0][0].float()
            logger.info(
                "afd colocated gate: layer %s -- what o_proj is given |%.5g| rms %.5g max %.5g",
                index, float(row.norm()), float(row.pow(2).mean().sqrt()),
                float(row.abs().max()),
            )
        return hook

    def watch_embedding(_module, _args, output):
        if seen["n"] >= limit * 4:
            return
        seen["n"] += 1
        row = output[0].float()
        logger.info("afd colocated: EMBEDDING -- rows %s wide %s |%.5g| rms %.5g max %.5g",
                    output.shape[0], row.numel(), float(row.norm()),
                    float(row.pow(2).mean().sqrt()), float(row.abs().max()))

    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens.register_forward_hook(watch_embedding)

    def watch_mlp(index):
        def hook(_module, _args, output):
            if seen["n"] >= limit * 8:
                return
            seen["n"] += 1
            out = output[0] if isinstance(output, tuple) else output
            row = out[0].float()
            logger.info("afd colocated mlp: layer %s -- rows %s |%.5g| rms %.5g max %.5g",
                        index, out.shape[0], float(row.norm()),
                        float(row.pow(2).mean().sqrt()), float(row.abs().max()))
        return hook

    for index, layer in enumerate(model.model.layers):
        layer.register_forward_hook(watch(index))
        if hasattr(layer, "mlp"):
            layer.mlp.register_forward_hook(watch_mlp(index))
        # o_proj's INPUT is the attention output after the output gate -- the one quantity the
        # span computes from a gate it saved on a previous call, and the one this side has no
        # reference for. The span's gated output is 15x smaller than the attention output it
        # came from, consistently, and whether that is what a trained gate does or a fault is not
        # decidable without the model's own number for it.
        if hasattr(layer, "o_proj"):
            layer.o_proj.register_forward_pre_hook(watch_gate(index))
    logger.info("afd colocated: watching %s layer(s) for the residual reference",
                len(model.model.layers))


def watch_linear_attention(model, runner) -> None:
    """Run the span's linear attention beside the model's own, under SGLANG_AFD_SELFCHECK.

    Every PIECE of `SpanRunner._linear_attention` has been checked against something outside
    itself -- the projection split bit-identical, the convolution against sglang's own kernels,
    the scaling and gates and recurrence against the fused kernel in `gdn_split.py`, the head
    expansion against that same reference, the tail against the model's. What none of that reaches
    is the COMPOSITION: correct pieces in the wrong order, or with one of them missing, is still
    wrong, and the arrangement is still wrong in a case where every piece is trivially exercised.

    The comparison needs a real ForwardBatch and the pool never has one, because a span does not
    run inside a model forward. It does have one HERE: the pool serves requests on its own port,
    and during those `Qwen3_5GatedDeltaNet.forward(hidden, forward_batch)` is the real thing. So
    the hook takes that call's input, runs the span's reimplementation on the same tensor with a
    zeroed state and ring, and reports the difference -- one process, one set of weights, one
    input.

    The control is the same span call on a SHUFFLED input. Without it a small number means only
    that two functions of the same tensor are close, which two wrong functions can also be.
    """
    import os
    import threading

    if not os.environ.get("SGLANG_AFD_SELFCHECK"):
        return
    import torch

    from sglang.srt.model_executor.forward_context import get_attn_backend

    seen = {}
    limit = int(os.environ.get("SGLANG_AFD_SELFCHECK", "1"))
    scratch = 10_000_019          # far from any real request id

    before = {}

    def snapshot(layer_id):
        """The model's own conv and recurrent state BEFORE it runs, one slot's worth.

        Taken in a PRE-hook. A forward hook fires after the layer has already advanced its cache,
        so seeding from what it finds there would start the two recurrences a step apart -- a
        subtler version of the confound this exists to remove. The first reading of this comparison
        seeded the span from ZERO while the model carried whatever the warmup had left on the same
        slot, and 19% to 84% relative difference followed from that alone.
        """
        def hook(_module, args):
            if seen.get(layer_id, 0) >= limit:
                return
            hidden = args[0]
            if hidden.dim() != 2 or hidden.shape[0] != 1:
                return
            try:
                # the LINEAR backend, not the hybrid wrapper around it. `get_attn_backend()`
                # returns a HybridLinearAttnBackend on this model, which holds a full-attention
                # and a linear-attention backend side by side and has no forward_metadata of its
                # own -- reaching for one gets an AttributeError naming the wrapper.
                backend = get_attn_backend()
                backend = getattr(backend, "linear_attn_backend", backend)
                cache = backend.req_to_token_pool.mamba2_layer_cache(layer_id)
                index = int(backend.forward_metadata.mamba_cache_indices[0])
                before[layer_id] = (cache.conv[0][index].clone(),
                                    cache.temporal[index].clone())
            except Exception as e:                       # noqa: BLE001 -- diagnostic, reported
                before[layer_id] = None
                logger.info("afd linear: layer %s state not readable: %r", layer_id, e)
        return hook

    def compare(layer_id, attn):
        def hook(_module, args, output):
            if seen.get(layer_id, 0) >= limit:
                return
            hidden = args[0]
            if hidden.dim() != 2 or hidden.shape[0] != 1:
                return                       # one row only: the case the fault survives in
            seen[layer_id] = seen.get(layer_id, 0) + 1

            from sglang.srt.afd.split_read_kernel import read_one, update_only

            # `buffer(layer)`, not `.state[layer]`. LinearStates keeps its tensors in `_states`
            # behind an allocator, and reaching for a public name that does not exist raised
            # INSIDE a forward hook -- which took the scheduler down with it rather than skipping
            # the diagnostic. Everything below is inside the try for the same reason: a
            # measurement must not be able to kill the thing it is measuring.
            try:
                slot = runner.states.slot_of(scratch)
                state = runner.states.buffer(layer_id)
                channels = attn.conv1d.weight.shape[0]
                taps = attn.conv1d.weight.shape[-1]

                def ask_host(lid, request_ids, q_tilde, step=None):
                    slots = torch.full((q_tilde.shape[0],), slot, device=q_tilde.device,
                                       dtype=torch.long)
                    return read_one(runner.states.buffer(lid), slots, q_tilde).float()

                def defer_update(lid, request_ids, k, v, alpha, beta):
                    slots = torch.full((k.shape[0],), slot, device=k.device, dtype=torch.long)
                    update_only(runner.states.buffer(lid), slots,
                                k=k, v=v, alpha=alpha, beta=beta)

                local = runner._local
                local.ask_host, local.defer_update = ask_host, defer_update

                held = before.get(layer_id)
                if held is None:
                    logger.info("afd linear: layer %s has no snapshot; not compared", layer_id)
                    return
                conv_before, ssm_before = held

                def span_of(x):
                    # SEEDED from the model's own state rather than zeroed. Two recurrences
                    # started from different states differ for that reason alone, and the size of
                    # the difference says nothing until they start from the same one.
                    state[slot].copy_(ssm_before.reshape(state[slot].shape).to(state.dtype))
                    ring = runner.states.conv_buffer(
                        layer_id, width=channels, taps=taps, dtype=x.dtype)
                    # sglang keeps K-1 columns of history; this side keeps K, whose newest column
                    # the call writes itself. The history lines up at the OLD end.
                    ring[slot].zero_()
                    history = conv_before.reshape(channels, -1).to(ring.dtype)
                    ring[slot][..., 1:] = history[..., -(taps - 1):]
                    return runner._linear_attention(attn, [scratch], layer_id, x).float()

                mine = span_of(hidden)
                theirs = output.float()
                order = torch.randperm(hidden.shape[1], device=hidden.device)
                shuffled = span_of(hidden[:, order])
            except Exception as e:                       # noqa: BLE001 -- diagnostic, reported
                logger.info("afd linear: layer %s could not be compared: %r", layer_id, e)
                return
            finally:
                runner.release(scratch)

            def against(a, b):
                a, b = a.reshape(-1), b.reshape(-1)
                d = float((a - b).norm() / (b.norm() + 1e-9))
                c = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
                return f"rel {d:.6g} cos {c:+.4f}"

            # the seeded state's own magnitude, because the fork this decides is whether the
            # residual error appears only where the state is NON-zero. If it does, the suspect is
            # the state -- and the first suspect there is this seeding, not the span: sglang keeps
            # (value heads, head_v, head_k) and a reshape onto a differently-ordered layout
            # permutes silently, which is a mistake this tree has made between two libraries
            # already.
            def shape_of_the_error(a, b):
                """Is the difference a single scale, or is it structured?

                A cosine of 0.999 with 3 to 18 percent relative error is mostly magnitude, and
                magnitude has two very different explanations: ONE number wrong everywhere, which
                names a missing or doubled factor, or a spread, which does not. The per-channel
                ratio separates them -- a pure scale has every channel at the same value.

                Channels where the reference is tiny are dropped: their ratio is dominated by
                rounding and would widen the spread whatever the cause.
                """
                a, b = a.reshape(-1), b.reshape(-1)
                keep = b.abs() > 0.05 * b.abs().max()
                if int(keep.sum()) < 8:
                    return "too few channels above the noise"
                ratio = (a[keep] / b[keep]).float()
                q = torch.quantile(ratio, torch.tensor([0.25, 0.5, 0.75], device=ratio.device))
                lo, mid, hi = (float(x) for x in q)
                spread = (hi - lo) / (abs(mid) + 1e-9)
                return (f"ratio median {mid:+.4f} iqr [{lo:+.4f}, {hi:+.4f}] "
                        f"spread {spread:.3f} over {int(keep.sum())} channels")

            logger.info(
                "afd linear: layer %s call %s -- span against the model %s | control %s | state "
                "|%.5g| conv |%.5g| | %s",
                layer_id, seen[layer_id] - 1, against(mine, theirs), against(shuffled, theirs),
                float(ssm_before.float().norm()), float(conv_before.float().norm()),
                shape_of_the_error(mine, theirs),
            )
        return hook

    installed = 0
    for index, layer in enumerate(model.model.layers):
        attn = getattr(layer, "linear_attn", None)
        if attn is None:
            continue
        attn.register_forward_pre_hook(snapshot(index))
        attn.register_forward_hook(compare(index, attn))
        installed += 1
    logger.info("afd linear: comparing the span against %s linear layer(s) of the model's own "
                "forward", installed)


def make_span_runner(model, *, device):
    """The pool's span runner, if --afd-span-cut asked for one. None otherwise.

    Sized by `max_requests` because a recurrent state cannot be evicted and rebuilt from a prefix
    the way a KV cache can -- it is the whole history compressed -- so the slot table refuses a
    request rather than dropping one, and the refusal has to be far from the working point.
    """
    if not span_cut_wanted():
        return None
    # after the switch, never before it: a pool serving the per-layer cut has no recurrent state
    # to size and must not be refused for a limit it does not need
    max_requests = _span_slots()
    from sglang.srt.afd.linear_state import LinearStates
    from sglang.srt.afd.read_point import layer_types_of
    from sglang.srt.afd.span import SpanRunner, group_layers

    config = model.config
    layer_types = layer_types_of(model)
    states = LinearStates(
        slots=max_requests,
        num_v_heads=config.linear_num_value_heads,
        head_k_dim=config.linear_key_head_dim,
        head_v_dim=config.linear_value_head_dim,
        device=device,
    )
    watch_colocated_residual(model)
    runner = SpanRunner(
        model, states, layer_types=layer_types, query_shift=_span_query_shift())
    watch_linear_attention(model, runner)
    spans = group_layers(layer_types)
    logger.info(
        "afd pool: the group cut. %s span(s) a decode step against %s per-layer calls, %s "
        "layer(s) served here entire, both recurrent states held here for up to %s request(s). "
        "The batch riding a span is fixed for its whole length: every stage in it has "
        "context-free latency, so there is nothing inside a span worth re-forming a batch for.",
        len(spans), len(layer_types) - 1,
        sum(len(s) for s in spans) - len([s for s in spans if s[0] >= 0]),
        max_requests,
    )
    return runner


def make_sweep_service(model, *, enabled, max_context: int, device):
    """The pool's cache and sweep, if --afd-kv-on-pool asked for them. None otherwise."""
    if not enabled:
        return None
    from sglang.srt.afd.pool_attention import KVHolder, SweepService
    from sglang.srt.afd.read_point import layer_types_of

    logger.info(
        "afd pool: holding the KV cache and the key/value projections, up to %s positions a "
        "request. The sweep is answered on the connection thread rather than queued for a "
        "departure: it reads the caller's own cache, so there is no shared weight read for a "
        "departure to amortise.",
        max_context,
    )
    return SweepService(model, KVHolder(device, max_context), layer_types_of(model))


def install_kv_on_pool(model, *, client, mode):
    """Which of the two lines to draw through attention, or neither.

        "cache"       the pool holds the cache and sweeps it; the host joins this step's token
        "projection"  the pool holds W_k and W_v; the host keeps the cache and the whole attention

    A null client is the reversed arrangement, where there is no weights pool for the key and
    value to move to. Checked before `mode`, because a run that asked for one of these lines and
    has nowhere to draw it should say so rather than quietly serving the unconverted stack.
    """
    if client is None:
        if mode:
            raise ValueError(
                f"--afd-pool-attention={mode!r} moves part of attention to a weights pool and "
                f"--afd-pool-addr names none. Either name one or drop the setting; installing "
                f"nothing here would serve the ordinary attention under this flag's name."
            )
        return None
    if not mode:
        return None
    from sglang.srt.afd.read_point import layer_types_of
    from sglang.srt.afd.remote_attention import (
        install_kv_projection,
        install_remote_attention,
    )

    from sglang.srt.afd.pool_client import PoolClient

    types = layer_types_of(model)
    if mode == "cache":
        client.require(PoolClient.NEEDS_CACHE | PoolClient.NEEDS_KV_PROJECTION)
        return install_remote_attention(model, client, types)
    if mode == "projection":
        client.require(PoolClient.NEEDS_KV_PROJECTION)
        return install_kv_projection(model, client, types)
    raise ValueError(f'--afd-pool-attention is "cache" or "projection", got {mode!r}')


def serve_pool_in_background(model, port: int, min_batch: int, max_wait_ms: int, device,
                             attention=None):
    """Answer feed-forward frames on a thread of an already-loaded server.

    The pool role reuses the ordinary launch path: the model is loaded, the scheduler starts, and
    nobody sends it a request. That is wasteful of a process and honest about what it is -- the
    alternative is a second entrypoint whose weight loading, quantisation and device placement
    would be a second implementation of the thing whose numbers must match the host's.
    """
    ready = threading.Event()
    thread = threading.Thread(
        target=run_pool,
        kwargs=dict(model=model, host="0.0.0.0", port=port, min_batch=min_batch,
                    max_wait_ms=max_wait_ms, device=device, ready=ready, attention=attention),
        daemon=True,
        name="afd-pool",
    )
    thread.start()
    if not ready.wait(timeout=60):
        raise RuntimeError(
            f"the afd pool did not bind port {port} within 60s. A host pointed at it would fail "
            f"at its first token instead, an hour into a benchmark."
        )
    return thread


def install_host_routing(model, pool_addr: str, sweep_ahead, connect_timeout_s: float = 30.0):
    """Point every routable layer's feed-forward at the pool, and open the sweep window.

    A host with no weights pool is a real arrangement, not a misconfiguration: the reversed one,
    where the feed-forward stays put and only the sweep travels, to a cache pool named by
    --afd-cache-addr. It was unreachable until now because this function was the only door into
    host mode and it required an address, so the arrangement the budget table ranks first at long
    context had never once been run.

    Returning (None, None) rather than raising is what makes it reachable, and the two callers
    that use the result already treat None as "no pool" -- `install_kv_on_pool` refuses a null
    client, and the router is simply not installed.
    """
    if not pool_addr:
        logger.info(
            "afd host: no weights pool. The feed-forward stays on this machine and only the "
            "sweep travels, which is the reversed arrangement."
        )
        return None, None
    client = PoolClient(pool_addr, connect_timeout_s)
    if span_cut_wanted():
        # asked for by name at the HELLO. The two cuts differ in what each END holds, not only in
        # what they say to each other: a host built for the group cut has no feed-forward weights
        # and no recurrent state, so against a per-layer pool it would not fail, it would run out
        # of layers to ask for. The capability bit is what turns that into a startup error.
        client.require(PoolClient.NEEDS_SPANS)
        return client, install_span_routing(model, client, sweep_ahead=sweep_ahead)
    client.require(PoolClient.NEEDS_FEED_FORWARD)
    return client, install_pool_routing(
        model, client, routable_layers(model), sweep_ahead=sweep_ahead
    )


def install_span_routing(model, client: PoolClient, *, sweep_ahead,
                         reply_timeout_s: float = 60.0):
    """Give whole groups of layers to the pool, keeping only the attentions here.

    `sweep_ahead` is accepted and NOT yet used, and that is deliberate rather than forgotten. The
    window under this cut sits between the two halves of a span's reply -- the pool hands over the
    query's source one feed-forward before the span's output, and the host should spend that
    feed-forward projecting the query and sweeping the cache. It does not yet; it collects both
    halves back to back. `SpanRouting.report()["sweep_window_open"]` says so, so a timing taken
    now cannot be quoted as this arrangement's without the report contradicting it.
    """
    from sglang.srt.afd.history_service import HistoryService
    from sglang.srt.afd.linear_history import HistoryCache
    from sglang.srt.afd.read_point import layer_types_of
    from sglang.srt.afd.span_routing import SpanClient, SpanRouting

    types = layer_types_of(model)
    routing = SpanRouting(
        model, SpanClient(client, reply_timeout_s=reply_timeout_s), types)

    # the history the pool calls back for. It lives here because it is the request's, and
    # because putting it here is what lets the pool stay out of a request's business between
    # that request's calls -- see linear_history.HistoryCache for what the round trips cost.
    config = model.config
    cache = HistoryCache(
        slots=_span_slots(),
        layers=len(types),
        value_heads=config.linear_num_value_heads,
        head_k_dim=config.linear_key_head_dim,
        head_v_dim=config.linear_value_head_dim,
        conv_width=1, conv_taps=1,        # the ring stays on the pool; this holds no convolution
        device=next(model.parameters()).device,
    )
    # the pool sends rows in the order this host sent them, so the ids are the ones the routing
    # is holding for the span in flight. Read through the routing rather than captured, because a
    # captured list would be the FIRST span's rows for every span after it.
    routing.history = HistoryService(cache, rows_of=lambda frame: routing.current_rows())
    client.serve = routing.history
    logger.info(
        "afd host: holding %s linear layer(s) of recurrent state for up to %s request(s), %s",
        sum(1 for t in types if t != "full_attention"), cache.slots, cache.report(),
    )
    logger.info("afd host: the group cut is installed. %s", routing.report())
    if sweep_ahead is not None:
        # it would otherwise fire from the per-layer prepare_mlp hook, on layers that no longer
        # run here at all -- a sweep launched for a feed-forward this host never issues
        sweep_ahead.routed_layers = set(routing.passengers) | set(routing.heads)
    return routing


def run_pool(model, host: str, port: int, min_batch: int, max_wait_ms: int,
             device: torch.device | str, ready: threading.Event | None = None, attention=None):
    """Serve until killed. Blocks."""
    logger.info("afd pool: %s layers, min_batch=%s, max_wait=%sms",
                len(model.model.layers), min_batch, max_wait_ms)
    import os

    departure = serve(
        forward=make_pool_forward(model),
        span=make_span_runner(model, device=device),
        host=host,
        port=port,
        min_batch=min_batch,
        max_wait_s=max_wait_ms / 1000.0,
        device=device,
        ready=ready,
        attention=attention,
    )
    # An operator asks for the riders histogram by naming a path. Without one the pool keeps the
    # record in memory and the health check reports "unknown" rather than guessing at a number
    # that decides whether this arrangement is paying for itself.
    departure.riders_path = os.environ.get("AFD_RIDERS_PATH")
    return departure


class PoolRouting:
    """Replaces the feed-forward of named layers with a call to the pool."""

    def __init__(self, model, client: PoolClient, layers: tuple[int, ...], request_id: int,
                 between: Callable[[int], None] | None = None):
        self.model = model
        self.client = client
        self.layers = layers
        # Frames are keyed by (request_id, layer), so an id reused while a frame is outstanding
        # crosses two answers. A fixed id is safe only while one caller talks to the pool; the
        # counter makes every call of this host unique, and the random base keeps two hosts on
        # one pool from starting at the same number.
        self.request_id = request_id
        base = request_id if request_id else int.from_bytes(os.urandom(4), "big") << 24
        self._ids = itertools.count(base + 1)
        # what runs while the pool works. Called with the layer whose feed-forward is in flight,
        # AFTER the issue: `issue` copies the hidden states to the host and so synchronises the
        # stream, and work launched before it would be waited on by the send.
        self.between = between
        self._undo: list[Callable] = []
        # Skill check 4: whether anything overlapped. Reported from here because the host runs
        # inside the scheduler process, where no caller can reach the client to ask -- and a
        # schedule whose windows all closed looks exactly like one whose windows all opened,
        # except in these two numbers.
        self._calls = 0
        self._local_fallbacks = 0
        self._install()

    def _install(self) -> None:
        for layer_id in self.layers:
            layer = self.model.model.layers[layer_id]
            original = layer.mlp.forward
            layer.mlp.forward = self._route(layer_id, original)
            self._undo.append(
                lambda ly=layer, o=original: setattr(ly.mlp, "forward", o)
            )

    def _route(self, layer_id: int, original: Callable) -> Callable:
        def routed(hidden_states, *args, **kwargs):
            if args or kwargs:
                # a MoE block takes a forward_batch and the pool has no way to carry it
                raise RuntimeError(
                    f"layer {layer_id}'s feed-forward takes more than the hidden states "
                    f"({len(args)} positional, {sorted(kwargs)}); this router only speaks for a "
                    f"dense MLP. Route it locally or teach the wire its other arguments."
                )
            device, dtype = hidden_states.device, hidden_states.dtype
            try:
                handle = self.client.issue(next(self._ids), layer_id, hidden_states)
            except (OSError, PoolClosed) as e:
                return self._without_the_pool(layer_id, original, hidden_states, e)
            if self.between is not None:
                # the window. The next converted layer's query was projected from THIS layer's
                # h_l, so its cache sweep can be launched now and will run on the GPU while the
                # host waits on the socket below.
                self.between(layer_id)
            try:
                out = self.client.collect(handle, device)
            except (OSError, PoolClosed) as e:
                # THIS is where a pool dying lands, and where the first version did not look. A
                # call is outstanding when the process goes away, so the failure arrives at the
                # collect, not the issue -- and an exception inside a forward pass kills sglang's
                # scheduler, so a pool restart took the whole server with it.
                return self._without_the_pool(layer_id, original, hidden_states, e)
            self._calls += 1
            if self._calls % REPORT_EVERY == 0:
                # only the calls since the last line: a cumulative mean carries the pool's JIT
                # warm-up forever, and a schedule that stopped overlapping halfway through would
                # be averaged back into looking fine
                report = self.client.overlap_report(last=REPORT_EVERY)
                logger.info(
                    "afd host: %s pool call(s); last %s: build %.2f ms, send %.2f ms, "
                    "outstanding %.2f ms, blocked %.2f ms, hidden %.1f%%",
                    self._calls,
                    REPORT_EVERY,
                    1e3 * report.get("mean_build_s", 0.0),
                    1e3 * report.get("mean_send_s", 0.0),
                    1e3 * report["mean_outstanding_s"],
                    1e3 * report["mean_blocked_s"],
                    100.0 * (1.0 - report["mean_blocked_s"] / report["mean_outstanding_s"])
                    if report["mean_outstanding_s"] > 0 else 0.0,
                )
            return out.to(dtype)

        return routed

    def _without_the_pool(self, layer_id: int, original, hidden_states, error):
        """Serve this layer locally because the pool is gone, and say so.

        The alternative is what the first version did: re-raise, and let an exception inside a
        forward pass kill the scheduler. A pool restart then stops a server that was carrying
        hundreds of requests, which is a worse outcome than serving them from weights that were
        supposed to be elsewhere.

        So the host computes it, and the degradation is COUNTED and logged rather than silent. The
        host has these weights -- it loads the whole stack -- so the answer is right; what is lost
        is the arrangement, and a run that fell back for half its layers must not be able to
        report the arrangement's throughput under the arrangement's name.
        """
        self._local_fallbacks += 1
        if self._local_fallbacks == 1 or self._local_fallbacks % 512 == 0:
            logger.warning(
                "afd host: the pool at %s is unreachable (%s). Running layer %s locally; %s "
                "layer(s) so far have been served without it. The answers are right and the "
                "arrangement is not in effect -- any timing taken from here is a colocated timing.",
                self.client.address, type(error).__name__, layer_id, self._local_fallbacks,
            )
        self.client.reconnect()
        out = original(hidden_states)
        return out[0] if isinstance(out, tuple) else out

    def remove(self) -> None:
        for fn in self._undo:
            fn()
        self._undo.clear()


def install_pool_routing(model, client: PoolClient, layers: tuple[int, ...],
                         request_id: int = 0, sweep_ahead=None) -> PoolRouting:
    if not layers:
        raise ValueError(
            "no layer routed to the pool. A host that runs every feed-forward itself is the "
            "colocated arrangement with a socket open beside it."
        )
    between = None
    if sweep_ahead is not None:
        # the schedule takes over the trigger for the layers it routes; without this handover the
        # sweep would fire from the prepare_mlp wrapper, BEFORE the issue, and the send would wait
        # on it -- a correct model with the window closed, and a benchmark that reads as "the
        # overlap does not help".
        sweep_ahead.routed_layers = set(layers)
        between = sweep_ahead.sweep_after_issue
    logger.info(
        "afd host: routing %s layer(s) to the pool at %s, %s of them opening a sweep window",
        len(layers), client.address,
        len(set(layers) & set(sweep_ahead.sweep_at)) if sweep_ahead is not None else 0,
    )
    return PoolRouting(model, client, tuple(layers), request_id, between=between)
