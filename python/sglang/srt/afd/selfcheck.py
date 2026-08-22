"""The instruments built to find why the group cut serves wrong text, and nothing else.

573 lines of this lived in `roles.py`, which is the arrangement's composition root and the file
upstream would read first. None of it is the arrangement: it runs the span's own linear attention
beside the model's inside one process, seeds it from the model's own state, and reports where they
part -- work that exists because five real bugs were fixed without any of them being the cause.

Kept, not deleted, and kept together. Ten separate times in that search a comparison turned out to
measure something other than what it claimed, and every one of those corrections is written into
the code here rather than into a commit message that nobody will read again. Whoever needs to
compare an arm against the model next will reach for this file, and the traps it names are the
ones they would otherwise walk into: a hook registered permanently and fired again by the control
that follows it; a pre-hook gated on one row when the failure is on a hundred and twenty-two; the
model's `mixed_qkv` being pre-convolution while this side's is post; a per-head decomposition read
as though it were the layer's own error.

All of it is gated on SGLANG_AFD_SELFCHECK and does nothing without it.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


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
            from sglang.srt.afd.span import _dump

            _dump("model", f"pre-mlp {index}", residual)
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

    entering_seen = {}

    def watch_entering(index):
        """The residual stream ENTERING a layer: residual + hidden_states, before prepare_attn.

        Not the same as the input to its linear attention, which is that stream normalised --
        RMSNorm removes the scale, so two streams differing in magnitude give the same normalised
        input, and layer 1's normalised input matching to 0.03% says nothing about the stream that
        produced it. This is the quantity the span's `post-mlp` trace of the previous layer can be
        set beside.
        """
        def hook(_module, args, kwargs):
            hidden = kwargs.get("hidden_states")
            residual = kwargs.get("residual")
            if hidden is None:
                return
            key = (index, hidden.shape[0])
            if entering_seen.get(key, 0) >= 1:
                return
            entering_seen[key] = 1
            stream = hidden if residual is None else hidden + residual
            from sglang.srt.afd.span import _dump

            _dump("model", f"entering {index} rows {stream.shape[0]}", stream)
            row = stream[0].float()
            logger.info("afd colocated entering: layer %s rows %s -- |%.5g| rms %.5g",
                        index, stream.shape[0], float(row.norm()),
                        float(row.pow(2).mean().sqrt()))
        return hook

    def watch_mlp_in(index):
        """The feed-forward's INPUT: post_attention_layernorm's output.

        Between layer 0's x + attn, which is exact, and layer 1's x + attn, which is 6.3% high,
        sits this and the feed-forward itself. The feed-forward's OUTPUT was compared and agrees
        to 0.6%; its input never was, and a normalisation between two agreeing quantities can
        still differ if it is fed or weighted differently.
        """
        def hook(_module, args):
            key = ("in", index, args[0].shape[0])
            if entering_seen.get(key, 0) >= 1:
                return
            entering_seen[key] = 1
            row = args[0][0].float()
            logger.info("afd colocated mlp-in: layer %s rows %s -- |%.5g| rms %.5g",
                        index, args[0].shape[0], float(row.norm()),
                        float(row.pow(2).mean().sqrt()))
        return hook

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
        layer.register_forward_pre_hook(watch_entering(index), with_kwargs=True)
        if hasattr(layer, "mlp"):
            layer.mlp.register_forward_hook(watch_mlp(index))
            layer.mlp.register_forward_pre_hook(watch_mlp_in(index))
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
    grabbed = {}

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
            if hidden.dim() != 2:
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
                # ONCE, and then removed. Registered permanently, this hook kept firing -- and
                # `compare` calls `span_of` twice, the second time on a SHUFFLED input for its
                # control. So the model's value was overwritten by the control's, and the two
                # "pre" tensors compared at 100% apart while the layer's outputs agreed to 0.5%.
                # A linear projection cannot do that, and the contradiction is what gave it away.
                inner = getattr(_module, "attn", None)
                if inner is not None and ("mixed", layer_id) not in before:
                    def grab_mixed(_m, args, kwargs, _lid=layer_id):
                        if ("mixed", _lid) in before:
                            return
                        got = kwargs.get("mixed_qkv")
                        if got is None:
                            return
                        before[("mixed", _lid)] = got.detach()
                        grabbed[("mixed", _lid)].remove()
                    grabbed[("mixed", layer_id)] = inner.register_forward_pre_hook(
                        grab_mixed, with_kwargs=True)

                attn_mod = getattr(_module, "out_proj", None)
                if attn_mod is not None and ("pre", layer_id) not in before:
                    def grab(_m, args, _lid=layer_id):
                        if ("pre", _lid) in before:
                            return
                        before[("pre", _lid)] = args[0].detach()
                        grabbed[_lid].remove()
                    grabbed[layer_id] = attn_mod.register_forward_pre_hook(grab)
                if layer_id < 2:
                    logger.info(
                        "afd colocated dtypes: layer %s -- ssm %s conv %s hidden %s",
                        layer_id, cache.temporal.dtype, cache.conv[0].dtype, hidden.dtype)

            except Exception as e:                       # noqa: BLE001 -- diagnostic, reported
                before[layer_id] = None
                logger.info("afd linear: layer %s state not readable: %r", layer_id, e)
        return hook

    attn_in_seen = {}

    def watch_attn_in(layer_id):
        """The model's own input to a linear attention, on ANY call, with its row count.

        Separate from the snapshot hook, which is gated on a single row. That gate put this line
        on decode steps while the span logs its own input on a multi-row first prefill, so the two
        sets were 1-row against 122-row -- the fifth time in this search that two correct
        measurements of different occasions were about to be compared, and the first caught before
        it produced a claim. The row count is printed so a reader can only pair like with like.
        """
        def hook(_module, args):
            key = (layer_id, args[0].shape[0])
            if attn_in_seen.get(key, 0) >= 1:
                return
            attn_in_seen[key] = 1
            row = args[0][0].float()
            logger.info("afd colocated attn-in: layer %s rows %s -- |%.5g| rms %.5g",
                        layer_id, args[0].shape[0], float(row.norm()),
                        float(row.pow(2).mean().sqrt()))
        return hook

    def compare(layer_id, attn):
        def hook(_module, args, output):
            if seen.get(layer_id, 0) >= limit:
                return
            hidden = args[0]
            if hidden.dim() != 2:
                return
            # MULTI-ROW too. This was gated to one row, which is a decode step -- and every
            # comparison it has produced ran the decode path: `_convolve`'s scatter branch and
            # OP_STATE_READ. The deployment fails on a 122-row PREFILL, which takes
            # `prefill_convolve` and OP_STATE_SCAN instead, and those two have never been inside
            # this comparison at all. "The linear attention is correct" was true of the half of it
            # that was measured.
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
                    """What the HOST does, including the part that makes a chunk a chunk.

                    A batched `read_one` over every row contracts them all against one unchanging
                    state -- right for a decode batch, where no two rows share a slot, and wrong
                    for a prefill chunk, whose rows are one request's consecutive tokens and each
                    reads what its predecessor wrote. That is `HistoryService._scan`, and this stub
                    did not have it.

                    Without it this comparison reported 42 of 48 value heads within 2% on a prefill
                    against 48 of 48 on a decode, and the shortfall landed on one key head's group
                    at a time -- a structure, chased for a round, produced entirely by the
                    instrument. The eighth time in this search that a measurement measured itself.
                    """
                    state_buf = runner.states.buffer(lid)
                    rows = q_tilde.shape[0]
                    slots = torch.full((rows,), slot, device=q_tilde.device, dtype=torch.long)
                    if rows == 1 or step is None:
                        logger.info("afd stub: layer %s batched read of %s row(s), step=%s",
                                    lid, rows, step is not None)
                        return read_one(state_buf, slots, q_tilde).float()
                    logger.info("afd stub: layer %s scanning %s rows sequentially", lid, rows)
                    k_s, v_s, alpha_s, beta_s = step
                    out = []
                    for t in range(rows):
                        one = slots[t : t + 1]
                        out.append(read_one(state_buf, one, q_tilde[t : t + 1]))
                        update_only(state_buf, one, k=k_s[t : t + 1], v=v_s[t : t + 1],
                                    alpha=alpha_s[t : t + 1], beta=beta_s[t : t + 1])
                    return torch.cat(out, dim=0).float()

                def defer_update(lid, request_ids, k, v, alpha, beta):
                    if k.shape[0] != 1:
                        return          # a multi-row rider advanced inside the scan already
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

                def span_of(x, reverse_ring=False):
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
                    kept = history[..., -(taps - 1):]
                    # The control for this seeding. sglang's conv_state holds the last K-1 inputs
                    # and this side's ring holds K with the newest LAST, so the K-1 go into
                    # columns 1..K-1. That is an argument, not a measurement -- and it only
                    # affects the COMPARISON: in deployment the pool builds its ring from its own
                    # tokens and never seeds from the model. So a wrong seed here would produce
                    # the 1-5% spread with nothing wrong in the arrangement at all.
                    ring[slot][..., 1:] = kept.flip(-1) if reverse_ring else kept
                    return runner._linear_attention(
                        attn, [scratch] * x.shape[0], layer_id, x).float()

                # `out_proj`'s INPUT is where the heads still exist: (rows, value heads x head
                # dim). Its OUTPUT is hidden_size wide and has no head structure at all -- the
                # first version of this decomposition reshaped the output to (-1, 48, 106) and the
                # pool refused to start, which is the shape saying so.
                # a temporary PRE-HOOK, not a reassignment: `attn.out_proj` is an nn.Module and
                # binding a plain function to that name raises "cannot assign ... as child
                # module". The model's own call has already returned by the time this runs -- this
                # is a forward hook -- so the only `out_proj` call inside `span_of` is the span's.
                caught = {}

                def capture(_m, args):
                    caught["pre"] = args[0].detach()

                handle = attn.out_proj.register_forward_pre_hook(capture)
                try:
                    mine = span_of(hidden)
                    reversed_ring = span_of(hidden, reverse_ring=True)
                finally:
                    handle.remove()
                mine_pre = caught.get("pre")
                caught["alpha"] = getattr(local, "last_alpha", None)
                caught["mixed"] = getattr(local, "last_packed", None)
                caught["after"] = getattr(local, "last_mixed", None)
                theirs = output.float()
                order = torch.randperm(hidden.shape[1], device=hidden.device)
                shuffled = span_of(hidden[:, order])
            except Exception as e:                       # noqa: BLE001 -- diagnostic, reported
                logger.info("afd linear: layer %s could not be compared: %r", layer_id, e)
                return
            finally:
                runner.release(scratch)

            def per_head(a, b, heads, layer_id=layer_id):
                """Where the difference lives, head by head.

                The output is `heads` blocks of `head_v_dim` side by side. A reimplementation that
                is merely imprecise is wrong a little everywhere; one that pairs a query with the
                wrong key, or expands the key heads across the value heads in the wrong order, is
                exactly right on some heads and exactly wrong on others. Norms cannot tell those
                apart and this can.
                """
                a = a.reshape(-1, heads, a.shape[-1] // heads).float()
                b = b.reshape(-1, heads, b.shape[-1] // heads).float()
                err = (a - b).norm(dim=-1) / (b.norm(dim=-1) + 1e-9)
                # ROWS 0, middle and last, not row 0 alone. Row 0 reads the initial state and
                # nothing else, so every prefill reading this comparison produced until now
                # described the chunk's FIRST TOKEN -- and the recurrence had not run yet. A stub
                # rewritten to scan token by token changed not one digit, which is what exposed it.
                rows = err.shape[0]
                for tag, r in (("row0", 0), ("mid", rows // 2), ("last", rows - 1)):
                    e = err[r]
                    top = torch.topk(e, min(3, e.numel()))
                    logger.info(
                        "afd rows: layer %s %s (of %s) -- %s/%s heads within 2%%, worst %s",
                        layer_id, tag, rows, int((e < 0.02).sum()), heads,
                        " ".join(f"h{int(i)}={float(v):.3f}"
                                 for v, i in zip(top.values, top.indices)),
                    )
                err = err[0]
                good = int((err < 0.02).sum())
                worst = torch.topk(err, min(4, err.numel()))
                line = (f"{good}/{heads} heads within 2%, worst "
                        + " ".join(f"h{int(i)}={float(v):.3f}"
                                   for v, i in zip(worst.values, worst.indices)))
                # Is the ordering a STRUCTURE or an ACCUMULATION? A prefill scan walks the chunk
                # token by token, so a head whose decay is closest to 1 carries its rounding
                # furthest -- and "the worst heads are the slowest-decaying heads" is a completely
                # different finding from "one key head's channels are mis-sliced". The per-head
                # decay and its rank correlation with the error separate them, and neither can be
                # read off the error alone.
                decay = caught.get("alpha")
                if decay is not None and decay.numel() == heads:
                    a = decay.float()
                    ranks = lambda t: t.argsort().argsort().float()
                    rho = float(torch.corrcoef(torch.stack([ranks(err), ranks(a)]))[0, 1])
                    line += (f" | alpha of those: "
                             + " ".join(f"{float(a[int(i)]):.4f}" for i in worst.indices)
                             + f" | alpha mean {float(a.mean()):.4f} max {float(a.max()):.4f}"
                             + f" | rank corr(err, alpha) {rho:+.3f}")
                return line

            def elementwise(a, b):
                d = (a.reshape(-1) - b.reshape(-1)).abs()
                scale = b.reshape(-1).abs() + 1e-6
                r = d / scale
                q = torch.quantile(r.float(), torch.tensor([0.5, 0.9, 0.99], device=r.device))
                return (f"elementwise |d|/|b| median {float(q[0]):.4g} p90 {float(q[1]):.4g} "
                        f"p99 {float(q[2]):.4g} max {float(r.max()):.4g}")

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

            heads = attn.num_v_heads // attn.attn_tp_size
            mine_mixed, theirs_mixed = caught.get("mixed"), before.get(("mixed", layer_id))
            if mine_mixed is not None and theirs_mixed is not None:
                kh = attn.num_k_heads // attn.attn_tp_size
                dk = attn.head_k_dim
                width = kh * dk

                def by_key_head(name, lo, hi):
                    """The packed projection's q and k blocks, sliced the way the heads are.

                    If key head 10's channels are already wrong HERE then the convolution put them
                    wrong; if they are right here and the output is wrong, the scan downstream did.
                    One boundary, and it separates the only two paths a prefill takes that a decode
                    does not.
                    """
                    a = mine_mixed[:, lo:hi].reshape(-1, kh, dk).float()
                    b = theirs_mixed.reshape(theirs_mixed.shape[0], -1)[:, lo:hi]
                    b = b.reshape(-1, kh, dk).float()
                    e = ((a - b).norm(dim=-1) / (b.norm(dim=-1) + 1e-9))[0]
                    top = torch.topk(e, min(3, e.numel()))
                    return (f"{name}: {int((e < 0.02).sum())}/{kh} key heads within 2%, worst "
                            + " ".join(f"k{int(i)}={float(v):.3f}"
                                       for v, i in zip(top.values, top.indices)))

                logger.info("afd packed (pre-conv): layer %s -- %s | %s", layer_id,
                            by_key_head("q", 0, width), by_key_head("k", width, 2 * width))

                # The pre-convolution packing is identical on both sides, so the next boundary is
                # the convolution's OUTPUT. The model's post-convolution tensor is a local inside
                # the backend and cannot be hooked, but it does not need to be: run sglang's own
                # `causal_conv1d_fn` on the packing both sides agree on, with an empty ring, and
                # that IS the model's path. The span's is `last_mixed`.
                after = caught.get("after")
                if after is not None:
                    try:
                        from sglang.srt.layers.attention.mamba.causal_conv1d import (
                            causal_conv1d_fn,
                        )

                        w = attn.conv1d.weight
                        want = causal_conv1d_fn(
                            theirs_mixed.reshape(theirs_mixed.shape[0], -1).t()
                            .unsqueeze(0).contiguous(),
                            w.view(w.shape[0], w.shape[2]), attn.conv1d.bias,
                            activation=attn.activation)[0].t()
                        mine_mixed2 = after.reshape(after.shape[0], -1)
                        a = mine_mixed2[:, width:2 * width].reshape(-1, kh, dk).float()
                        b = want[:, width:2 * width].reshape(-1, kh, dk).float()
                        e = ((a - b).norm(dim=-1) / (b.norm(dim=-1) + 1e-9))[0]
                        top = torch.topk(e, min(3, e.numel()))
                        logger.info(
                            "afd conv-out: layer %s -- k block %s/%s key heads within 2%%, worst %s",
                            layer_id, int((e < 0.02).sum()), kh,
                            " ".join(f"k{int(i)}={float(v):.3f}"
                                     for v, i in zip(top.values, top.indices)))
                    except Exception as e:                # noqa: BLE001 -- diagnostic, reported
                        logger.info("afd conv-out: layer %s not comparable: %r", layer_id, e)

            theirs_pre = before.get(("pre", layer_id))
            if mine_pre is not None and theirs_pre is not None:
                logger.info("afd linear split: layer %s -- %s | %s", layer_id,
                            per_head(mine_pre, theirs_pre, heads),
                            elementwise(mine_pre, theirs_pre))
            logger.info(
                "afd ring control: layer %s -- as seeded %s | history REVERSED %s",
                layer_id, against(mine, theirs), against(reversed_ring, theirs))
            logger.info("afd linear whole: layer %s -- %s",
                        layer_id, elementwise(mine, theirs))
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
        attn.register_forward_pre_hook(watch_attn_in(index))
        attn.register_forward_hook(compare(index, attn))
        installed += 1
    logger.info("afd linear: comparing the span against %s linear layer(s) of the model's own "
                "forward", installed)
