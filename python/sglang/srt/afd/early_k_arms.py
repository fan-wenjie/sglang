"""Three arms for the pre-registered question of which key may be early.

`benchmark/afd/EARLY_K_PREREGISTRATION.md` states the conjecture and fixes the decision rule. This
is the surgery the three arms need, kept as small as it can be:

  * driven one token at a time, so the model takes its own RECURRENT path -- the chunked prefill
    kernel does not contain the step the approximation is defined on
  * only `recurrent_gated_delta_rule` is replaced, not the layer's forward. Everything else --
    the projections, the convolution, the gates, the norm -- stays the model's own
  * the `exact` arm has to reproduce the stock function before any arm is reported. Otherwise the
    approximation and this reimplementation's own error land in the same difference

## The operator, and where each arm evaluates it

One operator appears twice in a linear layer:

    P(k) = I - beta k k^T

    state    S_t = alpha S_(t-1) P(k) + beta v k^T      survives the step
    output   o_t = alpha S_(t-1) P(k) q + beta (k.q) v  does not

    exact       P current in both
    mixed       P early in the output, P current in the state       the conjecture
    all_early   P early in both                                      the control

## The convolution, which is the part that is easy to get wrong

The convolution runs BEFORE the split into query, key and value and its ring holds three steps, so
it is itself a time-reused state. Under the conjecture the early key must be convolved WITHOUT
writing to that ring; the current key does the write. In `all_early` the early key MUST write it,
or the two arms differ in one place instead of two and the control is not a control.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

ARMS = ("exact", "mixed", "all_early")


def _recurrent(query, key, value, g, beta, initial_state, output_final_state,
               use_qk_l2norm_in_kernel=False, key_for_output=None, beta_for_output=None):
    """The gated delta rule, one step, written out so the operator can be evaluated twice.

    `key_for_output` and `beta_for_output` are what the OUTPUT's copy of P(k) uses. When they are
    None both copies use the state's key, which is the model.

    Shapes follow the stock function: (batch, seq, heads, dim) with seq == 1, and the state is
    (batch, heads, KEY dim, VALUE dim).

    That layout is transposed from sglang's, whose own comment says its buffer is
    (slots, value heads, head_v_dim, head_k_dim) "to match what the decode kernel expects". Both
    dimensions are 128 on this model, so nothing about the shape says which is which, and writing
    this against the wrong one produced a state that was 0.96 out and logits that disagreed on
    eight positions in thirteen -- caught only because the pre-registration required the exact arm
    to reproduce the stock function before any arm was reported.
    """
    q, k, v = query.float(), key.float(), value.float()
    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
    q = q * (q.shape[-1] ** -0.5)
    alpha = g.float().exp()                                   # g is already -exp(A_log) softplus(.)
    b = beta.float()
    S = initial_state.float()                                 # (batch, heads, k dim, v dim)

    q0, k0, v0 = q[:, 0], k[:, 0], v[:, 0]                    # (batch, heads, dim)
    a0, b0 = alpha[:, 0].unsqueeze(-1), b[:, 0].unsqueeze(-1)

    ko = k0 if key_for_output is None else key_for_output
    bo = b0 if beta_for_output is None else beta_for_output.unsqueeze(-1)
    if use_qk_l2norm_in_kernel and key_for_output is not None:
        ko = F.normalize(ko.float(), dim=-1)

    # the output's copy of P(k): q~ = q - beta (k.q) k
    kq = (ko * q0).sum(-1, keepdim=True)
    q_tilde = q0 - bo * kq * ko
    out = a0 * torch.einsum("bhkv,bhk->bhv", S, q_tilde) + bo * kq * v0

    # the state's copy, always with the key the caller passed as `key`
    h_k = torch.einsum("bhkv,bhk->bhv", S, k0)
    S_new = a0.unsqueeze(-1) * S + torch.einsum("bhk,bhv->bhkv", k0, b0 * (v0 - a0 * h_k))

    out = out.unsqueeze(1).to(value.dtype)                    # (batch, 1, heads, v dim)
    return out, (S_new.to(initial_state.dtype) if output_final_state else initial_state)


def _convolve(attn, x, ring, *, write: bool):
    """One step of the depthwise causal convolution, optionally without touching the ring.

    `ring` is (batch, channels, kernel) holding the last steps. The read-only form is what the
    conjecture requires of the early key: the ring is a state reused across three steps, so a key
    written into it would survive, and the whole claim is that only per-step values may be early.

    In the `all_early` arm the early key MUST write, or the two arms differ in one place instead
    of two and the control is not a control.
    """
    window = torch.cat([ring[..., 1:], x.unsqueeze(-1)], dim=-1)
    out = (window * attn.conv1d.weight.squeeze(1)).sum(-1)
    if attn.conv1d.bias is not None:
        out = out + attn.conv1d.bias
    if write:
        ring.copy_(window)
    return F.silu(out)


def _early_key_and_beta(layer, attn, h_prev, *, write_ring):
    """Project a key and a write strength from the SHIFTED residual, as the read point does.

    Runs the layer's own input norm and input projections against `h_(l-1)` instead of `x_l`, then
    the same convolution -- a key that skipped the convolution would be early AND unfiltered,
    which is two changes wearing one name.

    Only the KEY's channels are convolved. `in_proj_qkv` emits query, key and value concatenated
    and the convolution is depthwise, so each channel is filtered independently and a slice of it
    is a legitimate thing to take on its own.
    """
    ring = attn._afd_ring
    if ring is None:
        return None
    if ring.dim() != 3:
        raise RuntimeError(
            f"the convolution ring is {tuple(ring.shape)} where (batch, channels, taps) was "
            f"expected. Convolving the early key against the wrong layout would filter it with a "
            f"neighbour's history, and the arms would then differ by that rather than by the key."
        )
    normed = layer.input_layernorm(h_prev)
    mixed = attn.in_proj_qkv(normed)[:, -1]
    b = attn.in_proj_b(normed)[:, -1]
    convolved = _convolve(attn, mixed, ring, write=write_ring)
    width = attn.key_dim
    k_conv = convolved[:, width : 2 * width].reshape(-1, attn.num_k_heads, attn.head_k_dim)
    rep = attn.num_v_heads // attn.num_k_heads
    if rep > 1:
        k_conv = k_conv.repeat_interleave(rep, dim=1)
    return k_conv, b.sigmoid()


def install_arm(model, arm: str):
    """Replace every linear layer's recurrence with this arm's. Returns an undo callable."""
    if arm not in ARMS:
        raise ValueError(f"{arm!r} is not one of {ARMS}")
    undo = []
    stash = {}

    for index, layer in enumerate(model.model.layers):
        # h_l is the input to the post-attention norm: the residual after this layer's attention
        # and before its feed-forward, which is the value the read point reads
        def keep(_m, args, _i=index):
            stash[_i] = args[0].detach()

        handle = layer.post_attention_layernorm.register_forward_pre_hook(keep)
        undo.append(handle.remove)

    for index, layer in enumerate(model.model.layers):
        attn = getattr(layer, "linear_attn", None)
        if attn is None:
            continue
        attn._afd_ring = None

        def before(_m, args, kwargs, _a=attn, _l=layer, _i=index):
            _a._afd_early = None
            if _a._afd_arm == "exact":
                return None
            h_prev = stash.get(_i - 1)
            if h_prev is None:                    # the first layer has nothing beneath it
                return None
            cache = kwargs.get("cache_params")
            if cache is None or not cache.has_previous_state(_a.layer_idx):
                return None                       # the first token has no ring to convolve against
            _a._afd_ring = cache.layers[_a.layer_idx].conv_states
            _a._afd_early = _early_key_and_beta(
                _l, _a, h_prev, write_ring=(_a._afd_arm == "all_early"))
            return None

        handle = attn.register_forward_pre_hook(before, with_kwargs=True)
        undo.append(handle.remove)
        original = attn.recurrent_gated_delta_rule
        attn._afd_arm = arm
        attn._afd_early = None                                # set by the hook, per step

        def patched(query, key, value, g, beta, initial_state, output_final_state,
                    use_qk_l2norm_in_kernel=False, _a=attn, **kwargs):
            early = _a._afd_early if _a._afd_arm != "exact" else None
            k_out = early[0] if early is not None else None
            b_out = early[1] if early is not None else None
            return _recurrent(
                query, key, value, g, beta, initial_state, output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                key_for_output=k_out, beta_for_output=b_out,
            )

        attn.recurrent_gated_delta_rule = patched
        undo.append(lambda a=attn, o=original: setattr(a, "recurrent_gated_delta_rule", o))

    def remove():
        for fn in undo:
            fn()

    return remove
