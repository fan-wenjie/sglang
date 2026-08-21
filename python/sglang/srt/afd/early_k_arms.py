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


def install_arm(model, arm: str):
    """Replace every linear layer's recurrence with this arm's. Returns an undo callable."""
    if arm not in ARMS:
        raise ValueError(f"{arm!r} is not one of {ARMS}")
    undo = []
    for layer in model.model.layers:
        attn = getattr(layer, "linear_attn", None)
        if attn is None:
            continue
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
