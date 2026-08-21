"""Does the deployed arrangement's signature -- output approximately equals input -- appear on a
small Qwen3.5 stack, in one process, with no sockets and no second machine?

Measured, not argued: the span is run twice on different embeddings, and two ratios are printed.

    ||x - e|| / ||e||        how far the span moved its input. A near-identity is small
    ||x1-x2|| / ||e1-e2||    whether it moved at all as a function of the input

The reference is the span against ITSELF at two inputs, so no hand-written model is involved and
nothing here can agree with a fake.
"""
import sys, threading
sys.path.insert(0, "test/registered/unit")
import torch
from afd_tiny_stack import build_tiny_stack

stack, config, kinds = build_tiny_stack(device="cuda")
torch.manual_seed(0)

from sglang.srt.afd.linear_state import LinearStates
from sglang.srt.afd.span import SpanRunner, group_layers
from sglang.srt.afd.split_read_kernel import read_one, update_only

states = LinearStates(
    slots=4, num_v_heads=config.linear_num_value_heads,
    head_k_dim=config.linear_key_head_dim, head_v_dim=config.linear_value_head_dim,
    device=torch.device("cuda"))
runner = SpanRunner(stack, states, layer_types=kinds, query_shift=1)
print("spans:", group_layers(kinds))

# the host's side of a linear layer, in this process: one contraction of a state this end holds,
# and the deferred advance. Same functions the real HistoryService calls.
LAYERS = len(stack.layers)
VH, DK, DV = config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim
state = torch.zeros(LAYERS, 4, VH, DV, DK, device="cuda")

def ask_host(layer_id, request_ids, q_tilde, step=None):
    slots = torch.tensor([0] * q_tilde.shape[0], device="cuda")
    return read_one(state[layer_id], slots, q_tilde).float()

def defer_update(layer_id, request_ids, k, v, alpha, beta):
    slots = torch.tensor([0] * k.shape[0], device="cuda")
    update_only(state[layer_id], slots, k=k, v=v, alpha=alpha, beta=beta)

runner._local.ask_host = ask_host
runner._local.defer_update = defer_update

rows = 1
positions = torch.zeros(3, rows, dtype=torch.long, device="cuda")
H = config.hidden_size

def prologue(e, rid):
    state.zero_()
    runner._residual.clear(); runner._gate.clear()
    runner.run_prologue([rid], e, positions)
    return runner._residual[rid].clone()

VH_, DK_ = config.num_attention_heads, config.head_dim

def whole_chain(e, rid):
    """Every stage the pool runs, in order, with the host's attention output stood in for.

    The host's half is a random tensor of the right shape rather than real attention: what is
    being asked here is whether the POOL's chain carries a contribution into the residual, and a
    stage that dropped it would drop it whatever the attention returned.
    """
    state.zero_()
    runner._residual.clear(); runner._gate.clear()
    runner.run_prologue([rid], e, positions)
    after_prologue = runner._residual[rid].clone()
    attn_out = torch.randn(rows, VH_ * DK_, device="cuda", dtype=torch.bfloat16) * 0.05
    runner.run(spans[1][0], [rid], attn_out, positions) if False else runner.run(
        [rid], spans[1][0], attn_out, positions)
    after_middle = runner._residual[rid].clone()
    final = runner.run_epilogue([rid], spans[2][0], attn_out)
    return after_prologue, after_middle, final

spans = group_layers(kinds)
e0 = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16)
a, b, f = whole_chain(e0, 33)
ef = e0.float()
print("--- the pool's whole chain, each stage against the embedding it started from ---")
for name, t in (("after prologue", a), ("after middle span", b), ("final (epilogue)", f)):
    tf = t.float()
    moved = (tf - ef).norm() / ef.norm()
    cos = torch.nn.functional.cosine_similarity(tf.flatten(), ef.flatten(), dim=0)
    print(f"  {name:20s} ||x-e||/||e|| = {moved:9.6f}   cos = {cos:9.6f}")

e1 = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16)
e2 = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16)
x1, x2 = prologue(e1, 11), prologue(e2, 22)

x1, x2, e1f, e2f = x1.float(), x2.float(), e1.float(), e2.float()
moved = (x1 - e1f).norm() / e1f.norm()
sensitivity = (x1 - x2).norm() / (e1f - e2f).norm()
cos = torch.nn.functional.cosine_similarity(x1.flatten(), e1f.flatten(), dim=0)
print(f"||x-e||/||e||        = {moved:.6f}")
print(f"||x1-x2||/||e1-e2||  = {sensitivity:.6f}")
print(f"cos(x, e)            = {cos:.6f}")
