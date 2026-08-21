"""How much is there to hide, per layer, before anything is built to hide it.

    python -m sglang.srt.afd.hideable

Measured on Qwen3.8-27B-FP8 at batch 4, against a 6500 us pool round trip:

    softmax sweep, context   256      37 us      launch-bound; the cache is too small to matter
    softmax sweep, context  4096      42 us      still launch-bound
    softmax sweep, context 16384     278 us      the linear regime starts here
    softmax sweep, context 65536    1074 us      4% of the round trip, over 16 layers
    linear q^T S_(t-1)                21 us      and the update reads the same 12.6 MB again

This is why the linear layers carry a projection window and not a partitioned recurrence. The
recurrence's cost IS the state read, both halves need it, and splitting reads it twice to hide one
of the two reads. The projection is a different matter: a converted layer already runs it twice,
once per stream, so moving the early one into the window costs nothing and hides real work.

It is also why the arrangement does not pay here yet. A whole layer's attention is tens of
microseconds and a pool call is milliseconds -- the round trip is two orders of magnitude larger
than the work it is meant to overlap, and no schedule fixes that ratio. The interconnect is
0.3 ms; the other 6.2 ms is device-to-host, host-to-device, and Python.


A window is worth opening only if the half that can run early takes long enough to matter against
a pool round trip. That is arithmetic on this model's shapes, and it is cheaper to measure than to
discover after writing a kernel.

    softmax layer   the sweep reads the KV cache: 4 kv heads x 256 dim x L positions. Linear in
                    the context, which is the arrangement's premise
    linear layer    the early half is q^T S_(t-1): the recurrent state is 48 x 128 x 128 per
                    request and does not grow with context at all. The state is read by BOTH
                    halves -- the update needs S_(t-1) too -- so splitting reads it twice
"""
import sys

import torch

BS = 4
CONTEXTS = [256, 1024, 4096, 16384, 65536]
Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
V_HEADS, K_DIM, V_DIM = 48, 128, 128


def timed(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3          # microseconds


def main():
    sys.path.insert(0, "/home/user/experiment/sglang/python")
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

    dev = "cuda"
    print(f"  batch {BS}\n", flush=True)
    print("  softmax layer, the swept cache:", flush=True)
    for length in CONTEXTS:
        k_buffer = torch.randn(BS * length + 1, KV_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        v_buffer = torch.randn(BS * length + 1, KV_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        q = torch.randn(BS, Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        o = torch.empty(BS, Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        kv_indptr = torch.arange(BS + 1, device=dev, dtype=torch.int64) * length
        kv_indices = torch.arange(BS * length, device=dev, dtype=torch.int64)
        splits = 8
        logits = torch.empty(BS, Q_HEADS, splits, HEAD_DIM, device=dev, dtype=torch.float32)
        lse = torch.empty(BS, Q_HEADS, splits, device=dev, dtype=torch.float32)
        num_splits = torch.full((BS,), splits, device=dev, dtype=torch.int32)
        us = timed(lambda: decode_attention_fwd(
            q, k_buffer, v_buffer, o, kv_indptr, kv_indices, logits, lse,
            num_splits, splits, HEAD_DIM**-0.5, 1.0, 1.0))
        print(f"    context {length:6d}: sweep {us:8.1f} us", flush=True)
        del k_buffer, v_buffer, logits
        torch.cuda.empty_cache()

    print("\n  linear layer, the recurrent state:", flush=True)
    state = torch.randn(BS, V_HEADS, K_DIM, V_DIM, device=dev, dtype=torch.float32)
    q = torch.randn(BS, V_HEADS, K_DIM, device=dev, dtype=torch.float32)
    bytes_read = state.numel() * state.element_size()
    us = timed(lambda: torch.einsum("bhk,bhkv->bhv", q, state))
    print(f"    state {tuple(state.shape)} = {bytes_read/1e6:.1f} MB", flush=True)
    print(f"    q^T S_(t-1) (the early half): {us:8.1f} us  "
          f"= {bytes_read/1e6/us*1e6/1e9:.0f} GB/s, so it is the state read", flush=True)
    print(f"    the update reads the same state again: another {us:.1f} us that stays exposed",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
