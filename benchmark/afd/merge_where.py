"""Should the join and the merge run on the GPU or on the CPU?

    python -m sglang.srt.afd.merge_where --batch 8 --repeats 400

E's last step per layer combines two halves: the history the pool swept, which arrives over a
socket and is therefore already in host memory, and this step's own token, which the host has on
the device. Attention over a single position is not much of a computation -- the join's output IS
this token's value and its log partition is one dot product -- so where the two are combined is a
question about COPIES and KERNEL LAUNCHES, not about arithmetic.

    on the GPU     copy the pool's o and lse up, then run the join and the merge as kernels
    on the CPU     merge in host memory, then copy one result up

The second moves the same number of bytes across PCIe and launches nothing, which is why it is
worth asking about. Against it: this is a decode step, the host is running a Python scheduler, and
the numbers are small enough that a kernel launch and a PCIe descriptor are the same order as the
work itself. Neither direction is obvious, so both are timed.

## Why the copies are inside the timing

A benchmark of merge_state against a torch CPU add measures two kernels and answers nothing: in the
arrangement, one of them is preceded by a copy the other does not need and followed by a copy the
other does. Each arm here is timed end to end -- from "the socket has handed over host memory" to
"the device has the merged result" -- because that is the interval the schedule actually waits on.

## What the CPU arm assumes, and why the assumption is checked

The CPU arm needs this step's k and v in host memory. In the arrangement they are there already:
the host stages them for the append it posts to the pool. If that staging copy did not exist the
CPU arm would have to make it, and the arm is timed both ways -- `staged` for the arrangement as
built, `unstaged` for the arrangement without the append -- because quoting the first as if it were
the second would credit this merge with a copy somebody else pays for.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

Q_HEADS, HEAD_DIM = 24, 256


def _merge(o_hist, lse_hist, o_join, lse_join):
    """The mergeable-aggregate combine, written out rather than called.

    sglang's own merge_state is a CUDA kernel and has no CPU path, so a CPU arm that called it
    would be timing a device-to-host fallback. This is the same arithmetic in whatever backend the
    tensors are on, which is what makes the two arms comparable.
    """
    top = torch.maximum(lse_hist, lse_join)
    w_hist = torch.exp(lse_hist - top).unsqueeze(-1)
    w_join = torch.exp(lse_join - top).unsqueeze(-1)
    return (o_hist * w_hist + o_join * w_join) / (w_hist + w_join)


def _timed(fn, repeats: int, sync: bool) -> dict:
    for _ in range(max(8, repeats // 10)):
        fn()
    if sync:
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6)
    samples.sort()
    return {"median_us": statistics.median(samples),
            "p10_us": samples[len(samples) // 10],
            "p90_us": samples[9 * len(samples) // 10]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--repeats", type=int, required=True)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("  no device; this compares a device path against a host path")
    device = torch.device("cuda")
    rows = a.batch

    # what the socket handed over: the pool's swept history, in host memory, pinned because a
    # pageable copy measures the allocator rather than the link
    # (rows, heads, head_dim) rather than the flat (rows, heads*head_dim) the wire carries: the
    # log partition is PER HEAD, so a flat operand makes lse broadcast against the head dimension
    # and the whole merge silently wrong. The reshape is free and the shape states the invariant.
    o_hist_cpu = torch.randn(rows, Q_HEADS, HEAD_DIM, dtype=torch.float32).pin_memory()
    lse_hist_cpu = torch.randn(rows, Q_HEADS, dtype=torch.float32).pin_memory()
    # this step's own token: on the device, where the layer computed it
    v_gpu = torch.randn(rows, Q_HEADS, HEAD_DIM, dtype=torch.float32, device=device)
    lse_join_gpu = torch.randn(rows, Q_HEADS, dtype=torch.float32, device=device)
    # the same, staged in host memory because the append posts them to the pool anyway
    v_cpu = v_gpu.to("cpu").pin_memory()
    lse_join_cpu = lse_join_gpu.to("cpu").pin_memory()

    def on_gpu():
        o_hist = o_hist_cpu.to(device, non_blocking=True)
        lse_hist = lse_hist_cpu.to(device, non_blocking=True)
        return _merge(o_hist, lse_hist, v_gpu, lse_join_gpu)

    def on_cpu_staged():
        merged = _merge(o_hist_cpu, lse_hist_cpu, v_cpu, lse_join_cpu)
        return merged.to(device, non_blocking=True)

    def on_cpu_unstaged():
        v = v_gpu.to("cpu", non_blocking=True)
        lse = lse_join_gpu.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        merged = _merge(o_hist_cpu, lse_hist_cpu, v, lse)
        return merged.to(device, non_blocking=True)

    print(f"  batch {rows}, {Q_HEADS} head(s) x {HEAD_DIM}, "
          f"{rows * Q_HEADS * HEAD_DIM * 4 / 1024:.0f} KiB an operand, {a.repeats} repeat(s)\n")
    print(f"    {'arm':22} {'median':>9} {'p10':>9} {'p90':>9}")
    out = {}
    for name, fn in (("gpu (copy up, merge)", on_gpu),
                     ("cpu (append staged it)", on_cpu_staged),
                     ("cpu (copy down first)", on_cpu_unstaged)):
        out[name] = _timed(fn, a.repeats, sync=True)
        r = out[name]
        print(f"    {name:22} {r['median_us']:7.1f}us {r['p10_us']:7.1f}us {r['p90_us']:7.1f}us")

    gpu = out["gpu (copy up, merge)"]["median_us"]
    for name in ("cpu (append staged it)", "cpu (copy down first)"):
        ratio = out[name]["median_us"] / gpu
        verdict = "faster" if ratio < 1 else "slower"
        print(f"\n  {name} is {ratio:.2f}x the gpu arm -- {verdict}")
    print("\n  Per converted layer, so multiply by the number of layers that sweep and by the")
    print("  steps in a generation before treating any of it as a speedup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
