"""One cache-pool round trip, timed at the shapes the reversed arrangement actually uses.

    python -m sglang.srt.afd.cache_probe --cache HOST:PORT --rows 8 --context 1024 --repeats 200

The reversed arrangement measures 6.5 ms of extra cost per converted layer at 1k context, where
the whole cache is 32 KiB. That cannot be bandwidth, and it is 7x a measured round trip, so
something between the host's issue and the host's collect is costing far more than the wire. This
separates the candidates instead of guessing between them:

    append      one k/v pair posted. Grows the pool's history by one position
    sweep       one query answered against that history
    both        what a layer actually does, in the order it does it

If a lone sweep costs about what the protocol floor says, the pool is not the problem and the cost
is on the host -- a window that never opens, a collect that happens before the issue it was
supposed to overlap. If a lone sweep costs milliseconds, the pool is the problem, and the append
and sweep arms say which half.

## Why this talks to the pool directly rather than through an engine

An engine measurement includes the model, the scheduler, the Python that walks the layers, and the
attention backend, and every one of them has been the answer to a question like this at least once
in this tree. A socket, two frames and a stopwatch have none of that in them: whatever this
reports is the pool and the wire and nothing else.

The cache is filled by this tool before it is swept, because a sweep over an empty history is a
different and much cheaper operation -- the pool returns a zero output and a -inf partition without
touching a tensor, and timing that would report a round trip as free.
"""

from __future__ import annotations

import argparse
import socket
import statistics
import sys
import time

import torch

from sglang.srt.afd.protocol import (
    OP_APPEND,
    OP_HELLO,
    OP_RELEASE,
    OP_SWEEP_Q,
    Frame,
    decode,
    send_frame,
)

Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256


def _call(sock, frame: Frame) -> tuple:
    started = time.perf_counter()
    send_frame(sock, frame)
    reply = decode(sock)
    if reply is None:
        raise SystemExit("  the pool closed the connection mid-call")
    return (time.perf_counter() - started) * 1e3, reply


def _stats(samples: list) -> dict:
    samples = sorted(samples)
    return {"median_ms": statistics.median(samples),
            "p10_ms": samples[len(samples) // 10],
            "p90_ms": samples[9 * len(samples) // 10],
            "min_ms": samples[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--repeats", type=int, required=True)
    ap.add_argument("--layer", type=int, required=True)
    a = ap.parse_args()

    host, port = a.cache.split(":")
    sock = socket.create_connection((host, int(port)), timeout=60)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    hello_ms, reply = _call(sock, Frame(1, 0, [torch.zeros(1, 1)], OP_HELLO))
    print(f"  pool at {a.cache}: capabilities {reply.tensors[0].flatten().tolist()}, "
          f"HELLO round trip {hello_ms:.3f} ms", flush=True)

    ids = torch.arange(a.rows, dtype=torch.float32).view(-1, 1)
    k = torch.randn(a.rows, KV_HEADS * HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(a.rows, KV_HEADS * HEAD_DIM, dtype=torch.bfloat16)
    q = torch.randn(a.rows, Q_HEADS * HEAD_DIM, dtype=torch.bfloat16)

    print(f"  filling {a.rows} history(ies) to {a.context} position(s) at layer {a.layer}",
          flush=True)
    fill_started = time.perf_counter()
    fill_samples = []
    for step in range(a.context):
        ms, _ = _call(sock, Frame(0, a.layer, [k, v, ids], OP_APPEND))
        fill_samples.append(ms)
    fill_seconds = time.perf_counter() - fill_started
    print(f"    filled in {fill_seconds:.1f}s; the append cost drifted "
          f"{statistics.median(fill_samples[:50]):.3f} -> "
          f"{statistics.median(fill_samples[-50:]):.3f} ms across the fill", flush=True)

    lengths = torch.full((a.rows, 1), float(a.context))
    arms = {
        "sweep alone": lambda: _call(sock, Frame(0, a.layer, [q, ids, lengths], OP_SWEEP_Q)),
        "append alone": lambda: _call(sock, Frame(0, a.layer, [k, v, ids], OP_APPEND)),
    }
    out = {}
    for name, call in arms.items():
        for _ in range(max(8, a.repeats // 10)):
            call()
        out[name] = _stats([call()[0] for _ in range(a.repeats)])

    print(f"\n    {'arm':16} {'median':>9} {'p10':>9} {'p90':>9} {'min':>9}", flush=True)
    for name, s in out.items():
        print(f"    {name:16} {s['median_ms']:7.3f}ms {s['p10_ms']:7.3f}ms "
              f"{s['p90_ms']:7.3f}ms {s['min_ms']:7.3f}ms", flush=True)

    layer_ms = out["sweep alone"]["median_ms"] + out["append alone"]["median_ms"]
    print(f"\n  a converted layer makes both calls: {layer_ms:.3f} ms of pool time.")
    print(f"  The reversed arm measured 6.5 ms of extra cost per layer at 1k context. If the")
    print(f"  number above is far below that, the pool is answering quickly and the cost is on")
    print(f"  the host side -- a window that is not open, or a collect that does not overlap.")

    _call(sock, Frame(0, a.layer, [torch.zeros(1, 1)], OP_RELEASE))
    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
