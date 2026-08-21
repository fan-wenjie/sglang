"""What NCCL costs for the shapes a cache pool moves, measured against the socket protocol.

Two processes, one per machine:

    rank 0, on the host    python -m sglang.srt.afd.nccl_probe --rank 0 --world 2 \\
                               --master HOST:PORT --device cuda --rows 8
    rank 1, on the pool    python -m sglang.srt.afd.nccl_probe --rank 1 --world 2 \\
                               --master HOST:PORT --device cuda --rows 8

This exists for the arrangement that does not exist yet: several hosts sharing one cache pool. The
current one is a single host and a single pool talking over a socket, and for THAT the measurements
already say a transport change buys little -- the round trip is 934 us of which 130 us is the
network's own latency and the rest is a payload that cannot fill a 157 KiB pipe. What changes with
several hosts is that the pool's inbound becomes a gather, and a gather of N queries has N times
the bytes in flight, which is the regime where more sockets and a collective library start to pay.

## What is measured, and why each one

    send/recv      one query out, one output back, the pattern a sweep uses today. Compared
                   directly against the socket protocol's 934 us at the same payload
    gather         N ranks' queries into one, which is what a pool serving N hosts receives. The
                   number worth watching is whether it costs more than one send, because if it
                   does not then the pool's inbound is free up to N
    all_gather     the same, replicated. Included because it is what a naive implementation
                   reaches for and it moves N times the bytes for no reason here

## What this cannot show on this hardware

There is no InfiniBand and no libibverbs on either machine, so NCCL falls back to its socket
transport and GPUDirect is unavailable: the bytes still go GPU -> host -> socket -> host -> GPU.
The gain over a hand-written socket is therefore NCCL's C++ path and its multiple sockets per
connection, not a fundamentally shorter route. On a machine with RDMA the same script measures
something quite different, and that difference is the point of having the script.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

Q_HEADS, HEAD_DIM = 24, 256


def _timed(fn, repeats: int) -> dict:
    for _ in range(max(8, repeats // 10)):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6)
    samples.sort()
    return {"median_us": statistics.median(samples), "p10_us": samples[len(samples) // 10],
            "p90_us": samples[9 * len(samples) // 10], "min_us": samples[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--master", required=True, help="HOST:PORT the rendezvous store binds on")
    ap.add_argument("--device", required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--repeats", type=int, required=True)
    ap.add_argument("--iface", required=True,
                    help="NCCL_SOCKET_IFNAME. No default: NCCL picks an interface by heuristic "
                         "and on a machine with several it can pick the one that does not route "
                         "to the peer, which hangs rather than fails.")
    ap.add_argument("--out")
    a = ap.parse_args()

    host, port = a.master.rsplit(":", 1)
    os.environ.setdefault("MASTER_ADDR", host)
    os.environ.setdefault("MASTER_PORT", port)
    os.environ["NCCL_SOCKET_IFNAME"] = a.iface
    torch.cuda.set_device(0)

    dist.init_process_group(backend="nccl", rank=a.rank, world_size=a.world,
                            timeout=__import__("datetime").timedelta(seconds=120))
    peer = 1 - a.rank if a.world == 2 else None
    device = torch.device(a.device)

    q = torch.randn(a.rows, Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
    o = torch.empty_like(q)
    payload = q.numel() * q.element_size()
    if a.rank == 0:
        print(f"  {a.world} rank(s), {a.rows} row(s) = {payload / 1024:.1f} KiB a direction, "
              f"iface {a.iface}", flush=True)

    results: dict = {}

    if a.world == 2:
        def round_trip():
            if a.rank == 0:
                dist.send(q, dst=peer)
                dist.recv(o, src=peer)
            else:
                dist.recv(o, src=peer)
                dist.send(o, dst=peer)
        results["send_recv"] = _timed(round_trip, a.repeats)

    # The arms above let both ranks run in a tight loop, so a peer can post its half during the
    # previous iteration's slack and the timing catches only the tail of the wait. Everything
    # below puts a barrier first, which is what a real layer faces: the query cannot be sent until
    # the previous layer's output has arrived, so nothing is ever posted early.
    def barriered(body):
        def run():
            dist.barrier()
            body()
        return run

    def serial_round_trip():
        if a.rank == 0:
            dist.send(q, dst=peer)
            dist.recv(o, src=peer)
        else:
            dist.recv(o, src=peer)
            dist.send(o, dst=peer)

    if a.world == 2:
        results["barrier_only"] = _timed(barriered(lambda: None), a.repeats)
        results["send_recv_barriered"] = _timed(barriered(serial_round_trip), a.repeats)

    gathered = [torch.empty_like(q) for _ in range(a.world)] if a.rank == 0 else None

    def gather():
        dist.gather(q, gather_list=gathered, dst=0)
    results["gather"] = _timed(gather, a.repeats)

    every = [torch.empty_like(q) for _ in range(a.world)]

    def all_gather():
        dist.all_gather(every, q)
    results["all_gather"] = _timed(all_gather, a.repeats)

    # the collective spelling of one layer: every host's query up, the answer back down. With one
    # host the gather is a send and the broadcast is a receive, which is the question being asked
    # -- whether two one-way collectives beat one synchronous round trip when neither side may
    # run ahead
    answer = torch.empty_like(q)

    def gather_then_broadcast():
        dist.gather(q, gather_list=gathered, dst=0)
        dist.broadcast(answer, src=0)
    results["gather_bcast_barriered"] = _timed(barriered(gather_then_broadcast), a.repeats)

    if a.rank == 0:
        print(f"    {'collective':14} {'median':>9} {'p10':>9} {'p90':>9} {'min':>9}", flush=True)
        for name, r in results.items():
            print(f"    {name:14} {r['median_us']:7.0f}us {r['p10_us']:7.0f}us "
                  f"{r['p90_us']:7.0f}us {r['min_us']:7.0f}us", flush=True)
        print(f"\n  the socket protocol moves the same {payload / 1024:.0f} KiB each way in "
              f"934 us, of which 130 us is the link's own round trip.", flush=True)
        if a.out:
            json.dump({"config": vars(a), "payload_bytes": payload, "results": results},
                      open(a.out, "w"), indent=2)
            print(f"  wrote {a.out}", flush=True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
