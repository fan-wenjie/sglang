"""Where a pool call goes, on this model's real frame.

    python -m sglang.srt.afd.wire_breakdown <pool host:port>

Run it against the real pool and against a pool whose forward is the identity; the difference is
the feed-forward and everything else is transport, framing and Python. That decomposition is what
told this port that its problem was not the network:

    round trip, real pool, scheduler spinning     7.08 ms
    round trip, real pool, --sleep-on-idle        1.18 ms
    round trip, identity pool, no scheduler       0.80 ms
    d2h of the frame                              0.011 ms
    h2d of the frame                              0.014 ms

The 5.9 ms between the first two lines was the pool's own idle scheduler holding the GIL against
the thread answering frames. The copies, which were the obvious suspect, are a fortieth of a
millisecond.


One frame is [tokens, hidden] bfloat16: at batch 4 and hidden 5120 that is 40 KB each way. The
round trip against the real pool was 6.5 ms. This takes it apart, because "the interconnect is
slow" and "the copies and the interpreter are slow" call for opposite fixes and only one of them
is fixed by RDMA.

    wire        the same protocol against a pool whose forward is the identity, so the number is
                transport, framing and Python with the feed-forward removed
    d2h / h2d   the copies the client makes on this side, timed alone
"""
import socket
import statistics
import sys
import time

import torch

TOKENS, HIDDEN = 4, 5120


def main():
    sys.path.insert(0, "/home/user/experiment/sglang/python")
    from sglang.srt.afd.pool_client import PoolClient

    address = sys.argv[1]
    hidden = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)

    torch.cuda.synchronize()
    spans = []
    for _ in range(200):
        t = time.perf_counter()
        host_copy = hidden.cpu()
        spans.append(time.perf_counter() - t)
    d2h = statistics.median(spans) * 1e3

    spans = []
    for _ in range(200):
        t = time.perf_counter()
        host_copy.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
        spans.append(time.perf_counter() - t)
    h2d = statistics.median(spans) * 1e3

    client = PoolClient(address, 10.0)
    peer = client._sock.getpeername()
    local = client._sock.getsockname()
    print(f"  socket {local[0]}:{local[1]} -> {peer[0]}:{peer[1]}", flush=True)

    for _ in range(20):
        client.collect(client.issue(1, 0, hidden), "cuda")
    spans = []
    for i in range(300):
        t = time.perf_counter()
        client.collect(client.issue(i + 100, 0, hidden), "cuda")
        spans.append(time.perf_counter() - t)
    client.close()
    wire = statistics.median(spans) * 1e3

    print(f"  frame           {TOKENS} x {HIDDEN} bf16 = {TOKENS * HIDDEN * 2 / 1024:.0f} KB "
          f"each way", flush=True)
    print(f"  d2h (host side) {d2h:8.3f} ms", flush=True)
    print(f"  h2d (host side) {h2d:8.3f} ms", flush=True)
    print(f"  round trip with NO feed-forward  {wire:8.3f} ms   "
          f"(p10 {statistics.quantiles(spans, n=10)[0]*1e3:.3f}, "
          f"p90 {statistics.quantiles(spans, n=10)[8]*1e3:.3f})", flush=True)
    print(f"  against 6.5 ms measured with the real feed-forward: transport and Python are "
          f"{100*wire/6.5:.0f}% of it", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
