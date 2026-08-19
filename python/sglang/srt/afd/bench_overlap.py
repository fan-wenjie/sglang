"""How much of the pool's feed-forward the sweep actually hides, on this model's decode shapes.

    python -m sglang.srt.afd.bench_overlap --model <path> --context 8192 --layers 8 \\
        --steps 20 --link-us 0 --out afd_overlap.json

The claim the whole arrangement rests on is that a sweep over cached keys needs only the query, so
it can run while the pool computes the feed-forward that precedes it. Everything up to here has
established that the split is exact, that the read point moves, and that the plumbing carries the
work. None of it establishes that anything OVERLAPPED, and a benchmark that assumed it would have
reported a speedup made of nothing.

So this measures the two arms against each other with the same weights, the same shapes and the
same pool:

    synchronous   issue the feed-forward, wait for it, then attend. This is what a port that keeps
                  the standard read point must do: layer l+1's query is projected from x_{l+1},
                  which does not exist until the feed-forward has returned.
    q-first       issue the feed-forward, sweep the cache with the query that already exists, then
                  collect and fold the current position in.

Both arms do the same arithmetic. The only difference is the order, which is the point.

## What is real here and what is a stand-in

Real: the model's own feed-forward weights on the pool, its attention shapes (24 query heads, 4
key-value heads, head dim 256), its hidden width, and a genuine socket between the two sides.

A stand-in: the interconnect. `--link-us` adds a one-way delay to every frame so a reader can see
where the crossover sits, because on one card the two sides share a bus and the measured latency
is that bus, not a network. A single machine cannot answer whether this pays -- that is a ratio
between an interconnect's latency and a feed-forward's duration -- and this tool reports the ratio
it measured rather than implying it generalises.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time

import torch


class _Ready(threading.Event):
    port = 0


def start_pool(mlp, device, min_batch: int, max_wait_s: float, link_us: float) -> int:
    """Serve one layer's feed-forward. Returns the bound port."""
    from sglang.srt.afd.pool_server import serve

    delay = link_us / 1e6

    def forward(batch: torch.Tensor, layer: int) -> torch.Tensor:
        if delay:
            time.sleep(delay)          # the reply's half of a round trip
        with torch.no_grad():
            out = mlp(batch.to(device))
        out = out[0] if isinstance(out, tuple) else out
        torch.cuda.synchronize()
        return out

    ready = _Ready()
    threading.Thread(
        target=serve,
        args=(forward, "127.0.0.1", 0, min_batch, max_wait_s, device, ready),
        daemon=True,
    ).start()
    if not ready.wait(timeout=30):
        raise SystemExit("  the pool did not bind")
    return ready.port


def sweep(q, k_cache, v_cache, scale):
    """The cache side: attention over the cached positions, as (vector, log partition).

    `v_j` is absent from the signature and that absence is the protocol -- this is what can run
    before the feed-forward that produces the current value exists.
    """
    s = torch.einsum("hd,hjd->hj", q, k_cache) * scale
    lse = torch.logsumexp(s, dim=-1)
    return torch.einsum("hj,hjd->hd", torch.softmax(s, dim=-1), v_cache), lse


def join(o_lt, lse_lt, s_jj, v_j):
    """The compute side: fold the current position in. A two-way softmax is a logistic."""
    return torch.lerp(v_j, o_lt, torch.sigmoid(lse_lt - s_jj).unsqueeze(-1))


def one_step(client, mlp_input, layer, request_id, q, k_cache, v_cache, k_j, v_j, scale,
             qfirst: bool):
    """One decode step of one layer, in whichever order the arm calls for."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    handle = client.issue(request_id, layer, mlp_input)
    if not qfirst:
        # the standard read point: nothing may start until x_{l+1} is back
        client.collect(handle, q.device)
        o_lt, lse_lt = sweep(q, k_cache, v_cache, scale)
    else:
        o_lt, lse_lt = sweep(q, k_cache, v_cache, scale)
        client.collect(handle, q.device)
    s_jj = (q * k_j).sum(-1) * scale
    out = join(o_lt, lse_lt, s_jj, v_j)
    torch.cuda.synchronize()
    return time.perf_counter() - started, out


def run_requests(client, n_requests: int, steps: int, state, qfirst: bool) -> dict:
    """N requests against one pool, each in its own thread, each with its own cache.

    This is the case the arrangement exists for and the only one that can show it: a departure
    carries several callers, so the pool's cost is shared, and a request that stalls between its
    own calls blocks nobody. One request cannot demonstrate any of that -- and cannot fail the
    way concurrency fails, where two callers at the same layer overwrite one slot and both get
    plausible answers from the wrong request.
    """
    spans: dict[int, list] = {}
    checks: dict[int, float] = {}
    errors: list = []
    barrier = threading.Barrier(n_requests)

    def one_request(request_id: int):
        q, k_cache, v_cache, k_j, v_j, hidden_in, scale = state[request_id]
        mine = []
        try:
            for _ in range(3):
                one_step(client, hidden_in, 0, request_id, q, k_cache, v_cache, k_j, v_j,
                         scale, qfirst)
            barrier.wait(timeout=120)          # start the measured steps together
            for _ in range(steps):
                span, out = one_step(client, hidden_in, 0, request_id, q, k_cache, v_cache,
                                     k_j, v_j, scale, qfirst)
                mine.append(span)
            checks[request_id] = float(out.float().abs().sum())
        except BaseException as e:  # noqa: BLE001 -- reported, not swallowed
            errors.append((request_id, repr(e)))
        spans[request_id] = mine

    threads = [threading.Thread(target=one_request, args=(r,), daemon=True)
               for r in state]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)
    wall = time.perf_counter() - started
    if errors:
        raise SystemExit(f"  {len(errors)} request(s) failed: {errors[:3]}")
    flat = [s for v in spans.values() for s in v]
    return {
        "requests": n_requests,
        "steps_each": steps,
        "median_step_s": statistics.median(flat),
        "wall_s": wall,
        "steps_per_second": len(flat) / wall,
        "checksums": checks,
    }


def make_state(n_requests: int, n_q: int, head_dim: int, hidden: int, context: int) -> dict:
    """Each request gets its own cache and its own query, so a crossed reply is visible."""
    state = {}
    for request_id in range(1, n_requests + 1):
        torch.manual_seed(request_id)
        state[request_id] = (
            torch.randn(n_q, head_dim, device="cuda").float(),
            torch.randn(n_q, context, head_dim, device="cuda").float(),
            torch.randn(n_q, context, head_dim, device="cuda").float(),
            torch.randn(n_q, head_dim, device="cuda").float(),
            torch.randn(n_q, head_dim, device="cuda").float(),
            torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16),
            head_dim**-0.5,
        )
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    for name in ("context", "layers", "steps", "min-batch"):
        ap.add_argument(f"--{name}", type=int, required=True)
    ap.add_argument("--link-us", type=float, required=True)
    ap.add_argument("--max-wait-ms", type=float, required=True)
    ap.add_argument("--requests", type=int, required=True,
                    help="concurrent requests sharing the pool; 1 measures nothing "
                         "about pooling")
    a = ap.parse_args()

    from sglang.srt.afd.pool_client import PoolClient
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(a.model).get_text_config()
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim, hidden = cfg.head_dim, cfg.hidden_size
    scale = head_dim**-0.5
    print(f"  {n_q} query heads, {n_kv} key-value heads, head dim {head_dim}, "
          f"hidden {hidden}; context {a.context}", flush=True)

    # A feed-forward of this model's own shape. The pool's cost is what it reads, and what it
    # reads is these weights; a smaller stand-in would understate the thing being hidden.
    mlp = torch.nn.Sequential(
        torch.nn.Linear(hidden, cfg.intermediate_size, bias=False),
        torch.nn.SiLU(),
        torch.nn.Linear(cfg.intermediate_size, hidden, bias=False),
    ).to("cuda", torch.bfloat16).eval()

    port = start_pool(mlp, "cuda", a.min_batch, a.max_wait_ms / 1000.0, a.link_us)
    client = PoolClient(f"127.0.0.1:{port}", connect_timeout_s=10)

    torch.manual_seed(0)
    rep = n_q // n_kv
    k_cache = torch.randn(n_kv, a.context, head_dim, device="cuda",
                          dtype=torch.bfloat16).repeat_interleave(rep, 0).float()
    v_cache = torch.randn(n_kv, a.context, head_dim, device="cuda",
                          dtype=torch.bfloat16).repeat_interleave(rep, 0).float()
    q = torch.randn(n_q, head_dim, device="cuda").float()
    k_j = torch.randn(n_q, head_dim, device="cuda").float()
    v_j = torch.randn(n_q, head_dim, device="cuda").float()
    hidden_in = torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16)

    results = {}
    try:
        for arm, qfirst in (("synchronous", False), ("q-first", True)):
            for _ in range(3):        # warm the kernels; a first call times the compiler
                one_step(client, hidden_in, 0, 1, q, k_cache, v_cache, k_j, v_j, scale, qfirst)
            spans, out = [], None
            for _ in range(a.steps):
                span, out = one_step(client, hidden_in, 0, 1, q, k_cache, v_cache, k_j, v_j,
                                     scale, qfirst)
                spans.append(span)
            results[arm] = {
                "median_s": statistics.median(spans),
                "mean_s": statistics.fmean(spans),
                "min_s": min(spans),
                "steps": len(spans),
                "output_checksum": float(out.float().abs().sum()),
            }
            print(f"  {arm:<12} median {results[arm]['median_s'] * 1e3:.3f} ms", flush=True)

        concurrent = {}
        if a.requests > 1:
            state = make_state(a.requests, n_q, head_dim, hidden, a.context)
            for arm, qfirst in (("synchronous", False), ("q-first", True)):
                concurrent[arm] = run_requests(client, a.requests, a.steps, state, qfirst)
                c = concurrent[arm]
                print(f"  {a.requests} concurrent, {arm:<12} "
                      f"{c['steps_per_second']:.1f} steps/s, median step "
                      f"{c['median_step_s'] * 1e3:.3f} ms", flush=True)
            # every request must have got ITS OWN answer. Two callers at one layer sharing a slot
            # would each receive a plausible tensor from the other, and no assertion downstream
            # would catch it -- this is the only place it is visible.
            sync_sums = concurrent["synchronous"]["checksums"]
            qf_sums = concurrent["q-first"]["checksums"]
            for request_id, value in sync_sums.items():
                if abs(value - qf_sums[request_id]) > 1e-3:
                    raise SystemExit(
                        f"  request {request_id} got different answers from the two arms "
                        f"({value} and {qf_sums[request_id]}); a reply was crossed between "
                        f"requests, which is the failure the (request, layer) key exists for"
                    )
            if len({round(v, 3) for v in sync_sums.values()}) != len(sync_sums):
                raise SystemExit(
                    "  two requests produced identical checksums; their caches were meant to "
                    "differ, so either the state was shared or the replies were crossed"
                )
        report = client.overlap_report()
    finally:
        client.close()

    a_sync, a_qf = results["synchronous"], results["q-first"]
    if abs(a_sync["output_checksum"] - a_qf["output_checksum"]) > 1e-3:
        raise SystemExit(
            f"  the two arms computed different things "
            f"({a_sync['output_checksum']} and {a_qf['output_checksum']}); a speed comparison "
            f"between them would be meaningless"
        )
    hidden_s = a_sync["median_s"] - a_qf["median_s"]
    out = {
        "model": a.model,
        "config": vars(a),
        "shapes": {"q_heads": n_q, "kv_heads": n_kv, "head_dim": head_dim, "hidden": hidden,
                   "intermediate": cfg.intermediate_size},
        "arms": results,
        "concurrent": concurrent,
        "pool_calls": report,
        "single_request": {
            "hidden_s": hidden_s,
            "hidden_pct": 100.0 * hidden_s / a_sync["median_s"],
            "speedup": a_sync["median_s"] / a_qf["median_s"],
        },
    }
    print(f"\n  one request:  hid {hidden_s * 1e3:.3f} ms, "
          f"{out['single_request']['hidden_pct']:.1f}% of the synchronous step "
          f"({out['single_request']['speedup']:.2f}x)", flush=True)
    if a.min_batch > 1:
        print(f"  A lone caller waits the full {a.max_wait_ms} ms for a partner that never comes, "
              f"so this\n  arm is mostly that timeout. It is the batching trade-off, not a "
              f"property of the wiring.", flush=True)

    if concurrent:
        cs, cq = concurrent["synchronous"], concurrent["q-first"]
        out["concurrent_summary"] = {
            "requests": a.requests,
            "throughput_speedup": cq["steps_per_second"] / cs["steps_per_second"],
            "median_step_ratio": cq["median_step_s"] / cs["median_step_s"],
            "steps_per_second": {"synchronous": cs["steps_per_second"],
                                 "q-first": cq["steps_per_second"]},
        }
        print(f"  {a.requests} requests: {cs['steps_per_second']:.1f} -> "
              f"{cq['steps_per_second']:.1f} steps/s "
              f"({out['concurrent_summary']['throughput_speedup']:.2f}x), median step "
              f"{cs['median_step_s'] * 1e3:.3f} -> {cq['median_step_s'] * 1e3:.3f} ms",
              flush=True)
        print("  The concurrent figure is the one the arrangement is about: a pool is worth "
              "having\n  because several callers share a departure, and a lone caller cannot "
              "show that.", flush=True)

    print("\n  On one card the link is a bus, not a network. Whether this pays on real hardware "
          "is a\n  ratio between an interconnect's latency and a feed-forward's duration, and "
          "one machine\n  cannot answer it.", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
