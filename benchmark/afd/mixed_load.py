"""Many requests of many context lengths at once, and what the mixture costs each of them.

    python -m sglang.srt.afd.mixed_load --weights HOST:PORT \\
        --lengths 256,1024,4096,16384 --per-length 8 --waves 6 --out mixed.json

A uniform load answers "how fast is this at length L". Real traffic is not uniform, and three of
this arrangement's mechanisms are shaped by the mixture rather than by any one length:

    head-of-line blocking   a 16k sweep and a 256-token sweep in the same step take very different
                            times, and the short one's answer cannot leave before the step does
    departure width         `DepartureQueue` pads a batch to a fixed width. A mixture pads to the
                            widest member, so the short requests in a mixed batch pay for the long
                            ones' shape as well as for their own turn
    rendezvous depth        a request's two halves are held until both arrive. Halves belonging to
                            short requests arrive early and wait, so the buffer holds more at once
                            under a mixture than under any single length

None of the three shows up in a uniform run at any length, including the long one.

## What is actually compared

Each length is first run ALONE, then all lengths are run SHUFFLED TOGETHER. The number reported per
length is the mixed latency over the alone latency: how much this length pays for sharing a server
with the others. A single mixed run has nothing to be slow relative to, and "the 256s got 40 ms" is
not a finding without the 25 ms they get by themselves.

Both arms are measured the same way, because the question is not whether a mixture costs something
-- it does, on any server -- but whether disaggregation makes it cost MORE. Colocated is the
control for that, and without it a ratio of 1.8x could be this arrangement's fault or could be what
the scheduler does to any mixed batch.

## Requests must ARRIVE independently, or the buckets are one number wearing four hats

The first version of this submitted each wave as one `engine.generate([...])` call. Every request
in such a call starts and finishes in the same batch, so `e2e_latency` came back as the batch's
wall clock -- the 256-token bucket and the 4096-token bucket reported p50s of 1.599625 and
1.599642 seconds, identical to five figures, because they were the same measurement. Nothing about
head-of-line blocking can be read out of that, and the per-bucket table it printed was four copies
of one number with different labels.

So arrivals are generated here: exponential inter-arrival times at a requested rate, each request
submitted through `async_generate` when its turn comes, and latency measured client-side as
completion minus arrival. That is the definition that does not depend on the server agreeing about
when a request began, and it is the one under which a short request stuck behind a long one shows
up as a long latency for the SHORT request.

## Reading the tail, not the mean

Head-of-line blocking moves the tail and barely moves the mean: one short request stuck behind a
long one is one sample. p50, p95 and max are reported per bucket, and the p95/p50 spread is the
number that answers the blocking question. A mean would hide exactly the failure this tool is for.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import os
import sys
import time

# `under_test` moved out of the runtime package and into this directory when the arrangement's
# code was split for merging; five benchmarks kept importing it from where it used to be and none
# of them had been run since. A benchmark that cannot import is at least loud -- the dangerous
# version of this is the one that imports something ELSE of the same name.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from under_test import model_path

# named once, in the environment, and checked for quantisation before the weights
# load. It was written down separately in eight tools, which is eight chances to
# leave one on the old checkpoint and report its numbers under the new one's name.
MODEL = model_path()
FILLER = ("The scheduler assigns each warp a slot in the issue queue, and the memory controller "
          "coalesces the resulting requests into cache lines before they reach the crossbar. ")
TOKENS = 24          # generated per request, the same for every length so decode work is equal


def prompt_of(tokenizer, target: int) -> str:
    text = FILLER
    while len(tokenizer(text)["input_ids"]) < target:
        text += FILLER
    return tokenizer.decode(tokenizer(text)["input_ids"][:target])


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _summarise(latencies: list[float]) -> dict:
    return {
        "n": len(latencies),
        "p50": _percentile(latencies, 0.50),
        "p95": _percentile(latencies, 0.95),
        "max": max(latencies),
        "mean": statistics.fmean(latencies),
    }


async def _one(engine, prompt: str, delay: float, index: int, out: dict) -> None:
    """Sleep until this request's arrival time, then submit it and time it from there.

    Keyed by the caller's index rather than appended: results come back in completion order, and
    completion order is exactly what a mixture scrambles. Appending would hand a 16k request's
    latency to whichever bucket happened to be next in the plan.
    """
    await asyncio.sleep(delay)
    arrived = time.perf_counter()
    result = await engine.async_generate(
        prompt, sampling_params={"temperature": 0.0, "max_new_tokens": TOKENS})
    out[index] = (time.perf_counter() - arrived, result["meta_info"]["completion_tokens"])


async def _drive(engine, plan: list, prompts: dict, rate: float, rng: random.Random) -> dict:
    """Submit `plan` at exponential inter-arrival times, all in flight at once.

    Every task is created up front and sleeps to its own arrival, so a server that falls behind
    does not slow the arrivals down. A closed loop that waited for one request before sending the
    next would measure a different system -- one whose offered load drops when it gets slow, which
    is the opposite of what a queue does.
    """
    delay, delays = 0.0, []
    for _ in plan:
        delays.append(delay)
        delay += rng.expovariate(rate)

    out: dict[int, tuple] = {}
    await asyncio.gather(*[
        _one(engine, prompts[length], at, index, out)
        for index, (length, at) in enumerate(zip(plan, delays))
    ])
    return out


def _issue(engine, plan: list, prompts: dict, rate: float, seed: int) -> list[float]:
    """One latency per request, in the plan's order, measured from each request's own arrival.

    Driven on the ENGINE's own loop, not a fresh one. `sgl.Engine` builds an event loop when it is
    constructed and every one of its synchronous methods runs on that loop; the tokenizer manager's
    queues and futures are bound to it. Calling `asyncio.run` here made a second loop, and awaiting
    a primitive owned by the first one from inside the second does not raise -- it simply never
    wakes. The run held 75 GB of device memory at 0% utilisation for twelve minutes before that was
    noticed, so the symptom of this mistake is silence, not an error.
    """
    results = engine.loop.run_until_complete(
        _drive(engine, plan, prompts, rate, random.Random(seed)))
    latencies = []
    for index in range(len(plan)):
        latency, produced = results[index]
        if produced != TOKENS:
            raise SystemExit(
                f"  a request produced {produced} tokens rather than {TOKENS}. The buckets are "
                f"only comparable while every request does the same decode work, so a short "
                f"generation makes this measurement about something else."
            )
        latencies.append(latency)
    if len(latencies) > 4 and len(set(round(x, 4) for x in latencies)) == 1:
        raise SystemExit(
            f"  all {len(latencies)} latencies are identical to four decimals, so they are one "
            f"batch's wall clock rather than {len(latencies)} arrivals. This is the defect the "
            f"first version of this tool shipped with; lower --rate until requests are scheduled "
            f"separately."
        )
    return latencies


def alone(engine, prompts: dict, per_length: int, waves: int, rate: float, seed: int) -> dict:
    """Each length by itself, at the same arrival rate: the baseline the mixture is charged against.

    The rate is per-length here and total in the mixture, so a length alone sees the same arrivals
    per second that it sees inside the mixture. Holding the TOTAL rate equal instead would offer
    each length four times its share when alone, and the mixture would look good by comparison.
    """
    out = {}
    for length, prompt in prompts.items():
        plan = [length] * per_length
        _issue(engine, plan, prompts, rate, seed)                  # warm this length's shapes
        latencies = []
        for wave in range(waves):
            latencies.extend(_issue(engine, plan, prompts, rate, seed + wave))
            # printed per wave, not per phase. A run that says nothing for ten minutes cannot be
            # told from a run that has deadlocked, and one of these did deadlock: a second event
            # loop held 75 GB at 0% utilisation, silently, and looked exactly like waiting for
            # arrivals.
            print(f"      alone {length:6d}: wave {wave + 1}/{waves}, "
                  f"{len(latencies)} request(s) done", flush=True)
        out[length] = _summarise(latencies)
    return out


def mixed(engine, prompts: dict, per_length: int, waves: int, rate: float, seed: int) -> dict:
    """Every length shuffled into one arrival stream, bucketed back apart afterwards.

    Shuffled rather than ordered: a stream whose long requests all arrive first is a different
    experiment from one where they interleave, and a server good at the first would look good here
    without being good at anything real. The seed is in the record so the shuffle can be replayed.
    """
    rng = random.Random(seed)
    plan = [length for length in prompts for _ in range(per_length)]
    rng.shuffle(plan)

    _issue(engine, plan, prompts, rate, seed)                      # warm the mixture itself
    buckets: dict[int, list[float]] = {length: [] for length in prompts}
    started = time.perf_counter()
    for wave in range(waves):
        for length, latency in zip(plan, _issue(engine, plan, prompts, rate, seed + wave)):
            buckets[length].append(latency)
        print(f"      mixed: wave {wave + 1}/{waves}, "
              f"{sum(len(v) for v in buckets.values())} request(s) done", flush=True)
    seconds = time.perf_counter() - started

    summary = {length: _summarise(values) for length, values in buckets.items()}
    requests = sum(len(v) for v in buckets.values())
    summary["_throughput"] = {"requests": requests, "seconds": seconds,
                              "requests_per_second": requests / seconds,
                              "shuffle_seed": seed, "arrival_rate": rate}
    return summary


def report(label: str, solo: dict, mix: dict) -> None:
    print(f"\n  {label}", flush=True)
    print(f"    {'length':>8} {'alone p50':>10} {'mixed p50':>10} {'ratio':>7} "
          f"{'mixed p95':>10} {'p95/p50':>8}", flush=True)
    for length in sorted(solo):
        a, m = solo[length], mix[length]
        print(f"    {length:8d} {a['p50'] * 1e3:9.1f}m {m['p50'] * 1e3:9.1f}m "
              f"{m['p50'] / a['p50']:6.2f}x {m['p95'] * 1e3:9.1f}m "
              f"{m['p95'] / m['p50']:7.2f}x", flush=True)
    t = mix["_throughput"]
    print(f"    {t['requests']} request(s) in {t['seconds']:.1f}s = "
          f"{t['requests_per_second']:.2f} req/s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--per-length", type=int, required=True)
    ap.add_argument("--waves", type=int, required=True)
    ap.add_argument("--mem", type=float, required=True)
    ap.add_argument("--rate", type=float, required=True,
                    help="arrivals per second PER LENGTH; the mixture offers this times the "
                         "number of lengths, so each length sees the same rate either way")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]

    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = {length: prompt_of(tokenizer, length) for length in lengths}
    print(f"  lengths {lengths}, {a.per_length} request(s) each = "
          f"{len(lengths) * a.per_length} per wave, {a.waves} wave(s)", flush=True)

    out = {"config": vars(a), "arms": {}}
    for label, kw in (("colocated", {}),
                      ("A ffn on pool", {"afd_mode": "host", "afd_pool_addr": a.weights,
                                         "afd_query_shift_layers": 1, "afd_coverage": "all"})):
        print(f"    {label}: loading", flush=True)
        engine = sgl.Engine(model_path=MODEL, tp_size=1, mem_fraction_static=a.mem,
                            disable_cuda_graph=True, attention_backend="triton",
                            log_level="warning", **kw)
        try:
            solo = alone(engine, prompts, a.per_length, a.waves, a.rate, a.seed)
            mix = mixed(engine, prompts, a.per_length, a.waves,
                        a.rate * len(lengths), a.seed)
        finally:
            engine.shutdown()
        out["arms"][label] = {"alone": solo, "mixed": mix}
        report(label, solo, mix)

    print(f"\n  {'length':>8} {'colocated pays':>15} {'A pays':>9} {'A / colocated':>14}",
          flush=True)
    for length in lengths:
        ratios = {}
        for label in ("colocated", "A ffn on pool"):
            arm = out["arms"][label]
            ratios[label] = arm["mixed"][length]["p50"] / arm["alone"][length]["p50"]
        print(f"  {length:8d} {ratios['colocated']:14.2f}x {ratios['A ffn on pool']:8.2f}x "
              f"{ratios['A ffn on pool'] / ratios['colocated']:13.2f}x", flush=True)
    print("  the last column is what DISAGGREGATION costs a mixture, with the scheduler's own "
          "share of the interference divided out", flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
