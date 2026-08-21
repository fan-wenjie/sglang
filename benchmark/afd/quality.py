"""Bits per byte and perplexity of the stock stack against the Early-Q one, through the server.

    python -m sglang.srt.afd.quality --model <path> --tokens <val.u32> --shift 1 \\
        --sequences 24 --seq 1024 --seed 0 --bytes-per-token 4.3775 \\
        --mem-fraction 0.80 --out afd_bpb.json

Measured through `sgl.Engine` with input logprobs, which is the path a deployment runs. Two
engines are started in turn -- one at shift 0, one at the shift under test -- because the wiring
installs when the model loads, inside the scheduler process, and a caller holding an Engine cannot
reach the model to install it afterwards.

## Both arms are quantised, and that is the point

This checkpoint is FP8. The study's own figure for this model was measured in bfloat16, so the two
are not comparable and putting them side by side would fold quantisation into a number about a
rewiring. Both arms here are the same FP8 weights; what is reported is the DELTA between them,
which is what the rewiring is responsible for.

## Three arms, because two would confound two changes

Partitioning attention into a sweep and a join reassociates the additions inside one softmax, so
the split arm is not bit-identical to the fused one even though it is the same mathematics. The
split is on by default wherever the read point moved, so a two-arm run would report the read
point's cost and the reassociation's together under the read point's name.

    stock          shift 0, fused              the model as shipped
    shifted        shift N, fused              what moving the read point costs
    shifted_split  shift N, partitioned        what the schedule costs on top of that

`shifted_split` minus `shifted` is NOT a measurement of what the partition costs, and reading it
as one would be the mistake this paragraph exists to prevent. Scoring input logprobs is a prefill,
and the partition is refused in prefill -- a chunk's join is a causal attention among its own
tokens rather than one position per request. So the third arm runs the fused kernel at every layer
and its delta is exactly zero by construction.

What the arm is worth keeping for is the other direction: installing the schedule must not perturb
the forward passes it does not split. A non-zero delta here means the wiring changed something on
a path it was supposed to leave alone. The partition's own exactness is measured where it runs, by
`--afd-verify-split` on decode traffic.

## Why both bpb and ppl

Bits per byte is comparable across tokenisations and is what the study reports; perplexity is what
a serving audience reads. They move differently -- a relative change in one is not the relative
change in the other -- so quoting one as the other misstates the cost by a factor no reader can
reconstruct. `bytes_per_token` belongs to the corpus and the tokeniser, so both arms are handed the
same one rather than deriving it twice.

    nats per token   the model's own cross-entropy over the scored positions
    ppl              exp(nats per token)
    bpb              nats per token / (bytes per token * ln 2)

The same sequences, drawn once from one seed, are scored by both arms. Redrawing per arm would put
sampling noise inside an effect of a few hundredths of a bit, which is the same size.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np


def load_tokens(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise SystemExit(
            f"  no token cache at {path}. Build it with the study's own build_tokens rather than "
            f"re-tokenising here: a corpus tokenised twice by two code paths is two corpora."
        )
    return np.fromfile(path, dtype=np.uint32)


def draw(tokens: np.ndarray, sequences: int, seq: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(tokens) - seq - 1, size=sequences)
    return [tokens[s : s + seq].astype(np.int64).tolist() for s in starts]


def score(engine, prompts: list) -> tuple:
    """Sum the input-token logprobs the server returns, and count the positions they cover."""
    outputs = engine.generate(
        input_ids=prompts,
        sampling_params={"temperature": 0.0, "max_new_tokens": 1},
        return_logprob=True,
        logprob_start_len=0,
    )
    total, counted = 0.0, 0
    for out in outputs:
        for entry in out["meta_info"]["input_token_logprobs"]:
            logprob = entry[0]
            if logprob is None:
                # the first position has nothing predicting it; it is not a scored position
                continue
            total += float(logprob)
            counted += 1
    if counted == 0:
        raise SystemExit(
            "  the server returned no input logprobs, so this measured nothing. Check that "
            "return_logprob and logprob_start_len reached the scheduler."
        )
    return -total, counted


def as_metrics(nats_total: float, positions: int, bytes_per_token: float) -> dict:
    nats = nats_total / positions
    return {
        "positions": positions,
        "nats_per_token": nats,
        "ppl": math.exp(nats),
        "bpb": nats / (bytes_per_token * math.log(2)),
    }


def compare(stock: dict, shifted: dict) -> dict:
    out = {}
    for key in ("bpb", "ppl", "nats_per_token"):
        a, b = stock[key], shifted[key]
        out[key] = {
            "stock": a,
            "shifted": b,
            "absolute": b - a,
            "relative_pct": 100.0 * (b - a) / a,
        }
    return out


def run_arm(model_path: str, shift: int, prompts, mem_fraction: float,
            coverage: str, split: bool) -> tuple:
    import sglang as sgl

    engine = sgl.Engine(
        model_path=model_path,
        tp_size=1,
        mem_fraction_static=mem_fraction,
        disable_cuda_graph=True,
        attention_backend="triton",
        log_level="warning",
        afd_query_shift_layers=shift,
        afd_coverage=coverage,
        afd_split_attention=split,
    )
    try:
        return score(engine, prompts)
    finally:
        engine.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("model", "tokens", "out"):
        ap.add_argument(f"--{name}", required=True)
    for name in ("shift", "sequences", "seq", "seed"):
        ap.add_argument(f"--{name}", type=int, required=True)
    ap.add_argument("--bytes-per-token", type=float, required=True)
    ap.add_argument("--mem-fraction", type=float, required=True)
    ap.add_argument("--coverage", required=True, choices=("all", "softmax"))
    a = ap.parse_args()

    tokens = load_tokens(a.tokens)
    prompts = draw(tokens, a.sequences, a.seq, a.seed)
    print(f"  {len(tokens)} cached token(s); scoring {a.sequences} sequence(s) of {a.seq}",
          flush=True)

    arms = {}
    for name, shift, split in (
        ("stock", 0, False),
        ("shifted", a.shift, False),
        ("shifted_split", a.shift, True),
    ):
        nats, positions = run_arm(a.model, shift, prompts, a.mem_fraction, a.coverage, split)
        arms[name] = as_metrics(nats, positions, a.bytes_per_token)
        print(f"  {name:14s} shift={shift} split={str(split):5s}  "
              f"bpb {arms[name]['bpb']:.4f}  ppl {arms[name]['ppl']:.3f}  "
              f"({arms[name]['positions']} positions)", flush=True)

    counted = {name: arm["positions"] for name, arm in arms.items()}
    if len(set(counted.values())) != 1:
        raise SystemExit(
            f"  the arms scored different numbers of positions ({counted}); they are not one "
            f"measurement"
        )

    stock, shifted = arms["stock"], arms["shifted"]
    delta = compare(stock, shifted)
    split_delta = compare(shifted, arms["shifted_split"])
    for key in ("bpb", "ppl"):
        d = delta[key]
        print(f"  read point  {key}: {d['stock']:.4f} -> {d['shifted']:.4f}   "
              f"{d['absolute']:+.4f}  ({d['relative_pct']:+.2f}%)", flush=True)
    for key in ("bpb", "ppl"):
        d = split_delta[key]
        print(f"  schedule    {key}: {d['stock']:.4f} -> {d['shifted']:.4f}   "
              f"{d['absolute']:+.4f}  ({d['relative_pct']:+.2f}%)", flush=True)
    print("  the schedule's delta is a prefill number and the partition is decode-only, so zero "
          "here says the wiring left prefill alone -- not that the partition is free", flush=True)

    out = {
        "model": a.model,
        "config": vars(a),
        "arms": arms,
        "stock": stock,
        "shifted": shifted,
        "delta": delta,
        "schedule_delta_prefill_only": split_delta,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
