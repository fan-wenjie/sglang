"""One rung's verdict against colocated: the tokens AND how far the hidden state drifted.

Token identity is a yes or a no, and a rung that happens to agree on eight greedy tokens has said
less than it looks like it has -- batch composition alone parts the same model's greedy output on
3 of 4 prompts. The hidden state says HOW FAR apart the two arrangements are, which is what
localises a fault that has not yet reached the argmax.

The quantity is the FINAL hidden state, taken through `--enable-return-hidden-states` on both
servers. One well-defined tensor per token, produced by each side's own complete forward, with no
layout convention to get wrong. That matters more than depth here: ten times in this search a
comparison of some intermediate turned out to measure a different quantity, a different row, or a
different occasion, and every one of those was a per-layer probe. This one cannot be, because
neither side is asked for anything but its answer.

    python benchmark/afd/rung_verdict.py --colocated 127.0.0.1:31000 --rung HOST:31001 \
        [--via 'ssh ...'] [--prompt "..."]

Reported per prompt:

    tokens      identical, or the first index where they part
    relative    |h_rung - h_colocated| / |h_colocated|, over the final hidden state
    cosine      the direction, which separates "scaled" from "turned"
    per token   the same two, so a drift that grows with depth of generation is visible

A rung that is token-identical AND sits at bfloat16 rounding is the one to build the next rung on.
A rung that is token-identical with a relative difference of a few percent has a fault that has
not surfaced yet, and building on it would bury it.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess


def ask(endpoint: str, prompt: str, tokens: int, via: str | None) -> dict:
    body = json.dumps({
        "text": prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": tokens},
        "return_hidden_states": True,
    })
    curl = ["curl", "-s", "-m", "240", f"http://{endpoint}/generate",
            "-H", "Content-Type: application/json", "-d", body]
    argv = (via.split() + [" ".join(f"'{c}'" if " " in c else c for c in curl)]) if via else curl
    out = subprocess.run(argv, capture_output=True, text=True).stdout
    return json.loads(out)


def flatten(hidden) -> list[list[float]]:
    """One row a token, whatever nesting the server used.

    The server returns a token's state nested more deeply than one level -- the first version of
    this assumed two and raised `float() argument must be ... not 'list'`. Descend until the
    leaves are numbers, then take the last axis as the state and everything above it as tokens.
    """
    def depth(x):
        return 1 + depth(x[0]) if isinstance(x, list) and x and isinstance(x[0], list) else 1

    def rows(x):
        if depth(x) == 1:
            return [[float(v) for v in x]]
        out = []
        for part in x:
            out.extend(rows(part))
        return out

    return rows(hidden)


def compare(a: list[float], b: list[float]) -> tuple[float, float]:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    diff = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    dot = sum(x * y for x, y in zip(a, b))
    return diff / (nb + 1e-9), dot / (na * nb + 1e-9)


# One line a side a rung. A rung whose installation cannot be confirmed reads the same whether it
# ran or not, and that has happened three times in two days.
INSTALLED = {
    "host": ("the group cut is installed", "the per-layer linear cut is installed"),
    "pool": ("span(s) a decode step",),
}


def confirm_installed(log: str, side: str, via: str | None) -> None:
    """Refuse to report a verdict until the log says the arm was INSTALLED.

    "The arm is available" is a REGISTRATION message and it is not the same thing. sglang spawns
    the scheduler, so a package imported by the argument check registers in the parent and leaves
    the child's registry empty: both ends logged the arm as available, neither installed it, the
    standard arrangement served, and the verdict came back token-identical with a hidden-state
    drift of one percent. That reads as "the cut works" and it means "the cut did not run".

    So the number is gated on the install line rather than on the availability line, and a missing
    log is a refusal rather than a warning. A verdict that cannot say which arrangement produced
    it is worse than no verdict.
    """
    for needle in INSTALLED[side]:
        cmd = ["grep", "-cF", needle, log]
        argv = (via.split() + [" ".join(f"'{c}'" if " " in c else c for c in cmd)]) if via else cmd
        got = subprocess.run(argv, capture_output=True, text=True).stdout.strip()
        if got.isdigit() and int(got) > 0:
            print(f"  {side} log confirms the arrangement is installed ({needle!r} x{got})")
            return
    needle = " or ".join(repr(n) for n in INSTALLED[side])
    raise SystemExit(
        f"REFUSED: {log} does not contain {needle!r}, so the {side} is not running the "
        f"arrangement this verdict would be attributed to. Check for 'arm is available' -- that "
        f"is registration, not installation, and it is what fooled this once already."
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--colocated", required=True)
    p.add_argument("--rung", required=True)
    p.add_argument("--host-log", default=None, help="the rung host's log; the verdict is refused "
                                                    "unless it shows the arrangement installed")
    p.add_argument("--pool-log", default=None)
    p.add_argument("--via", default=None, help="a command that runs curl where the rung lives")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--prompt", action="append", default=None)
    args = p.parse_args()

    if args.host_log:
        confirm_installed(args.host_log, "host", args.via)
    if args.pool_log:
        confirm_installed(args.pool_log, "pool", None)

    prompts = args.prompt or [
        "The capital of France is",
        "The",
        "Explain why the sky is blue in one sentence.",
    ]
    worst = 0.0
    for prompt in prompts:
        base = ask(args.colocated, prompt, args.tokens, None)
        rung = ask(args.rung, prompt, args.tokens, args.via)
        same = base["output_ids"] == rung["output_ids"]
        where = "identical" if same else next(
            (f"part at token {i}" for i, (x, y) in
             enumerate(zip(base["output_ids"], rung["output_ids"])) if x != y),
            "differ in length")
        print(f"\n  prompt {prompt!r}")
        print(f"    tokens   {where}")
        hb = base["meta_info"].get("hidden_states")
        hr = rung["meta_info"].get("hidden_states")
        if hb is None or hr is None:
            print("    hidden   not returned -- both servers need --enable-return-hidden-states")
            continue
        rows_b, rows_r = flatten(hb), flatten(hr)
        for i, (x, y) in enumerate(zip(rows_b, rows_r)):
            rel, cos = compare(y, x)
            worst = max(worst, rel)
            print(f"    token {i:<3} relative {rel:.6g}   cosine {cos:+.6f}")

    print(f"\n  worst relative difference over all prompts and tokens: {worst:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
