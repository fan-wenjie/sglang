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
        [--via 'ssh ...'] [--prompt "..."] [--repeats 2]

EVERY PROMPT IS ASKED TWICE, and that is not for averaging. A recurrent state has no length --
whatever is in its buffer IS the history -- so a slot handed to a new request without being
cleared gives it the previous request's memory, fluently. The FIRST request through a fresh pool
is always right. This instrument reported "not identical, relative 1.28" for three arrangements in
a row and none of those numbers was about the arrangement named; they were second requests through
one uncleared slot. Each side is now compared against its own earlier answer, and a side that
disagrees with itself voids the verdict rather than colouring it.

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
    "pool": ("span(s) a decode step", "the per-layer linear cut is served here"),
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
    p.add_argument("--repeats", type=int, default=2, help="how many times each prompt is asked. "
                                                          "TWO is the floor and the default; see "
                                                          "below")
    args = p.parse_args()

    if args.host_log:
        confirm_installed(args.host_log, "host", args.via)
    if args.pool_log:
        confirm_installed(args.pool_log, "pool", None)
    if args.repeats < 2:
        print("  WARNING: one pass a prompt cannot see a slot handed on with the previous "
              "request's state still in it, which is the fault this instrument missed for days")

    prompts = args.prompt or [
        "The capital of France is",
        "The",
        "Explain why the sky is blue in one sentence.",
    ]
    worst = 0.0
    unstable = []
    for prompt in prompts:
        print(f"\n  prompt {prompt!r}")
        first = {}
        for turn in range(args.repeats):
            base = ask(args.colocated, prompt, args.tokens, None)
            rung = ask(args.rung, prompt, args.tokens, args.via)
            label = f"pass {turn}"
            where = _where(base["output_ids"], rung["output_ids"])
            print(f"    {label}  tokens {where}")

            # Each side against ITSELF on the earlier pass. This is the null the token comparison
            # cannot provide: a server whose second answer differs from its first has handed a slot
            # on with the previous request's history in it, and BOTH sides could do it -- the
            # colocated one runs sglang's own state machinery and is the reference only for as long
            # as it is stable.
            for side, answer in (("colocated", base), ("rung", rung)):
                previous = first.get(side)
                if previous is None:
                    first[side] = answer["output_ids"]
                elif previous != answer["output_ids"]:
                    unstable.append((prompt, side, turn))
                    print(f"    {label}  {side} DIFFERS FROM ITS OWN PASS 0 -- "
                          f"{_where(previous, answer['output_ids'])}. A slot was reused with "
                          f"state still in it; every number below is about two arrangements only "
                          f"if this line is absent")

            hb = base["meta_info"].get("hidden_states")
            hr = rung["meta_info"].get("hidden_states")
            if hb is None or hr is None:
                print("    hidden   not returned -- both need --enable-return-hidden-states")
                continue
            rows_b, rows_r = flatten(hb), flatten(hr)
            for i, (x, y) in enumerate(zip(rows_b, rows_r)):
                rel, cos = compare(y, x)
                worst = max(worst, rel)
                print(f"    {label}  token {i:<3} relative {rel:.6g}   cosine {cos:+.6f}")

    print(f"\n  worst relative difference over all prompts, passes and tokens: {worst:.6g}")
    if unstable:
        print(f"  REFUSED as a verdict: {len(unstable)} pass(es) disagreed with the same server's "
              f"own earlier answer to the same prompt: {unstable}. That is one arrangement "
              f"disagreeing with itself, and no comparison between two of them means anything "
              f"until it is gone.")
        return 1
    print(f"  every side answered the same on all {args.repeats} passes, so the slots were clean")
    return 0


def _where(a: list, b: list) -> str:
    if a == b:
        return "identical"
    return next((f"part at token {i}" for i, (x, y) in enumerate(zip(a, b)) if x != y),
                "differ in length")



if __name__ == "__main__":
    raise SystemExit(main())
