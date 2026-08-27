"""At the first token that differs under concurrency, was the choice close or clear?

The group cut decodes a different token depending on what else is in the batch; colocated, on the
same build with the same sixteen recurrent-state slots, does not. Two mechanisms would both
produce that and they need different fixes:

    arithmetic   batch composition changes the order of a reduction, the top two logits were
                 nearly tied, and the argmax flipped. The history was right. This is the benign
                 half -- annoying, bounded, and the same thing any batched server can show, except
                 that colocated did NOT show it here, so it would still be ours
    history      a row's reading was fetched for the wrong request. Then the distribution at that
                 position is not a perturbed version of the right one, it is a different
                 distribution, and the winner usually wins clearly

The measurement that separates them is the GAP between the top two logprobs at the first position
where the two runs disagree. A near-tie says arithmetic; a clear gap says the model was confidently
answering a different question.

This asks the server for `top_logprobs_num=2` and does the comparison at that one position. It
does not average over positions: an average would mix the position that flipped with hundreds that
did not, and the flip is the whole event.

    python benchmark/afd/why_it_diverged.py --host http://127.0.0.1:31002 [--width 4] [--rounds 3]
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import sys
import urllib.request

PROMPT = "A short definition of entropy is"
BESIDE = [
    "The capital of France is",
    "Count from one to five:",
    "The three primary colours are",
]


def generate(
    host: str, prompt: str, *, tokens: int = 24, logprobs: bool = False
) -> dict:
    body = {
        "text": prompt,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": tokens},
    }
    if logprobs:
        body["return_logprob"] = True
        body["top_logprobs_num"] = 2
    request = urllib.request.Request(
        f"{host}/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as reply:
        return json.loads(reply.read())


def _chosen(answer: dict) -> list:
    """(token_id, logprob) for each generated position."""
    meta = answer["meta_info"]
    return [(int(t), float(lp)) for lp, t, _ in meta["output_token_logprobs"]]


def _top_two(answer: dict) -> list:
    """[(id, logprob), (id, logprob)] for each generated position."""
    meta = answer["meta_info"]
    out = []
    for position in meta["output_top_logprobs"]:
        out.append([(int(t), float(lp)) for lp, t, _ in position[:2]])
    return out


def first_difference(a: list, b: list) -> int | None:
    for index, (left, right) in enumerate(zip(a, b)):
        if left[0] != right[0]:
            return index
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://127.0.0.1:31002")
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    print("alone ...")
    alone = generate(args.host, PROMPT, logprobs=True)

    for round_index in range(args.rounds):
        print(f"concurrency {args.width}, round {round_index} ...")
        with futures.ThreadPoolExecutor(max_workers=args.width) as pool:
            jobs = [pool.submit(generate, args.host, PROMPT, logprobs=True)]
            jobs += [
                pool.submit(generate, args.host, p) for p in BESIDE[: args.width - 1]
            ]
            together = jobs[0].result()
            for job in jobs[1:]:
                job.result()

        position = first_difference(_chosen(alone), _chosen(together))
        if position is None:
            print("  no difference this round")
            continue

        print(f"  first difference at output position {position}")
        for label, answer in (("alone", alone), ("together", together)):
            pair = _top_two(answer)[position]
            gap = pair[0][1] - pair[1][1]
            print(
                f"    {label:9s} chose {pair[0][0]:>7} at {pair[0][1]:8.4f}, "
                f"runner-up {pair[1][0]:>7} at {pair[1][1]:8.4f}, gap {gap:.4f}"
            )
        # The reading. A gap of a few thousandths is a coin the reduction order flipped; a gap of
        # a nat or more is not something reassociating a sum does.
        gap_alone = (lambda p: p[0][1] - p[1][1])(_top_two(alone)[position])
        print(
            "    -> near-tie, consistent with reduction order"
            if gap_alone < 0.05
            else "    -> a clear winner alone, so the concurrent run was answering differently"
        )
        return 0

    print("nothing diverged; run it again or raise --rounds")
    return 1


if __name__ == "__main__":
    sys.exit(main())
