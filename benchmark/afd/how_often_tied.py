"""How often are the top two candidates close enough that a reduction order decides the token?

`stress.py` forgives a greedy flip under concurrency when the top two logprobs at that position
were within `A_TIE`, and `why_it_diverged.py` shows that the one flip anyone has examined was tied
at 0.0000. Both are about a position that already flipped. Neither says how often such a position
occurs, and without that the sentence "the cost is small" is being said without its number -- the
submission report says so about itself.

This measures the base rate: over real prompts, what fraction of decoded positions have a top-two
gap small enough that a reassociated reduction could turn the argmax over.

    It needs ONE server and does not need AFD. A tie is a property of the model and the prompt,
    not of the arrangement -- the arrangement only supplies a different reduction order to
    exploit one. Run it against whatever is up; say in the writeup which that was, because the
    model decides the answer and the arrangement does not.

    It reports POSITIONS, not prompts, and prints the count beside every rate. A fraction with no
    denominator is how "one position, one round" became a claim about the arrangement in the
    first place.

The threshold is imported from `stress.py` rather than repeated, so the number that forgives a
flip and the number that counts one cannot drift apart.

    python benchmark/afd/how_often_tied.py --host http://127.0.0.1:31002 --prompts prompts.txt
    python benchmark/afd/how_often_tied.py --host ... --self-check
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from stress import A_TIE

# Bands to report, in logprob units. A_TIE is inserted at its place among them by `_bands`, so it
# is visible as a row rather than as a separate summary line that has to be read against the rest.
BANDS = (0.0, 0.001, 0.01, 0.1, 0.5, 1.0)

# Prompts to use when none are given. Deliberately plain and deliberately few: this is a default
# that the caller should replace, and a default that produced a confident-looking number over
# hand-picked prompts would be worse than one that is obviously a sample.
DEFAULT_PROMPTS = (
    "The capital of France is",
    "A short definition of entropy is",
    "Count from one to five:",
    "The three primary colours are",
    "Water boils at",
)


def generate(host: str, prompt: str, *, tokens: int) -> dict:
    """Greedy decode with the top two logprobs at every output position."""
    body = {
        "text": prompt,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": tokens},
        "return_logprob": True,
        "top_logprobs_num": 2,
    }
    request = urllib.request.Request(
        host.rstrip("/") + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        answer = json.loads(response.read())
    return answer[0] if isinstance(answer, list) else answer


def gaps_of(answer: dict) -> list:
    """The top-two logprob gap at each output position.

    A position whose second candidate the server did not return is skipped rather than counted as
    a wide gap: it is a position this measurement has nothing to say about, and rolling it into
    the denominator would push every rate down by however many of them there are.
    """
    out = []
    for top in answer["meta_info"]["output_top_logprobs"]:
        if top is None or len(top) < 2:
            continue
        out.append(float(top[0][0]) - float(top[1][0]))
    return out


def _bands() -> tuple:
    return tuple(sorted(set(BANDS) | {A_TIE}))


def report(gaps: list) -> int:
    """Print the distribution. Returns the number of positions at or below `A_TIE`."""
    total = len(gaps)
    if not total:
        print("  no position had two candidates; nothing to report")
        return 0
    print(f"  {total} positions, gap between the top two logprobs")
    at_tie = 0
    for edge in _bands():
        under = sum(1 for g in gaps if g <= edge)
        mark = (
            "  <- A_TIE, the threshold stress.py forgives a flip under"
            if edge == A_TIE
            else ""
        )
        print(f"    <= {edge:<6.3f} {under:6d}  {100.0 * under / total:6.2f}%{mark}")
        if edge == A_TIE:
            at_tie = under
    print(f"    max      {max(gaps):6.3f}")
    return at_tie


def self_check() -> int:
    """Run the arithmetic against answers whose result is known, and watch it fail first.

    A check that has only ever been seen green is a check nobody has reason to trust: it may be
    reading the wrong key, or a threshold it can never reach, and it would print exactly what it
    prints now. So this feeds `gaps_of` three shapes with known answers, and one of them is the
    shape that would silently inflate every rate -- a position with one candidate.

    It was watched failing before it was believed, twice, and both are worth repeating if this
    file is edited:

        drop the `len(top) < 2` guard in `gaps_of`      IndexError, exit 1
        set A_TIE to something unreachable (-1.0)       the rate is 0, correctly, and the band
                                                       table prints `<= -1.000` beside the
                                                       others, so the unreachable threshold is
                                                       visible instead of reading as "no ties"

    The second is why `_bands` folds A_TIE into the table rather than reporting it on a line of
    its own. A threshold that cannot be reached and a measurement that found nothing print the
    same number; only the neighbouring rows tell them apart.
    """

    def answer(tops):
        return {"meta_info": {"output_top_logprobs": tops}}

    cases = [
        ("two candidates, clear", [[(-0.5, 1, "a"), (-2.5, 2, "b")]], [2.0]),
        ("two candidates, tied", [[(-1.0, 1, "a"), (-1.0, 2, "b")]], [0.0]),
        ("one candidate, skipped", [[(-1.0, 1, "a")]], []),
        ("no candidates, skipped", [None], []),
    ]
    problems = []
    for name, tops, want in cases:
        got = gaps_of(answer(tops))
        if got != want:
            problems.append(f"{name}: expected {want}, got {got}")
    # And the denominator itself: a skipped position must not reach `report`.
    mixed = gaps_of(answer([[(-1.0, 1, "a"), (-1.0, 2, "b")], [(-1.0, 1, "a")]]))
    if len(mixed) != 1:
        problems.append(f"a one-candidate position reached the denominator: {mixed}")
    for problem in problems:
        print("  SELF-CHECK FAILED:", problem)
    print(
        "  self-check: 5 cases" + (" -- FAILED" if problems else " -- all as expected")
    )
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", help="a server serving /generate, e.g. http://127.0.0.1:31002"
    )
    parser.add_argument(
        "--prompts", help="a file of prompts, one a line; omit for the sample"
    )
    parser.add_argument("--tokens", type=int, default=64, help="output tokens a prompt")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="check the arithmetic against known answers and exit, touching no server",
    )
    args = parser.parse_args()

    if args.self_check:
        return self_check()
    if not args.host:
        parser.error("--host is required (or --self-check)")

    if args.prompts:
        with open(args.prompts) as handle:
            prompts = [line.strip() for line in handle if line.strip()]
    else:
        prompts = list(DEFAULT_PROMPTS)
        print("  using the built-in sample of 5 prompts; pass --prompts for a real one")

    gaps = []
    for prompt in prompts:
        answer = generate(args.host, prompt, tokens=args.tokens)
        gaps += gaps_of(answer)
    print(
        f"  {len(prompts)} prompts, {args.tokens} tokens asked of each, against {args.host}"
    )
    at_tie = report(gaps)
    print()
    print(
        f"  {at_tie} of {len(gaps)} positions sit within {A_TIE} of a tie. That is the rate at "
        f"which\n  a reassociated reduction has a token to take, on THESE prompts and this model."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
