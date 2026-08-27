"""Four things that only break under load, run against a host that is serving.

Every case here is a failure this arrangement has actually had, or the shape of one it is known
to be exposed to. None of them shows up in a single-request smoke test, which is what every
verification of this cut has been so far:

    concurrency     `_scan` refused two requests in one prefill chunk and killed the host at
                    concurrency 2. `_refuse_split_runs` fixed that ONE shape. Nothing has ever
                    put more than two through it.
    slot reuse      the SECOND request through a slot is the test. A recurrent state has no
                    length, so a slot handed to a new request still holds the old one's history
                    unless something clears it, and the answer stays fluent either way.
    abort           a request withdrawn mid-decode has to give its slot back. #59 built that
                    path; it has never run while other requests were using the pool.
    pool restart    the host must survive the pool going away and say so -- `PoolClosed`, named,
                    not a socket error and not a hang.

What this does NOT measure is latency. Run this way the wire is loopback, so every number about
time is about this machine's memory bus. The pass/fail here is behavioural.

    python benchmark/afd/stress.py --host http://127.0.0.1:31002 [--case all]
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import sys
import time
import urllib.error
import urllib.request

PROMPTS = [
    "The capital of France is",
    "Count from one to five:",
    "A short definition of entropy is",
    "The three primary colours are",
]


def generate(
    host: str,
    prompt: str,
    *,
    tokens: int = 24,
    rid: str | None = None,
    logprobs: bool = False,
) -> dict:
    """One completion, greedy, so a repeat of the same prompt must give the same text."""
    body = {
        "text": prompt,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": tokens},
    }
    if logprobs:
        body["return_logprob"] = True
        body["top_logprobs_num"] = 2
    if rid is not None:
        body["rid"] = rid
    request = urllib.request.Request(
        f"{host}/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as reply:
        return json.loads(reply.read())


def _text(answer) -> str:
    return answer["text"] if isinstance(answer, dict) else answer[0]["text"]


# How close the top two logprobs have to be for a flip between them to count as arithmetic
# rather than as the wrong history. Measured, not chosen: the one flip seen so far was tied at
# 0.0000 -- equal to every digit the server returns -- and a reduction reassociated by batch
# composition moves a logprob by that much and no further. See `why_it_diverged.py`.
A_TIE = 0.05


def case_concurrency(host: str, *, width: int = 4, rounds: int = 3) -> list:
    """Several requests in flight at once, and the same answers EXCEPT where the choice was tied.

    This asked for byte-identical greedy output at first, and that is a property this arrangement
    cannot have. Values cross the wire in bf16 and come back, so the reductions differ from the
    colocated ones; where the top two logprobs are exactly equal, which happens, the argmax flips
    on nothing. Demanding equality made every tie a red test and buried the failure worth seeing.

    What is worth seeing is a flip where the model was NOT torn: that means the row's history came
    from the wrong request, and the answer is then confidently about something else. So a
    divergence is forgiven only when the one-at-a-time run was within `A_TIE` of a tie at the
    position that diverged.

    Still checked against the one-at-a-time answer taken first, because two concurrent runs
    agreeing with each other proves only that they are wrong the same way.
    """
    problems = []
    alone = {p: generate(host, p, logprobs=True) for p in PROMPTS}
    for round_index in range(rounds):
        with futures.ThreadPoolExecutor(max_workers=width) as pool:
            got = list(
                pool.map(lambda p: (p, generate(host, p, logprobs=True)), PROMPTS)
            )
        for prompt, answer in got:
            if _text(answer) == _text(alone[prompt]):
                continue
            position, gap = _first_flip(alone[prompt], answer)
            if position is None:
                continue
            if gap <= A_TIE:
                continue
            problems.append(
                f"round {round_index}: {prompt!r} flipped at output position {position} where "
                f"the top two were {gap:.4f} apart -- too far to be a reduction order, so the "
                f"history it read was not its own\n      alone: {_text(alone[prompt])!r}"
                f"\n      together: {_text(answer)!r}"
            )
    return problems


def _first_flip(alone: dict, together: dict):
    """(position, the alone run's top-two gap there) for the first token that differs."""
    left = [int(t) for _, t, _ in alone["meta_info"]["output_token_logprobs"]]
    right = [int(t) for _, t, _ in together["meta_info"]["output_token_logprobs"]]
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            top = alone["meta_info"]["output_top_logprobs"][index][:2]
            gap = float(top[0][0]) - float(top[1][0]) if len(top) > 1 else float("inf")
            return index, gap
    return None, 0.0


def case_slot_reuse(host: str, *, passes: int = 6) -> list:
    """The same prompt over and over, each time through whatever slot is free.

    A slot's recurrent state has no length. If it is handed to a new request without being
    cleared, the new request reads the old one's history -- and the reply is still fluent, which
    is why this needs an equality check rather than an eyeball. The first answer is the reference
    and every later one must match it exactly.
    """
    problems = []
    reference = _text(generate(host, PROMPTS[0]))
    for index in range(1, passes):
        # a different prompt in between, so the slot is genuinely reused rather than kept
        generate(host, PROMPTS[(index % (len(PROMPTS) - 1)) + 1], tokens=12)
        text = _text(generate(host, PROMPTS[0]))
        if text != reference:
            problems.append(
                f"pass {index}: the same prompt decoded differently after the slot was reused\n"
                f"      first: {reference!r}\n      now:   {text!r}"
            )
    return problems


def case_abort(host: str, *, width: int = 3) -> list:
    """Withdraw a long request while others are decoding, then check the survivors.

    The slot the aborted request held has to go back, and the requests beside it must be
    undisturbed -- an abort that frees the wrong slot takes a live request's history with it.
    """
    problems = []
    alone = _text(generate(host, PROMPTS[0]))
    long_id = "stress-abort"

    with futures.ThreadPoolExecutor(max_workers=width + 1) as pool:
        pool.submit(generate, host, PROMPTS[2], tokens=512, rid=long_id)
        time.sleep(2.0)
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{host}/abort_request",
                    data=json.dumps({"rid": long_id}).encode(),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=30,
            ).read()
        except urllib.error.HTTPError as failure:
            problems.append(f"the abort endpoint refused: {failure}")
        beside = list(pool.map(lambda p: _text(generate(host, p)), PROMPTS[:width]))

    if beside and beside[0] != alone:
        problems.append(
            f"a request beside the aborted one decoded differently\n"
            f"      alone: {alone!r}\n      beside: {beside[0]!r}"
        )
    # and the host is still serving afterwards
    if _text(generate(host, PROMPTS[0])) != alone:
        problems.append(
            "after the abort the host no longer decodes the reference prompt"
        )
    return problems


def case_pool_gone(host: str) -> list:
    """Only meaningful when the pool is stopped BETWEEN the two halves of this.

    Not run by default: it needs the operator to kill the pool, and killing it from here would
    leave the host talking to nothing for every later case. `--case pool_gone` runs it and
    expects the pool to be down already; what it asserts is that the failure is NAMED --
    `PoolClosed` reaching the client -- rather than a socket error or a wait that never ends.
    """
    try:
        answer = _text(generate(host, PROMPTS[0]))
    except urllib.error.HTTPError as failure:
        detail = failure.read().decode(errors="replace")
        if "PoolClosed" in detail or "pool" in detail.lower():
            return []
        return [
            f"the pool is down and the host said something else entirely: {detail[:400]}"
        ]
    except Exception as failure:  # noqa: BLE001 -- the point is to see what it IS
        return [
            f"the pool is down and the host raised {type(failure).__name__}: {failure}"
        ]
    return [f"the pool is down and the host answered anyway: {answer!r}"]


CASES = {
    "concurrency": case_concurrency,
    "slot_reuse": case_slot_reuse,
    "abort": case_abort,
    "pool_gone": case_pool_gone,
}
BY_DEFAULT = ("concurrency", "slot_reuse", "abort")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://127.0.0.1:31002")
    parser.add_argument("--case", default="default", choices=("default", "all", *CASES))
    args = parser.parse_args()

    chosen = (
        BY_DEFAULT
        if args.case == "default"
        else (tuple(CASES) if args.case == "all" else (args.case,))
    )
    failed = 0
    for name in chosen:
        began = time.perf_counter()
        try:
            problems = CASES[name](args.host)
        except Exception as failure:  # noqa: BLE001 -- a case that dies is a result
            problems = [f"the case itself raised {type(failure).__name__}: {failure}"]
        took = time.perf_counter() - began
        print(f"{name:12s} {'PASS' if not problems else 'FAIL'}  {took:6.1f}s")
        for problem in problems:
            print(f"    {problem}")
        failed += bool(problems)
    print(f"\n{len(chosen) - failed}/{len(chosen)} cases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
