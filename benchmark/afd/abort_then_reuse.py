"""A request is aborted mid-generation; does the next one inherit its recurrent state?

This walks the one path in the slot bookkeeping that nothing has ever walked. A request that ENDS
normally and one that is ABORTED leave the same thing behind -- a slot holding a compressed
history -- and the design says neither is cleaned up at the end: the next request to land on that
slot clears it, because a request that crashes never reaches its own ending. That is a deliberate
choice (see `slot_reset`) and it has never been exercised from the abort side.

What makes it worth a script rather than an argument: the failure would be silent. A slot handed
on with somebody else's history produces fluent text conditioned on a prompt the caller never
sent, and no status code, no log line and no assertion in the server says so. It cost this project
several days from the other direction, when the ending was never signalled at all.

    python benchmark/afd/abort_then_reuse.py --colocated 127.0.0.1:31000 --host HOST:31001 \
        [--via 'ssh ...'] [--rounds 3]

The comparison is against the colocated model on the same prompt, which is the only reference that
cannot itself be contaminated: it holds no AFD slot table. A run is a PASS when the answer after an
abort is byte-identical to the answer without one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time


def call(endpoint: str, path: str, body: dict, via: str | None, timeout: int = 240) -> dict:
    curl = ["curl", "-s", "-m", str(timeout), f"http://{endpoint}{path}",
            "-H", "Content-Type: application/json", "-d", json.dumps(body)]
    argv = (via.split() + [" ".join(f"'{c}'" if " " in c else c for c in curl)]) if via else curl
    out = subprocess.run(argv, capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"raw": out}


def generate(endpoint: str, prompt: str, tokens: int, via: str | None, rid: str | None = None):
    body = {"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": tokens}}
    if rid is not None:
        body["rid"] = rid
    return call(endpoint, "/generate", body, via)


def abort_after(endpoint: str, rid: str, delay: float, via: str | None) -> None:
    """Abort from another thread while the generation is still running."""
    time.sleep(delay)
    call(endpoint, "/abort_request", {"rid": rid}, via, timeout=20)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--colocated", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--via", default=None, help="a command that runs curl where the host lives")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--long-prompt", default="Write a long essay about the history of the sea.")
    p.add_argument("--tokens", type=int, default=12)
    args = p.parse_args()

    reference = generate(args.colocated, args.prompt, args.tokens, None)["text"]
    clean = generate(args.host, args.prompt, args.tokens, args.via)["text"]
    print(f"  colocated        {reference!r}")
    print(f"  host, no abort   {clean!r}   {'same' if clean == reference else 'DIFFERS'}")

    failures = 0
    for turn in range(args.rounds):
        rid = f"abort-me-{turn}"
        killer = threading.Thread(
            target=abort_after, args=(args.host, rid, 1.5, args.via), daemon=True)
        killer.start()
        # a long generation, killed 1.5s in: it will have advanced its slot's state by then
        generate(args.host, args.long_prompt, 400, args.via, rid=rid)
        killer.join()

        after = generate(args.host, args.prompt, args.tokens, args.via)["text"]
        same = after == reference
        failures += 0 if same else 1
        print(f"  after abort {turn}    {after!r}   {'same' if same else 'DIFFERS'}")

    if failures:
        print(f"\n  {failures} of {args.rounds} answers after an abort differ from the same "
              f"prompt's answer without one. A slot was handed on with the aborted request's "
              f"history still in it -- the output stays fluent and nothing else reports it.")
        return 1
    print(f"\n  every answer after an abort matches the clean one: the slot the aborted request "
          f"left was cleared before the next request used it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
