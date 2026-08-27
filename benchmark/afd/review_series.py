"""Split this branch into five series a reviewer can take one at a time, and prove the split.

Push-back item 1 says this is one branch of ~12k lines and should be a series. It is now five, and
this script is both the definition of the split and its proof -- so that "five reviewable series"
is a thing anyone can regenerate and check rather than a claim in a document.

## Why a partition of FILES rather than a rebase of commits

The commits interleave: a protocol change, then a pool fix, then a report, then the protocol
again. Rebasing them into topic order would mean rewriting each one's content, and the result
would be a set of commits nobody ever ran. The branch is instead almost purely ADDITIVE -- one
deletion in the whole diff -- and an additive diff partitions exactly by file. So each series is
"these files, at their final content", which is what a reviewer reads anyway.

The counts are deliberately NOT written here. They were, and they drifted: this file said 48
commits and 76 files where the report said 68 and 80, the report's own paragraph then said 48
again two lines later, and the branch was at neither. A count maintained by hand in prose is
wrong by the next commit and says nothing about when it went wrong, so `check_partition` below
prints the live ones -- files, insertions, deletions and commits, read from the repository at the
moment anyone asks.

The cost of that choice, stated because it is real: a series shows a file's END state, not how it
got there. Someone who wants the intermediate reasoning still has the commits. What the series
buy is a reading ORDER with a dependency guarantee, which the commits do not have.

## The three properties, all machine-checked below

    partition   the five file sets are disjoint and their union is exactly the branch's diff.
                Not a sampling and not an overlap -- every file lands in exactly one series
    composition the five applied in order give a tree BYTE-IDENTICAL to afd/main. This is the
                one that makes the split trustworthy: nothing was dropped, edited or reordered
                into a different meaning on the way
    closure     at each series' tip, every module added so far imports, and that series' own
                tests pass. This is what "reviewable on its own" has to mean -- a series that
                needs a later one to import is not a series, it is a slice

The order is forced by the import graph, not chosen for narrative. `arms` precedes `pool_client`
because `pool_client` imports it; `slotted_kv` precedes `pool_attention` for the same reason. The
one place taste entered is that the flags and the refusals come FIRST -- a reviewer meets what an
operator can ask for, and what is refused at startup, before meeting any mechanism.

Run: python benchmark/afd/review_series.py [--base SHA] [--keep]
"""

from __future__ import annotations

import argparse
import subprocess
import sys

A = "python/sglang/srt/afd/"
T = "test/registered/unit/afd/"

# Where afd/main forks from upstream. Passed rather than computed so a re-run on a rebased branch
# fails loudly instead of silently comparing against a different fork point.
DEFAULT_BASE = "d8433868ce8f0e525648426c13c8df9e4f798af4"

# The five commit messages, in full. They live HERE and not in the commits because `build` rewrites
# the commits every run: a body added with `git commit --amend` survives until the next person
# regenerates the split, and then it is gone with no trace that it existed. A series whose message
# says only its title is a series a reviewer has to reverse-engineer, which is the opposite of what
# the split is for.
BODY_1 = """First of five. Read in this order; the order is forced by the import graph, not
chosen to tell a story.

Attention and feed-forward on different machines. The host runs every attention
and owns everything belonging to a request -- the KV cache, the linear layers'
recurrent states. The pool holds the static weights and answers feed-forward
calls for whoever asks, holding nothing between them. The split is by STATE, not
by cost: a stateless pool need not be reserved for a request between that
request's own calls.

This series is what an operator may ask for and what is refused before a model
loads. It comes first because everything below is unreachable without it, and
because the refusals say more about the arrangement's limits than the code does.
`compatibility.py` refuses tensor and pipeline parallelism, dp attention,
speculative decoding, LoRA, multimodal and decode CUDA graphs AT STARTUP, each
with the reason rather than the flag -- unsupported combinations otherwise fail
the same silent way, with a server that starts and tokens that are fluent.

`arms.py` is the registry a derived line announces itself through, so nothing
under `srt/afd` ever names a derived package. It holds no arm of its own.

Seven flags, in three groups. Three say who a process is and how the two ends
find each other: --afd-mode, --afd-pool-addr, --afd-bootstrap-port. Two size a
departure: --afd-min-batch, --afd-max-wait-ms. Two move the line through the
model: --afd-pool-attention, --afd-pool-linear.

There is no flag for the attention partition here. `split_attention.py` carries
the partition itself, and it is reached from its own test rather than from a
forward pass: the code that consults it on real traffic is the derived line's,
and this branch declares no switch it does not implement.
"""

BODY_2 = """Second of five, and it depends on nothing above it.

The frame format, the socket that carries it, how the two ends find each other,
and the rule deciding which waiting callers ride one departure. A departure is
the batching unit: several callers' rows leave together so the pool reads each
weight once for all of them, which is the whole reason a pool is worth having
rather than a remote procedure call.

`protocol.py` is the one file both ends parse, so an opcode added on one side and
not the other is a framing error rather than a wrong answer. Ops are keyed by
(request, layer, op) because one request at one layer can have more than one
answer in flight and they are not interchangeable.
"""

BODY_3 = """Third of five. Depends on the wire, and on nothing below it.

The half of the split that never moves. A softmax layer holds a KV cache; a
linear-attention layer holds a recurrent state. Both belong to the request, and a
pool holding either could not be released and retaken between one request's own
calls -- which is the property the whole arrangement is for.

A recurrent state is the whole history compressed. It cannot be evicted and
rebuilt from a prefix the way a KV cache can, so the slot table REFUSES rather
than evicting, and where that refusal falls is a number the operator chose
(--max-running-requests) rather than one invented here.

`history_service` answers the pool's callbacks: a read of the state, an update
that is off the critical path, and a scan for a prefill chunk whose tokens read
each other's updates and therefore cannot have the two separated.
"""

BODY_4 = """Fourth of five. Depends on the wire and on the host's state, because a departure
calls BACK to whoever holds the history.

The half that holds every weight and nothing per request. Departures dispatch
through a table, so a different cut adds its own op without editing the server
and an unclaimed opcode is refused by name.

Two things here are load-shaped and were found by pushing on them rather than by
reading:

  * a departure never runs on the thread owning the caller's socket. That thread
    is the socket's only reader, so a callback taken there waits for a message
    only it could deliver -- a deadlock that appears as a watchdog timeout naming
    nothing
  * a departure that fails closes its riders' sockets with shutdown() and not
    just close(). The connection thread is blocked reading the same fd, so it
    holds a reference and close() sends no FIN. Measured both ways: the far end
    had not noticed two seconds later
"""

BODY_5 = """Last of five. `roles.py` knows about every module and is the only thing that does.

The client, the per-layer routing, and the loader machinery that lets the host
never allocate what the pool computes -- `feed-forward built with storage=False`,
which is weights never taken rather than weights given back. Measured on this
branch with no arm installed: the host's avail mem goes 34.30 -> 15.12 GB, so
19.18 GB of weights stay, against 51.05 GB for the whole model on one card.

Docs and benchmarks ride here because they describe the finished arrangement and
nothing earlier can. Among them `review_series.py`, which defines this five-way
split and proves it: the file sets are disjoint and their union is exactly the
branch's diff, the five compose to a byte-identical tree, and at each tip every
module so far imports with that series' own tests green. The per-series counts
are printed by the closure check rather than repeated here, for the reason the
docstring of this file gives.

It earns its keep. Three tools added after the split was first written failed the
partition check immediately; without it they would have composed correctly and
gone unreviewed.

The submission report -- what this buys, what it costs, and what a reviewer
should push back on -- is the pull request's own description rather than a file
in the tree. It is addressed to someone deciding whether to look at this, which
is an audience a merged repository no longer has; `python/sglang/srt/afd/
README.md` is what stays, and it is addressed to whoever has to work on this
afterwards.
"""

SERIES = [
    dict(
        branch="afd/review-1-flags",
        body=BODY_1,
        title="afd: the flags, what they refuse, and where a derived arm announces itself",
        why=(
            "What an operator can ask for and what is refused before a model loads. First "
            "because everything below is unreachable without it, and because the refusals say "
            "more about the arrangement's limits than any amount of the code does."
        ),
        mods=["compatibility", "arms", "mode_select"],
        extra=[
            "python/sglang/srt/server_args.py",
            "python/sglang/srt/arg_groups/afd_hook.py",
            # here and not with the modules that read it: `environ.py` carries every AFD
            # environment variable, and the first reader of one is `compatibility.py` in this
            # same series. A file lands in exactly one series, so it lands in the earliest that
            # needs it -- putting it later would leave series 1 importing a descriptor no tip
            # before it defines, which is what the closure check is for.
            "python/sglang/srt/environ.py",
        ],
        tests=["test_afd_compatibility.py", "test_afd_stands_alone.py"],
    ),
    dict(
        branch="afd/review-2-protocol",
        body=BODY_2,
        title="afd: the wire -- frames, transport, rendezvous, and who boards a departure",
        why=(
            "The frame format, the socket that carries it, how the two ends find each other, "
            "and the rule that decides which waiting callers ride one departure. Depends on "
            "nothing above it."
        ),
        mods=[
            "protocol",
            "transport",
            "rendezvous",
            "seating",
            "boarding",
            "meter",
            "slots",
        ],
        extra=[],
        tests=[
            "test_afd_transport.py",
            "test_afd_rendezvous.py",
            "test_afd_seating.py",
            "test_afd_positions.py",
            "test_afd_split_exactness.py",
            "test_afd_mode_select.py",
        ],
    ),
    dict(
        branch="afd/review-3-state",
        body=BODY_3,
        title="afd: the state the host keeps -- recurrent histories, the KV slots, the scan",
        why=(
            "The half of the split that never moves. Everything here belongs to a request, "
            "which is the whole reason the pool can be stateless and shared."
        ),
        mods=[
            "slotted_kv",
            "linear_state",
            "linear_history",
            "split_read_kernel",
            "history_service",
            "linear_runner",
        ],
        extra=[],
        tests=[
            "test_afd_mix_is_the_models.py",
            "test_afd_linear_history.py",
            "test_afd_split_read_kernel.py",
            "test_afd_prefill_scan.py",
            "test_afd_history_service.py",
        ],
    ),
    dict(
        branch="afd/review-4-pool",
        body=BODY_4,
        title="afd: the pool -- a departure table, the sweeps it parks, the layers it serves",
        why=(
            "The half that holds every weight and nothing per request. The departure table is "
            "the extension point that lets a cut add an op without editing the server."
        ),
        mods=[
            "pool_server",
            "pool_linear",
            "pool_attention",
        ],
        extra=[],
        tests=[
            "test_afd_pool_server_methods.py",
            "test_afd_boarding.py",
            "test_afd_reserved_ops.py",
            "test_afd_update_does_not_block.py",
            "test_afd_departure_registry.py",
            "test_afd_pool_attention.py",
            "test_afd_batched_cache.py",
        ],
    ),
    dict(
        branch="afd/review-5-host",
        body=BODY_5,
        title=(
            "afd: the host side, the composition root, and everything that reads the whole thing"
        ),
        why=(
            "The client, the per-layer routing, the loader's absence machinery, and `roles` -- "
            "which knows about every module and is the only one that does. Docs and benchmarks "
            "ride here because they describe the finished arrangement and nothing else can."
        ),
        mods=[
            "pool_client",
            "slot_reset",
            "layer_kinds",
            "linear_routing",
            "remote_attention",
            "split_attention",
            "absent_ffn",
            "dispatcher",
            "roles",
        ],
        extra=[
            "python/sglang/srt/model_executor/model_runner.py",
            "python/sglang/srt/model_loader/loader.py",
            "python/sglang/srt/afd/README.md",
            "benchmark/afd/README.md",
            "docs/docs/advanced_features/afd_disaggregation.mdx",
            "docs/docs.json",
            "benchmark/afd/astcheck.py",
            "benchmark/afd/crossover.py",
            "benchmark/afd/how_often_tied.py",
            "benchmark/afd/launch_or_read.py",
            "benchmark/afd/pool_amortisation.py",
            "benchmark/afd/review_series.py",
            "benchmark/afd/stress.py",
            "benchmark/afd/catch_hang.sh",
            "benchmark/afd/why_it_diverged.py",
            "benchmark/afd/two_callers.py",
        ],
        tests=[
            "test_afd_absent_weights_load.py",
            "test_afd_arrangement_word.py",
            "test_afd_kept_parameters.py",
            "test_afd_inbound.py",
            "test_afd_dispatcher.py",
            "test_afd_partition_indices.py",
            "test_afd_handshake.py",
            "test_afd_async_pool.py",
            "test_afd_pool_failure.py",
            "test_afd_linear_layer.py",
            "test_afd_imports.py",
            "test_afd_tiny_stack.py",
            "afd_tiny_stack.py",
        ],
    ),
]


def files_of(series: dict) -> list:
    return (
        [A + m + ".py" for m in series["mods"]]
        + series["extra"]
        + [T + t for t in series["tests"]]
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def check_partition(base: str) -> list:
    """Disjoint, and the union is exactly the branch's diff.

    A split that quietly drops a file would still compose to something that imports and passes
    most tests -- the missing file would simply not be reviewed. So this is checked against the
    diff itself rather than against the sum of what the series claim.
    """
    want = set(_git("diff", "--name-only", f"{base}..afd/main").split())
    got = [f for s in SERIES for f in files_of(s)]
    problems = []
    dupes = sorted({f for f in got if got.count(f) > 1})
    if dupes:
        problems.append(f"in more than one series: {dupes}")
    if want - set(got):
        problems.append(f"in the branch, in no series: {sorted(want - set(got))}")
    if set(got) - want:
        problems.append(f"in a series, not in the branch: {sorted(set(got) - want)}")
    print(
        f"partition   {len(got)} files across {len(SERIES)} series, branch has {len(want)}"
    )
    # The live counts, printed rather than written down: see the note in this file's docstring on
    # what three hand-maintained copies of them cost.
    shortstat = _git("diff", "--shortstat", f"{base}..afd/main").strip()
    commits = _git("rev-list", "--count", f"{base}..afd/main").strip()
    print(f"            {shortstat}, in {commits} commits")
    return problems


def build(base: str) -> str:
    """Create the five branches, each on the previous one."""
    _git("checkout", "-q", "-B", "afd/review-base", base)
    previous = "afd/review-base"
    for series in SERIES:
        _git("checkout", "-q", "-B", series["branch"], previous)
        _git("checkout", "afd/main", "--", *files_of(series))
        _git("add", "-A")
        _git("commit", "-q", "-m", series["title"], "-m", series["body"])
        stat = _git("diff", "--shortstat", f"{previous}..{series['branch']}").strip()
        print(f"  {series['branch']:24s} {stat}")
        previous = series["branch"]
    return previous


def check_composition(tip: str) -> list:
    """The five, in order, against afd/main -- byte for byte."""
    residue = _git("diff", "--stat", "afd/main", tip).strip()
    print(f"composition {'byte-identical to afd/main' if not residue else 'RESIDUAL'}")
    return (
        []
        if not residue
        else [f"the five series do not compose to afd/main:\n{residue}"]
    )


def check_closure(python: str) -> list:
    """At each tip: every module so far imports, and that series' own tests pass."""
    import os

    env = dict(os.environ, PYTHONPATH="python")
    problems, so_far = [], []
    for series in SERIES:
        _git("checkout", "-q", series["branch"])
        so_far += series["mods"]
        code = "import importlib\n" + "\n".join(
            f'importlib.import_module("sglang.srt.afd.{m}")' for m in so_far
        )
        imported = subprocess.run(
            [python, "-c", code], capture_output=True, text=True, env=env
        )
        tests = [T + t for t in series["tests"] if t.startswith("test_")]
        ran = subprocess.run(
            [
                python,
                "-m",
                "pytest",
                *tests,
                "-q",
                "--no-header",
                "-p",
                "no:cacheprovider",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        summary = [l for l in ran.stdout.strip().splitlines() if l.strip()]
        print(
            f"  {series['branch']:24s} {len(so_far):2d} modules import "
            f"{'OK' if not imported.returncode else 'FAIL'} | "
            f"{summary[-1] if summary else 'no output'}"
        )
        if imported.returncode:
            problems.append(
                f"{series['branch']}: a module imports something no earlier series carries\n"
                + "\n".join(imported.stderr.strip().splitlines()[-3:])
            )
        if ran.returncode:
            problems.append(
                f"{series['branch']}: its own tests do not pass at its own tip"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", default=DEFAULT_BASE, help="the fork point from upstream"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the review branches in place afterwards",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if _git("status", "--porcelain").strip():
        print("the working tree is dirty; this rewrites branches and refuses to run")
        return 2

    problems = check_partition(args.base)
    print("building")
    tip = build(args.base)
    problems += check_composition(tip)
    print("closure")
    problems += check_closure(args.python)

    _git("checkout", "-q", "afd/main")
    if not args.keep:
        for series in SERIES:
            _git("branch", "-q", "-D", series["branch"])
        _git("branch", "-q", "-D", "afd/review-base")

    print()
    for problem in problems:
        print("PROBLEM:", problem)
    print(
        "PASS: five series, disjoint, composing to afd/main" if not problems else "FAIL"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
