# Rebuilding the arrangement as a ladder from standard AFD

## Why

Standard AFD -- feed-forward on the pool, attention on the host, no shift -- is **token-identical
to colocated** on three prompts. The group cut is not. Between them sit four changes that were
landed as one, and days of inspection have not found which one breaks it: five real bugs were
found and fixed and none was the cause, and ten separate times a comparison turned out to measure
something other than what it claimed.

A ladder replaces inspection with construction. Each rung is a working server with one more change
in it, and each rung is judged the same way: **token-identical to colocated, three prompts**. The
rung that stops being identical is the change that breaks it, and no instrument is needed to say
so.

## The rungs

    0   standard AFD                    feed-forward on the pool. VERIFIED identical.
    1   + projections on the pool       W_o, W_q, W_kv move (was task 43)
    2   + linear attention on the pool  whole layers move; the recurrent state stays on the host
                                        and is read across the wire
    3   + one round trip a group        the residual and the gate stay on the pool between layers
    4   + the shifted read point        the next group's query projected from inside the span

Rung 4 is today's arrangement. Rungs 1 to 3 do not exist as separate settings: `--afd-span-cut` is
all of them at once, which is why the failure has no smaller box to be in.

## What has to be built, and how the rungs are selected

NOT by a level flag in the shared code. An ordinal `--afd-cut-level {0..4}` was the first design
here and it is the wrong one: it puts knowledge of the derived arm inside the standard path, so
every rung's branch lives in the file that is supposed to be the arm-independent half. Standard
AFD must not know that a query shift exists.

Each rung is a CLASS instead, and each derives from the one below it:

    StandardPool                          rung 0, in afd/
      ProjectionsOnPool(StandardPool)     rung 1
        LinearOnPool(ProjectionsOnPool)   rung 2
          GroupedSpan(LinearOnPool)       rung 3
            ShiftedReadPoint(GroupedSpan) rung 4

all but the first in the derived package. The choice happens once, at the composition root, by
which class is constructed -- nothing below it branches. A rung is then not a configuration to be
read but a type to be instantiated, and "which rung is running" is answerable by asking the object
what it is rather than by reading a flag through six call sites.

The repository's own style rule prefers composition to inheritance and forbids mixins. This chain
is neither a mixin nor a grab bag: it is a linear specialisation where each rung genuinely IS the
one below it plus one change, which is the shape the ladder has by construction. Where a rung
needs to vary a step rather than extend it, the step is a collaborator held by the class and
swapped, not an overridden method.

## The split, by what a module actually says rather than by memory

31 modules, 10,046 lines. The first cut was drawn from a seed list I chose, which is not a
criterion -- it decides the answer in advance. The criterion used instead is mechanical: a module
belongs to the derived arm only if its CODE, not its comments, speaks of the shift, the read point
or the span. Counted that way:

    stays in query-shift    span 61   wiring 45   span_routing 44   read_point 20
                            checkpoint 18   early_q 11
    borderline              supported 4   sweep_ahead 4
    moves up to shared      crossover 0   parked_sweeps 0   split_read_kernel 0
                            split_attention 0   early_k_arms 0   linear_history 1
                            history_service 2   linear_state 2   seating 2

The recurrent-state family -- `linear_state`, `linear_history`, `split_read_kernel`,
`history_service` -- has nothing to do with the shift. It is the general machinery for "the state
stays on the host and is read across the wire", which is exactly what rung 2 needs and what any
future cut that moves a stateful layer would need. It belongs to the shared half, and the first
classification put it in the derived one only because I had reached for it while building the
span.

Same for `seating` (a batching policy), `split_attention` (partitioning a softmax attention, useful
to anything that wants to start a sweep early or late), and `split_read_kernel` (one pass over a
state emitting two readings).

That leaves the derived arm at roughly 3,000 lines against 7,000 shared, rather than the even
split the seed list produced. The shared half is what sglang keeps.

## The property upstream needs: AFD ships without the derived arm

AFD may land upstream before anything derived from it does, so standard AFD has to work with the
derived code **absent** -- not merely disabled. Those are different properties. A flag defaulting
to off still fails when a module imports the derived package inside an `if` and the directory is
not there, and it fails at the first request rather than at startup, on a machine with no way to
fix it.

Machine-checkable, as one grep: no file under `sglang/srt/afd/` may name a derived package.
`test_afd_stands_alone.py` is that grep, plus the registry's own contract.

The check found two real debts on the day it was written, and they are held as a ratchet that may
only shrink:

    roles.py        reads `server_args.afd_query_shift_layers`
    checkpoint.py   `RETIRED_KEYS` maps the derived arm's retired flag names

A derived arm announces itself through `afd/arms.py` instead: it imports that module and registers
a factory, and whoever wants the arm imports the derived package -- which is the only thing that
puts it in the registry. Delete the derived directory and the registry is empty, `resolve` returns
None, and the standard arrangement takes the only path there is.

## Branches, and what belongs to which

    origin/main
      └── afd                    the arrangement itself, complete, on its own
            └── afd-query-shift  the derived arm, in its own directory

`afd` is forked from `origin/main` rather than continued from `afd/span-cut`, so what upstream
sees is the arrangement and not the four days of instruments built to debug something else.

**The batch re-forming belongs to AFD, not to the derived arm.** It is the travel-group model: a
group fills a coach, tours the cities that are seen together, and disperses only where the visit
takes an unpredictable time. The principle is the one this work established by measurement --
re-form a batch only where LATENCY VARIES. A feed-forward and a linear attention are context-free:
361 microseconds and 200, whoever is riding. A softmax attention is 19 microseconds at 1k and 2397
at 128k, and a batch that waits for its slowest rider there wastes the difference.

So the coach, the waiting room, the seating and the departure policy are AFD's own, and every
`afd_query_shift` inherits rather than reimplements. The pieces already exist and already sit on
the shared side: `seating`, and the departure machinery in `pool_server`.

What the derived arm adds on top is only the read point -- the query projected from earlier in the
stack -- and whatever the ladder proves it cannot do without. Every rung that turns out to be
AFD's rather than the arm's moves down into `afd`, and the subclass shrinks by that much.

## Standard AFD has no overlap of its own, so it needs two batches staggered

Within one request the two machines cannot work at once. The host's key and value are projected
from `x_l`, and `x_l` is what the pool returns -- so the host waits for the pool, then the pool
waits for the host, and each is idle for the other's turn. That is what the shifted read point
buys back, and it is exactly what standard AFD must not depend on.

The arrangement that fills both machines without touching the model is two batches, staggered:
while the pool runs batch A's feed-forward, the host runs batch B's attention, and the next step
they swap. Neither batch is overlapped with itself, so nothing about the model changes and no
query moves -- it is a scheduling property, and it belongs to AFD.

Note what it costs and what it does not. It doubles the requests in flight, so it doubles the KV
cache the host must hold at a given depth, which is the resource the whole arrangement is short of
-- that trade has to be measured, not assumed. It does NOT need the read point, and a measurement
of it must not be reported beside a shifted-read measurement as though they were the same
arrangement.

Last of the work, after the ladder has said what breaks the derived arm. Recorded now so that when
the pipeline turns out to be half idle, the reason is already written down.

## The verdict, and the gate it has to pass first

`benchmark/afd/rung_verdict.py` reports both halves: whether the tokens are identical, and how far
the final hidden state drifted, per token, as a relative difference and a cosine. It refuses to
report anything until the logs show the arrangement INSTALLED -- the host's "the group cut is
installed", the pool's "span(s) a decode step".

That gate is not ceremony. The same script, run against an arrangement whose arm had registered
but not installed, reported:

    tokens identical, relative 0.005 to 0.0125, cosine 0.99999

which reads as "the cut works". It was standard AFD. With the arm genuinely installed, the same
prompt gives:

    tokens part at token 0, relative 1.17, cosine 0.217

Two orders of magnitude apart, and both runs logged "the 'query-shift' arm is available". A
verdict that cannot say which arrangement produced it is worse than no verdict, so the script
refuses rather than warns.

rung 4's baseline is therefore: parts at the first token, cosine 0.22 to 0.67 across the
generation. Every rung below it is measured the same way and the first one that stops matching is
the answer.

## What the ladder has said so far

    rung 0   standard AFD                token-identical to colocated, three prompts
                                         and it really was routing: "routing 64 layer(s) to the
                                         pool", all of them, not a subset
    rung 4   group cut + shifted read     parts at token 0, relative 1.17, cosine 0.217
    rung 3   group cut, read point at 0   parts at token 0, relative 1.23, cosine 0.128

Both rungs with installation confirmed on both ends, which is the part that makes them worth
anything. rung 3 and rung 4 are the same to within their own scatter, so the shifted read point is
not the cause -- and this is the first time that has been said about a run where the arrangement
was known to be running. Every earlier shift-0 control was taken before "registered" and
"installed" were known to be different questions, and none of them can be relied on.

The fault is at rung 3 or below: the group cut without the shift. That is the linear attention on
the pool, the residual and gate held there between layers, and one round trip a group instead of
one a layer. rungs 1 and 2 do not exist yet as separate settings and have to be built.

## A trap rung 0 could have fallen into and did not

`routable_layers` excludes any layer whose feed-forward is a `Qwen2MoeSparseMoeBlock`, because a
sparse block takes a forward batch the wire does not carry. If this model's layers were sparse,
standard AFD would have routed NOTHING, run the whole stack on the host, and reported
token-identical to colocated for the same reason the uninstalled arm did -- because the
arrangement under test was not running.

Checked rather than assumed. The config has no `num_experts`, and the host logged "routing 64
layer(s) to the pool" -- all of them. rung 0's verdict stands.

Worth writing down because it is the third instance of one shape in two days: an arrangement that
looks like it is running, agrees with the baseline, and agrees because it is not running. The
others were the arm that registered without installing, and `--afd-coverage` accepted and ignored.
Every rung's verdict now has to carry evidence that the rung was in effect, not merely that the
server started.

## Order of work

0. **Find the fault.** The ladder is how, and it is the reason the branches are laid out this way:
   each rung is a working server, and the rung that stops being token-identical to colocated is
   the change that breaks it. Organising the code is not a detour from the debugging -- it IS the
   instrument, and the only one in this search that cannot measure the wrong thing.
1. **Anchor the original.** Tag it, so every later claim about "what it used to do" is checkable
   rather than remembered.
2. **Carve the package in two** as a certified pure relocation -- `mechanical-refactor-verify`,
   prepare + move + postpare, byte-diff proof. No behaviour changes in the same commit.
3. **Build rung 1**, run the three prompts, record the verdict. Then rung 2, then 3, then 4.
4. **Rewrite the query-shift half on the shared core** as each rung demands it, migrating the old
   code rather than copying it, and dropping what the ladder shows is unnecessary.

Rung by rung, the verdict is the same sentence: identical to colocated, or not. The first "not" is
the answer this search has been unable to reach by any other means.
