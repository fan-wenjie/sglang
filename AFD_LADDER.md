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

## What has to be built

Each rung needs a setting that turns on exactly one change. They are not independent knobs on a
finished system -- rung 3 presupposes rung 2 -- so one ordinal flag, `--afd-cut-level {0,1,2,3,4}`,
says how far up the ladder to go, and the code branches once per rung rather than once per feature.

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

## Order of work

1. **Anchor the original.** Tag it, so every later claim about "what it used to do" is checkable
   rather than remembered.
2. **Carve the package in two** as a certified pure relocation -- `mechanical-refactor-verify`,
   prepare + move + postpare, byte-diff proof. No behaviour changes in the same commit.
3. **Build rung 1**, run the three prompts, record the verdict. Then rung 2, then 3, then 4.
4. **Rewrite the query-shift half on the shared core** as each rung demands it, migrating the old
   code rather than copying it, and dropping what the ladder shows is unnecessary.

Rung by rung, the verdict is the same sentence: identical to colocated, or not. The first "not" is
the answer this search has been unable to reach by any other means.
