# SPDX-License-Identifier: Apache-2.0
"""DSA's indexer and its key cache, forked to carry VestigeKV's sketch in the
same per-token record so the sequence is scanned once.

FORKED_FROM lists the upstream paths and their blob hashes at the fork point;
every file in this package starts byte-identical to the one it names, so the
diff that follows reads as a change against the original rather than as new
code. Nothing imports this package yet.

Why a fork rather than a second structure beside DSA's. Per decode step the
sequence is currently read four times over, each pass with its own layout and
its own kernel:

    DSA indexer, pooled index cache        33 B/token/layer
    VestigeKV archive scan, csk + rho     132 B/token/layer
    pack arena refresh, aidx/rho/arch     ~12 B/token/layer
    tier-1 kept set, latent rows           32 B/token/layer

What is established is the outcome, not yet the mechanism. Three attempts in
a row -- 4:1 pooling, halving the rank, fp8 storage -- each shrank the bytes
of ONE pass, and each moved the decode slope by 2.9% or less. So the bytes of
that pass are not what the slope is made of.

A Python-timed microbenchmark suggested the scan carries a ~15 us floor
independent of archive size, which would explain it, but that timing includes
Triton's host-side launch path and the served scan is captured in a CUDA
graph where none of that exists. Treat the floor as unverified until the
per-kernel GPU time says so; the conclusion it was offered for -- that the
scan's bytes do not drive the slope -- rests on the three null results and
stands without it.

WHAT THE TRACE THEN SAID, and why nothing here is wired yet. A marker-anchored
decode window at 4k and at 32k gives, per step:

    _prologue_scores   57.50 us   +37.85 growth   tier 1's kept-set scoring
    _scan_batched      37.21 us    +1.15 growth   the archive scan, one launch
    DSA logits         39.89 us    +7.14 growth   eleven launches
    DSA top-k          90.12 us   +19.80 growth   eleven launches

Four facts follow, and together they retire the merge this package was forked
for:

1. The scan is ALREADY one launch covering every layer. The "four passes"
   framing that motivated the fork counted it as one of four; it is in fact the
   one structure that already does what the single-pass principle asks.
2. The two passes are at opposite ends of the step. The scan sits at the top,
   before any layer overwrites qbuf, which is what keeps stale-by-one sound;
   DSA's indexer runs per layer throughout. The whole model forward separates
   them, so co-locating their bytes cannot help -- L2 is long gone.
3. Merging them in TIME would split the scan's single launch into eleven, the
   opposite of the goal.
4. The logits kernel has three backends -- deepgemm, cutedsl, aiter -- and none
   is Triton, so a fused kernel would mean reimplementing a tuned fp8 GEMM,
   which is the failure this repo's fork rule exists to prevent.

The layout module stays because it is correct and tested, and because the
interleaved shape may serve a different consumer later. The growth is in
_prologue_scores, which is tier 1's kept set scored densely at rho*S rows;
scoring it by the sketch instead was measured and fails on all three of its
uses (the max, the entropy and the gate). So the slope's remaining lever is
rho itself.
"""
