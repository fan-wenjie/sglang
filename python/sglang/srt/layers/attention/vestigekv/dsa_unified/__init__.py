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

So the target is the format, not any single term's size: one interleaved
record per token holding DSA's indexer key, VestigeKV's rank-r sketch, its
residual norm and tier 1's keep flag, read once front to back with every
consumer taking its slice from registers. That also collapses the two address
spaces -- DSA addresses pool slots, the archive addresses position within the
request -- whose mapping needs a scatter today and silently produced wrong
rows when it was written as a binary search.

The cache layout is upstream's, which is exactly why this is a fork: merging
the sketch into the record is not something a backend beside DSA can do.
"""
