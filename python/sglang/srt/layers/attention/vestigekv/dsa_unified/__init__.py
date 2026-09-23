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

and a microbenchmark of the archive scan (2026-09-23) shows each pass carries
a launch floor: the kernel takes ~15 us whether the archive holds 32k rows or
131k, so its time does not track its bytes at all in this range. That is the
measurement that retired three attempts in a row -- 4:1 pooling, halving the
rank, and fp8 storage all shrank the bytes of ONE pass and moved the decode
slope by 2.9% or less. Shrinking bytes cannot pay when the cost is a fixed
floor per pass, multiplied by the number of passes.

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
