# Measurement tools for the attention/feed-forward disaggregation arm

Nothing here runs in a server. These are the tools that produced the numbers in
`AFD_FINDINGS.md`, kept because a finding whose measurement cannot be rerun is a
claim rather than a result.

They live outside `python/sglang/srt/afd/` for one reason: a reviewer opening the
runtime package should see the modules a server actually loads, and there were
twenty-six of these against eighteen of those. A tool that is imported by nothing
sglang runs does not belong in the package sglang runs.

Every one of them takes its model from `AFD_MODEL` and refuses a quantised
checkpoint unless `AFD_ALLOW_QUANTISED=1` is set -- the effects being measured are
smaller than what quantisation moves, so a quantised arm against an unquantised
one compares checkpoints rather than arrangements.

    AFD_MODEL=/path/to/checkpoint python benchmark/afd/context_arms.py --help
