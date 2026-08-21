# Early-Q verification, recovered from the deployment host

These seven files existed ONLY on the remote host. They were never in the repository, so no gate
ever read them, and they rotted quietly while the code they check was renamed underneath them:
`afd_coverage` became `query_shift_coverage`, `afd_q_shift_layers` became `afd_query_shift_layers`,
and `Frame` went from carrying one tensor to carrying a tuple. Fourteen of sixty cases are red on
arrival for exactly those reasons.

They are committed here in the state they were found, before any repair, so that the repair is a
diff against something rather than a rewrite of something lost. What they cover is checks 1-3 of
the `afd-early-q` skill -- the read point is the tensor it claims to be, the coverage is what was
asked for, and the split reproduces fused attention with a control -- which is the standard the
group cut still has to meet.

Run them with `pytest test/srt/afd/`. They are torch-only apart from `test_early_q_end_to_end.py`,
which builds a `ServerArgs`.
