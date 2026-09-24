# A custom kernel starts as a fork of a working one

When a path needs a kernel this repo does not have, fork the closest mature
kernel and change the few lines that differ. Do not write one from scratch
because the math is easy to state.

The math is not what makes a decode kernel fast. What makes it fast is the
tiling, the split-and-merge schedule, the block shapes chosen per head count,
the load widths, the software pipelining depth, the dtype of each accumulator,
and a dozen decisions that look arbitrary and are each the residue of a
measurement. A fresh kernel reproduces the math and none of that, so it lands
somewhere between 2x and 10x off, and the gap reads as "this idea does not
work" rather than "this kernel is unfinished" -- which is the expensive
failure, because the idea gets abandoned on the strength of a bad
implementation.

So:

- **Start from the upstream kernel that already serves the case**, and diff
  against it. For a decode attention path that means
  `sglang/kernels/ops/attention/decode_attention.py`; for a linear-attention
  path, the corresponding `fla` kernel. Keep the parameter names so the diff
  stays legible to someone who knows the original.
- **Change the minimum.** Most variants this repo needs differ in where a row
  id comes from, not in what is done with the row. Isolate that, leave the
  schedule alone.
- **Prefer no fork at all.** If the upstream kernel can be pointed at different
  inputs to get the behaviour (a different index array, a different indptr),
  do that and write no kernel. Host-side aliasing beats a copy, and a copy
  beats a fork.
- **A genuinely new kernel needs a reason in its docstring**: which upstream
  kernel was considered, and what made it unusable.

The exceptions are kernels with no upstream counterpart at all -- the sigma
transform, the certificate scan, the CSR pack. Those were written here because
nothing upstream does that work, and each carries measurements in its comments
for the same reason: so the next person changing a block size knows what the
current one bought.
