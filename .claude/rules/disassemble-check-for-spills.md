# Disassemble a changed kernel and check it does not spill

Any change to a Triton kernel on the decode path — a new branch, a new
constexpr variant, a new live value — is disassembled before it is timed. A
compiled kernel's register allocation is a property of the build, not of the
data, so a variant that spills makes **every** step slower, including the steps
whose feature never fires.

```bash
python mexp/kimi/fence_disasm.py --out <dir>      # the VestigeKV stage-1 pair
cuobjdump -res-usage <kernel>.cubin               # REG, STACK for any cubin
grep -c LDL <kernel>.sass                         # local-memory reloads
```

`STACK: 0` and no `LDL`/`STL` is the bar. Compare the new variant against the
one it is meant to match, not against nothing: the question is whether the
feature costs registers, and the answer is a number.

## What this caught

The fenced row source in `decode_fork.py` was written as a branch inside the
row loop (`elif vk_fenced:`). Each arm defined `kv_loc`, and the compiler
carried a separate downstream address tensor for each across the loop:

| build | REG | STACK | LDL/STL | SASS lines |
|---|---|---|---|---|
| FENCE=False | 200 | 0 | 0 | 2231 |
| FENCE=True, branch in the loop | 255 | 40 B | 22 | 2996 |
| FENCE=True, value select | 200 | 0 | 0 | 2415 |

The spilling build cost about 0.10 ms/step at 256k, paid by the 99.7% of scans
that never fence. The timings could not attribute it — the profile's per-kernel
means showed the fenced and unfenced builds within a few microseconds, because
a fence fires on a few percent of steps and the average hid both the fenced
work and the spill.

## The shape that avoids it

Select the **value**, not the path, when two row sources feed one downstream
computation:

```python
# Spills: two definitions of kv_loc, two address tensors live across the loop
if fenced:
    kv_loc = <page table read>
else:
    kv_loc = <kept/fetch read>

# Does not: one definition, the inapplicable load predicated off
a = tl.load(..., mask=live & (not fenced), other=0)
b = tl.load(..., mask=live & fenced, other=0)
kv_loc = tl.where(fenced, b, a)
```

"Mutually exclusive branches share registers" holds only when their live ranges
are disjoint. A branch inside a loop whose body uses the result does not
qualify, because both arms stay live across the whole loop.

`test_vestigekv_decode_fork.py::TestDecodeForkRegisters` pins this: it fails if
the fenced build spills or uses more registers than the unfenced one.

## When the two paths must be separate code

Selecting the value works when both arms do the same kind of work and only the
operand differs. It does **not** work when the point of one arm is to *remove*
an instruction: an affine row address (`base + offs_n`, no id load) lets the
pipeliner emit `cp.async`, and a predicated-off indirect load still blocks
that, so writing both and selecting gives up the whole gain.

Where an arm exists to delete work, the arms have to be separate code, and the
branch has to sit **outside** the loop -- one complete loop per arm, shared
values defined and initialized above the branch. Inside the loop both arms'
address tensors stay live and the allocator spills, which is the failure this
rule exists for. The cost is a second copy of the loop body in the SASS, so
check the instruction count as well as the spill.

Two checks, not one, when an arm is supposed to be cheaper:

1. `STACK: 0` and no `LDL`/`STL`, as above.
2. The cheaper arm actually got what it was written for -- e.g. `LDGSTS` or a
   TMA instruction in its SASS. An arm that duplicates the loop and does not
   show the instruction it was written to enable has cost code size and bought
   nothing.

And measure the precondition before writing either: the VestigeKV page table's
affinity is a run-time property of the allocator, reported as
`pagetable_affine` in the stats line (`_probe_page_table`). An affine arm is
all-or-nothing per lane, so a ratio short of 1 means the arm cannot be taken
and the code should not be written.
