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
