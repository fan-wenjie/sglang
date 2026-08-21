"""How many of sglang's models this wiring could be installed on, counted rather than guessed.

The Early-Q wiring wraps four things a model file happens to expose. Three of them are common and
one is not, and the difference decides how much of a port a second family is:

    input_layernorm                  nearly universal
    a per-layer mlp                  nearly universal
    layer_communicator.prepare_mlp   where h_l is read from
    forward_prepare_* (four)         where the query is projected apart from the key and value

Counted by reading the model files, so the number is about this checkout rather than about a
memory of it.

    python -m sglang.srt.afd.portability
"""

from __future__ import annotations

import pathlib
import re
import sys

MODELS = pathlib.Path("python/sglang/srt/models")
MARKERS = {
    "layer_communicator": r"layer_communicator",
    "prepare_mlp": r"prepare_mlp",
    "forward_prepare_native": r"def forward_prepare_native",
    "all four prepare variants": None,     # computed below
    "input_layernorm": r"input_layernorm",
}
FOUR = ("forward_prepare_cuda_fused", "forward_prepare_fused_gate",
        "forward_prepare_native", "forward_prepare_npu")


def main() -> int:
    files = sorted(p for p in MODELS.glob("*.py") if not p.name.startswith("__"))
    counts = {name: 0 for name in MARKERS}
    all_four = []
    for path in files:
        text = path.read_text()
        for name, pattern in MARKERS.items():
            if pattern and re.search(pattern, text):
                counts[name] += 1
        if all(f"def {m}" in text for m in FOUR):
            counts["all four prepare variants"] += 1
            all_four.append(path.stem)

    print(f"  {len(files)} model file(s) in this checkout\n")
    for name in MARKERS:
        n = counts[name]
        print(f"    {name:26} {n:4d}  {100 * n / len(files):5.1f}%")
    print(f"\n  models with all four prepare variants, which the query split needs:")
    print(f"    {', '.join(all_four) if all_four else 'none'}")
    print(f"\n  The wiring installs on the intersection. Everything outside it needs those names")
    print(f"  mapped -- which is a port, not a flag -- and check_supported names what is missing")
    print(f"  rather than letting a forward pass discover it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
