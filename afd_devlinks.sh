#!/bin/bash
# Generate the .so development symlinks the JIT link step needs.
#
# The cu13 wheels ship libcudart.so.13 and no unversioned libcudart.so, so `-lcudart` fails at the
# link step of a kernel compiled during a model load. The symlinks were made by hand the first
# time, which meant the second machine had no way to know they were needed -- a build step nothing
# had ever read. This is that step, in the tree, run by afd_env.sh on every source.
#
#     ./afd_devlinks.sh            uses $CU13 from afd_env.sh
#     ./afd_devlinks.sh <cu13dir>  names the wheel directory explicitly
set -euo pipefail
CU13_DIR="${1:-${CU13:-}}"
if [ -z "$CU13_DIR" ] || [ ! -d "$CU13_DIR/lib" ]; then
    echo "no cu13 lib directory: pass one, or set CU13 (got '${CU13_DIR}')" >&2
    exit 1
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/.cuda-devlinks"
mkdir -p "$OUT"
made=0
for lib in "$CU13_DIR"/lib/lib*.so.*; do
    [ -e "$lib" ] || continue
    base="$(basename "$lib")"
    # libfoo.so.13 and libfoo.so.13.1.2 both map to libfoo.so; take the shortest soname only, so a
    # fully-versioned file cannot win the link over its own soname
    stem="${base%%.so.*}.so"
    target="$OUT/$stem"
    if [ ! -e "$target" ] || [ "$(readlink "$target")" != "$lib" ]; then
        ln -sf "$lib" "$target"
        made=$((made + 1))
    fi
done
echo "afd: $made link(s) written to $OUT ($(ls "$OUT" | wc -l) total)"
