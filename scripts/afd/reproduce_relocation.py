"""Reproduce a relocation commit from its parent, and byte-diff the result.

The skill's generator infers a recipe from function-level moves and reports UNSUPPORTED for this
shape: five whole files moved with `git mv`, and every importer repointed. The property it checks
is the one that matters and it does not need the generator -- regenerate the commit from its
parent with faithful primitives, then compare bytes. An empty diff is the proof; any residual is a
non-move change that rode along, which is exactly what wants surfacing.

Faithful here means two primitives and nothing else:

    move    `git mv OLD NEW`, no edit to the file's contents
    repoint `sglang.srt.afd.<name>` -> `sglang.srt.afd_query_shift.<name>`, whole-token, applied
            to every file that contains it, longest module name first so `span_routing` is not
            rewritten as `span` + `_routing`

If the commit contains anything else -- a renamed symbol, a changed signature, a line of new logic
-- the diff will show it. That is the point: a reshape must not ride along with a move, and this
is how anyone re-checks that claim without taking my word for it.

    python scripts/afd/reproduce_relocation.py <commit>
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

MOVED = ["span_routing", "span", "read_point", "early_q", "selfcheck"]


def git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout


def reproduce(commit: str) -> int:
    root = git("rev-parse", "--show-toplevel").strip()
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "work")
        git("worktree", "add", "--detach", work, f"{commit}~1", cwd=root)
        try:
            for name in MOVED:
                old = f"python/sglang/srt/afd/{name}.py"
                new = f"python/sglang/srt/afd_query_shift/{name}.py"
                if os.path.exists(os.path.join(work, old)):
                    git("mv", old, new, cwd=work)

            listing = git("grep", "-l", "sglang.srt.afd.", "--", "python", "test",
                          cwd=work).split()
            for rel in listing:
                path = os.path.join(work, rel)
                text = open(path).read()
                for name in MOVED:                       # longest first
                    text = text.replace(f"sglang.srt.afd.{name}",
                                        f"sglang.srt.afd_query_shift.{name}")
                open(path, "w").write(text)

            git("add", "-A", cwd=work)
            diff = subprocess.run(
                ["git", "diff", "--stat", commit], cwd=work, capture_output=True, text=True,
            ).stdout.strip()
            if not diff:
                print(f"PASS  {commit[:9]} reproduces from its parent, byte for byte")
                return 0
            print(f"RESIDUAL  {commit[:9]} does not reproduce; what rode along:\n{diff}")
            return 1
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", work], cwd=root,
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(reproduce(sys.argv[1] if len(sys.argv) > 1 else "HEAD"))
