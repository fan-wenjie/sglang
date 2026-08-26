"""This arm's own flags, refused before a model loads.

Both checks were in `arg_groups/afd_hook.py`, which had to import this package by name to run
them. That is the dependency direction the split exists to remove: AFD ships without this arm, and
a shared module that imports it inside an `if` still fails when the directory is not there. They
are unchanged otherwise -- each still guards the failure it was written for, and each of those
failures was silent.

Reached through `SpanArm.check_args`, which `arms.check_args` calls for every registered arm.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MAX_REASONABLE_SHIFT = 64


def check(server_args) -> None:
    """Everything this arm refuses, in one call."""
    _check_arm_is_present(server_args)
    _check_shift(server_args)


def _check_shift(server_args) -> None:
    """A layer count, and there is no fractional setting: a query read from between a block's two
    sub-layers would be read from a point where the residual stream has no value. A rounded value
    would run a different arm than the one the flag names."""
    shift = getattr(server_args, "afd_query_shift_layers", None)
    if shift is None:
        # unset: the checkpoint's own read point is resolved at load, where its config is in hand
        return
    if not isinstance(shift, int) or isinstance(shift, bool):
        raise TypeError(f"--afd-query-shift-layers is a layer count, got {shift!r}")
    if shift < 0:
        raise ValueError(f"--afd-query-shift-layers must not be negative, got {shift}")
    if shift > MAX_REASONABLE_SHIFT:
        raise ValueError(
            f"--afd-query-shift-layers={shift} is deeper than any stack this serves. The offset "
            f"is N-0.5 layers, so N is a layer count and not a half-layer count -- the study's "
            f"unit is 2N-1, and a value that looks like a half-layer count here would silently "
            f"run twice the depth it names."
        )
    if shift > 0 and server_args.afd_mode == "null":
        logger.warning(
            "--afd-query-shift-layers=%s with the two sides colocated: the query is read early "
            "and there is no pool call for it to overlap. This measures what the rewiring costs, "
            "which is a real question, but it is not what the arrangement buys.",
            shift,
        )
    if shift == 0 and server_args.afd_mode != "null":
        logger.info(
            "afd %s with --afd-query-shift-layers=0: the standard read point, so the host waits "
            "for each feed-forward before its next attention. This is the synchronous baseline.",
            server_args.afd_mode,
        )


def _check_arm_is_present(server_args) -> None:
    """An arm asked for by flag but not installed is refused at startup, not ignored.

    The derived arms live outside `sglang.srt.afd` and announce themselves by being imported. That
    is what lets AFD ship without them. It also means a server can be launched with the arm's flag
    set, find nothing registered, and serve the STANDARD arrangement in silence -- which is
    exactly what happened the first time this indirection ran: both ends took `--afd-span-cut`,
    neither installed anything, and the output became correct because the arrangement under test
    had stopped running. A quiet fallback that produces right answers is the worst kind: it reads
    as a fix.

    So the flag is honoured or the server does not start. Importing the arm's package here is what
    registers it, and the ImportError is reported with the flag that asked for it.
    """
    from sglang.srt.afd.installer import span_cut_wanted_for

    if not span_cut_wanted_for(server_args):
        return
    from sglang.srt.afd.arms import available

    # This check runs FROM the arm, so reaching it at all proves the package imported. What it
    # can still catch is the half-failure: the package present, the import fine, and nothing in
    # the registry -- which is what a rename or a botched register() leaves behind, and it
    # serves the standard arrangement in silence.
    if not available():
        raise ValueError(
            "--afd-span-cut was given, the arm's package imported, and nothing registered. The "
            "arrangement that would run is the standard one."
        )
