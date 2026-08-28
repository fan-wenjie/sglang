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
    _warn_the_flag_overrides_the_checkpoint(server_args, shift)
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



def _warn_the_flag_overrides_the_checkpoint(server_args, shift: int) -> None:
    """Say, every time the flag is set, that it is overriding what the checkpoint says.

    A checkpoint trained for this read point declares it as `query_shift_layers` in its own
    config, and unset is how a deployment gets that value: `checkpoint.resolve` asks the flag
    first and the checkpoint second, so a deployment that names nothing serves what its weights
    were repaired for. The flag exists for the measurement that has to run a checkpoint at a read
    point it was NOT repaired for -- which is a real question and is the only thing it is correct
    for. Every other use projects a query from an input the weights never saw.

    So it warns whenever it is set, and loudest when it CONTRADICTS a checkpoint that stated its
    own: that is the case where a repaired model is served at the wrong point, and the symptom is
    fluent text from a model nobody trained -- there is nothing downstream to catch it.

    A host is exempt from the comparison: its path is `pool://host:port`, it has no config to
    read, and a host whose flag contradicts the pool's pushed word is already refused at adoption.
    """
    from sglang.srt.afd.model_files import is_pool_path

    path = getattr(server_args, "model_path", None)
    if path is None or is_pool_path(path):
        return

    from sglang.srt.afd.checkpoint import stated_shift

    stated = stated_shift(path)
    if stated is None:
        logger.warning(
            "--afd-query-shift-layers=%s was given and this checkpoint states no read point of "
            "its own, so its weights were never repaired for one. This is a measurement "
            "configuration: it prices what the rewiring costs BEFORE repair. Serving it is "
            "serving a model nobody trained, fluently and undetectably.",
            shift,
        )
    elif int(stated) != int(shift):
        logger.warning(
            "--afd-query-shift-layers=%s CONTRADICTS this checkpoint, which states %s. The flag "
            "wins and the weights do not: they were repaired for %s, and every layer's query "
            "will be projected from an input they never saw. Unset the flag to serve the "
            "checkpoint as trained; keep it only if you are measuring the difference.",
            shift,
            stated,
            stated,
        )
    else:
        logger.info(
            "--afd-query-shift-layers=%s repeats what the checkpoint already states, so it "
            "changes nothing. Leaving it unset is how a native checkpoint is served.",
            shift,
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
