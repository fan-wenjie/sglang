"""Where a derived arm announces itself, so this package never has to name one.

AFD may land upstream before anything derived from it does, so the standard arrangement has to
work with the derived code ABSENT -- not merely disabled. That is a stronger property than a flag
defaulting to off, and it is the one worth checking: a module that imports
a derived package inside an `if` still fails when that directory is not there, and it fails at
the first request rather than at startup.

So nothing in `sglang.srt.afd` refers to a derived package by name. A derived arm imports this
module and registers a factory; whoever wants that arm imports the derived package, which is the
only thing that puts it in the registry. With the derived package deleted the registry is empty,
`resolve` returns None, and the standard arrangement takes the only path there is.

The registry is deliberately tiny. It holds no configuration and makes no decisions -- the
composition root asks for an arm by the name the operator gave, gets a factory or nothing, and
constructs. Anything richer would be this package knowing about arms again by another route.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

_ARMS: dict[str, Callable] = {}
_LOADED = False


def register(name: str, factory: Callable) -> None:
    """Announce a derived arm. Called by the derived package at import, by nobody else.

    A second registration under one name is refused rather than overwritten: two arms answering to
    one name is a coin toss over which arrangement a run measured, decided by import order.
    """
    if name in _ARMS and _ARMS[name] is not factory:
        raise ValueError(
            f"two arms are registered as {name!r}. Which one a run used would be decided by "
            f"import order, and the arrangement it measured would not be recorded anywhere."
        )
    _ARMS[name] = factory
    logger.info("afd: the %r arm is available", name)


def load() -> None:
    """Import whatever arm packages are installed, HERE, in the calling process.

    The registry is a module-level dict, so it is per-process, and the roles that consult it live
    in a process sglang spawns. An import in the parent leaves this empty in the child -- the arm
    announces itself, the announcement is logged, and nothing is installed.

    Discovery is by naming convention rather than by a list this package maintains: any top-level
    `sglang.srt.afd_*` package that is not `afd` itself is imported once. That keeps the rule "AFD
    never names a derived arm" while still letting the arm be found where it is needed, and an arm
    that is not installed simply is not there to import.
    """
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    import importlib
    import pkgutil

    import sglang.srt as srt

    for module in pkgutil.iter_modules(srt.__path__):
        if module.name.startswith("afd_") and module.ispkg:
            try:
                importlib.import_module(f"sglang.srt.{module.name}.installer")
            except ImportError as e:
                logger.info("afd: %s is present but did not register an arm: %s", module.name, e)


def resolve(name: str | None) -> Callable | None:
    """The factory for an arm, or None when nothing has registered it.

    None is not an error here. It is the ordinary state when only standard AFD is installed, and
    the caller decides whether the operator asked for something that is missing.
    """
    if name is None:
        return None
    return _ARMS.get(name)


def check_args(server_args) -> None:
    """Let every registered arm refuse a configuration of its own flags.

    An arm that adds a flag owns the refusal that flag needs, and the refusal has to run before a
    model loads rather than at the first token. Putting those checks in the shared argument hook
    made that file name a derived package by hand, which is the one dependency direction that must
    not exist -- AFD ships without the arms, and a module that imports one inside an `if` still
    fails when the directory is not there.

    An arm with nothing to check simply has no `check_args`. With no arms installed this iterates
    an empty dict, which is the ordinary case and not a degraded one.
    """
    for name, factory in sorted(_ARMS.items()):
        check = getattr(factory, "check_args", None)
        if check is None:
            continue
        check(server_args)


def install_transforms(*, model, model_config, server_args):
    """Let whatever arm this build carries transform the LOADED model. None when there is none.

    The one hook a derived arm needs inside the model runner, and it names no arm. Before this
    existed, the runner -- a frozen file -- imported a derived package by name inside a `try`, read
    three of that arm's own flags off `server_args`, and handed them over. Every one of those is a
    dependency from the half that ships to the half that may not.

    Returns what the arm returns, for the runner to hold. More than one arm transforming one model
    is refused: two rewritings of the same layers compose into an arrangement nobody described,
    and the run would be attributed to whichever was asked for.
    """
    # HERE, in this process. The model runner lives in a scheduler sglang spawned, and an arm
    # imported by the launcher registers in the parent and leaves this registry empty in the
    # child -- which is what `load` exists for and what this function did not do. The symptom was
    # a shift that quietly stopped installing: a quality re-run reported the shifted arm's bits
    # per byte as exactly the unshifted arm's, 0.713010 against 0.713010, and an arm that is not
    # installed is indistinguishable from an arm that costs nothing.
    load()

    installed = []
    for name, factory in sorted(_ARMS.items()):
        transform = getattr(factory, "transform_model", None)
        if transform is None:
            continue
        result = transform(model=model, model_config=model_config, server_args=server_args)
        if result is not None:
            installed.append((name, result))
    if len(installed) > 1:
        raise ValueError(
            f"{len(installed)} arms transformed this model: {[n for n, _ in installed]}. Each "
            f"rewrites the same layers, so what ran is neither of them and the run would be "
            f"reported under whichever name was asked for."
        )
    return installed[0][1] if installed else None


def absent_classes(server_args) -> tuple:
    """Module classes an arm computes remotely, to be built with no storage at all.

    The fifth entry point, and the one that reaches furthest into the shared half: it decides what
    the LOADER allocates, before any model exists. `absent_ffn` already does this for the
    feed-forward -- the parameters are constructed on the meta device, so the memory is never
    taken rather than taken and released -- and that distinction is the whole point. Releasing
    afterwards recovers steady-state memory and does nothing about the peak, and the peak is what
    fails on a card smaller than the checkpoint.

    An arm may only name a class whose EVERY instance it computes remotely. A class shared with
    modules the host still uses cannot be named here: the loader wraps the class, not the
    instances, so naming one would strip storage from things nobody routed and the symptom is a
    meta tensor reaching a kernel -- a message about devices, from somewhere that names neither
    the arm nor the module. What cannot be named this way is released after the routing instead,
    which is a different mechanism with a different cost.

    Returns an empty tuple when no arm is installed, which is the ordinary case.
    """
    found: list = []
    for name, factory in sorted(_ARMS.items()):
        declare = getattr(factory, "absent_classes", None)
        if declare is None:
            continue
        classes = tuple(declare(server_args))
        if classes:
            logger.info("afd: the %r arm builds %s class(es) with no storage", name, len(classes))
        found.extend(classes)
    return tuple(found)


def sweep_schedule(transform):
    """The sweep schedule an arm's transform produced, or None.

    The host router opens its window between issuing a feed-forward and collecting it, and the
    schedule that fires in that gap belongs to whichever arm moved a read point early enough to
    have one. `SweepAhead` is shared -- the schedule is AFD's own idea -- so this asks for a
    named attribute rather than reaching through the arm's own structure, which is what the
    frozen file used to do (`self.afd_early_q.hooks.sweep_ahead`, two levels into a package that
    may not be installed).

    An arm that moves nothing early sets it to None. With no arm at all there is nothing to ask.
    """
    return None if transform is None else transform.sweep_ahead


def available() -> tuple[str, ...]:
    """The arms that have registered, for an error message that can name the alternatives."""
    return tuple(sorted(_ARMS))
