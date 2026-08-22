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


def resolve(name: str | None) -> Callable | None:
    """The factory for an arm, or None when nothing has registered it.

    None is not an error here. It is the ordinary state when only standard AFD is installed, and
    the caller decides whether the operator asked for something that is missing.
    """
    if name is None:
        return None
    return _ARMS.get(name)


def available() -> tuple[str, ...]:
    """The arms that have registered, for an error message that can name the alternatives."""
    return tuple(sorted(_ARMS))
