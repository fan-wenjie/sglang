"""Where a converted checkpoint's read point comes from, and who may override it.

A checkpoint repaired at one read point must be SERVED at that read point. Fine-tuning W_q to
read h_{l-1} and then serving the result at the standard read point gives a model whose query
projection was trained for an input it is no longer given: fluent output from weights that no
longer match their wiring, and nothing that fails.

So the read point belongs to the checkpoint, and the config file is where a checkpoint says
things about itself:

    "afd_q_shift_layers": 1,
    "afd_coverage": "all"

Either at the top level or inside `text_config`; both are read, the text config wins, because on
a vision-language wrapper the text stack is what was converted.

## The resolution, and why the command line only warns

    config says N, nothing on the command line     serve at N. The checkpoint knows.
    nothing anywhere                               serve at 0. An unconverted checkpoint.
    command line says M, config says the same      fine, said twice.
    command line says M, config says N != M        WARN and use M. An override is legitimate --
                                                   measuring what a shift costs on an unconverted
                                                   checkpoint is exactly how the study's
                                                   forward-only numbers were taken -- but it is
                                                   also how someone serves a repaired checkpoint
                                                   wrongly, and those two look identical from
                                                   here. The warning names both readings.
    command line says M, checkpoint is unconverted  info, not a warning. Nothing is being
                                                   contradicted.

The command line cannot be silently right: `--afd-q-shift-layers` defaults to None, meaning "take
the checkpoint's", and 0 is a real value meaning "serve the standard wiring even if the checkpoint
was converted". A default of 0 would have made "unset" and "explicitly standard" the same, and the
override warning would never fire for the case it exists for.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SHIFT_KEY = "afd_q_shift_layers"
COVERAGE_KEY = "afd_coverage"


def _from_config(hf_config, key: str):
    """The value a checkpoint states about itself, text config first."""
    text = getattr(hf_config, "text_config", None)
    if text is not None and hasattr(text, key):
        return getattr(text, key)
    if hasattr(hf_config, key):
        return getattr(hf_config, key)
    return None


def resolve_shift(requested, hf_config) -> int:
    """The read point to serve at, from the checkpoint and whatever the caller asked for."""
    stated = _from_config(hf_config, SHIFT_KEY)
    if stated is not None and not isinstance(stated, int):
        raise TypeError(
            f"the checkpoint states {SHIFT_KEY}={stated!r}, which is not a layer count. A "
            f"checkpoint that cannot say where its query is read from should say nothing."
        )

    if requested is None:
        if stated is None:
            return 0
        logger.info(
            "afd: serving at the checkpoint's own read point, %s layer(s) back "
            "(%.1f layers, %s half-layers)",
            stated, stated - 0.5 if stated else 0.0, max(2 * stated - 1, 0),
        )
        return stated

    if stated is None:
        logger.info(
            "afd: --afd-q-shift-layers=%s on a checkpoint that states no read point. Its weights "
            "were not repaired for this wiring, so this measures what the rewiring costs before "
            "any repair.",
            requested,
        )
        return requested

    if requested == stated:
        return requested

    logger.warning(
        "afd: --afd-q-shift-layers=%s OVERRIDES the checkpoint's own %s. Two things look like "
        "this and only one is intended: measuring a shift the checkpoint was not repaired for, "
        "or serving a repaired checkpoint at the wrong read point -- in which case its query "
        "projection is being given an input it was not trained on, and the output will read "
        "fluently and be wrong. Serving at %s.",
        requested, stated, requested,
    )
    return requested


def resolve_coverage(requested, hf_config) -> str:
    """Which layers the shift reaches, resolved the same way."""
    stated = _from_config(hf_config, COVERAGE_KEY)
    if stated is not None and stated not in ("all", "softmax"):
        raise ValueError(
            f"the checkpoint states {COVERAGE_KEY}={stated!r}; it is \"all\" or \"softmax\""
        )
    if requested is None:
        return stated if stated is not None else "all"
    if stated is not None and requested != stated:
        logger.warning(
            "afd: --afd-coverage=%s overrides the checkpoint's own %s. The two coverages cost "
            "differently -- the study measured +0.0181 bits per byte at the softmax layers alone "
            "against +0.0211 at all of them -- so a repaired checkpoint served at the other one "
            "is not the model that was repaired. Serving at %s.",
            requested, stated, requested,
        )
    return requested


def stamp(config_dict: dict, shift: int, coverage: str) -> dict:
    """Write the read point into a config a conversion is about to save.

    Used by whatever repairs a checkpoint: the shift it was repaired at travels with the weights,
    so nobody has to remember it, and `resolve_shift` can warn when someone contradicts it.
    """
    out = dict(config_dict)
    target = out.get("text_config")
    if isinstance(target, dict):
        target = dict(target)
        target[SHIFT_KEY] = shift
        target[COVERAGE_KEY] = coverage
        out["text_config"] = target
    else:
        out[SHIFT_KEY] = shift
        out[COVERAGE_KEY] = coverage
    return out
