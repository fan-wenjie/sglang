"""Where a converted checkpoint's read point comes from, and who may override it.

A checkpoint repaired at one read point must be SERVED at that read point. Fine-tuning W_q to
read h_{l-1} and then serving the result at the standard read point gives a model whose query
projection was trained for an input it is no longer given: fluent output from weights that no
longer match their wiring, and nothing that fails.

So the read point belongs to the checkpoint, and the config file is where a checkpoint says
things about itself:

    "query_shift_layers": 1,
    "query_shift_coverage": "all"

Either at the top level or inside `text_config`; both are read, the text config wins, because on
a vision-language wrapper the text stack is what was converted.

## The resolution, and why the command line only warns

    config says N, nothing on the command line     serve at N. The checkpoint knows. Info, since
                                                   this is the one routine path and a routine
                                                   path that warns teaches people to skip warnings.
    nothing anywhere                               serve at 0, silently. An unconverted
                                                   checkpoint served the way it was trained.
    command line says M, config says the same      fine, said twice. Silent.
    command line says M != 0, config says nothing  WARN. The weights were never repaired for this
                                                   read point, so the model being served is one
                                                   nobody trained -- correct as a measurement of
                                                   what the rewiring costs before repair, which is
                                                   how the study's forward-only numbers were
                                                   taken, and wrong as a deployment.
    command line says M, config says N != M        WARN. Either measuring an unrepaired shift or
                                                   serving a repaired checkpoint at the wrong read
                                                   point, and those two look identical from here.
                                                   The warning names both readings.

Every path that serves weights at a read point they were not repaired for warns; nothing else
does. A warning here means the command line asked for something only a specific intent wants, and
if that intent is a measurement campaign the line is the record of it.

The command line cannot be silently right: the derived override defaults to None, meaning "take
the checkpoint's", and 0 is a real value meaning "serve the standard wiring even if the checkpoint
was converted". A default of 0 would have made "unset" and "explicitly standard" the same, and the
override warning would never fire for the case it exists for.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# A checkpoint's config is the CHECKPOINT's vocabulary, not this subsystem's. It spells things
# out -- num_attention_heads, not num_attn_heads -- so these do too, and neither carries the
# server's own flag namespace.
SHIFT_KEY = "query_shift_layers"
COVERAGE_KEY = "query_shift_coverage"

# What these were called before the rename. A checkpoint that states a read point must never have
# it silently ignored -- that is the one failure this whole file exists to prevent -- so an old
# name is refused rather than skipped over.
RETIRED_KEYS = {"afd_query_shift_layers": SHIFT_KEY, "afd_coverage": COVERAGE_KEY}


def _refuse_retired_keys(hf_config) -> None:
    """A config written against the old spelling is refused, not ignored.

    Ignoring it would serve a repaired checkpoint at the standard read point -- its query
    projection handed an input it was never trained on -- and the output would read fluently. A
    rename that can silently do that is worse than no rename.
    """
    for old, new in RETIRED_KEYS.items():
        if _has(hf_config, old) and not _has(hf_config, new):
            raise ValueError(
                f"this checkpoint states {old!r}, which was renamed to {new!r}. It is refused "
                f"rather than ignored: a checkpoint that says where its query is read from must "
                f"not be served as though it had said nothing, because the result reads fluently "
                f"and is a model nobody trained."
            )


def _has(hf_config, key: str) -> bool:
    text = getattr(hf_config, "text_config", None)
    return (text is not None and hasattr(text, key)) or hasattr(hf_config, key)


def _from_config(hf_config, key: str):
    """The value a checkpoint states about itself, text config first.

    getattr with a default is defensive access when the field is always there. Here its absence IS
    the answer -- an unconverted checkpoint says nothing about a read point, and that is the case
    this function exists to report. The alternative the rule prefers, always setting the field,
    would mean writing into a HuggingFace config class this code does not own.
    """
    text = getattr(hf_config, "text_config", None)
    if text is not None and hasattr(text, key):
        return getattr(text, key)
    if hasattr(hf_config, key):
        return getattr(hf_config, key)
    return None


# What `resolve_shift` last returned in this process. One process serves one checkpoint, so every
# reader that asks after install gets the value the model is actually being served at -- including
# the two-end arrangement handshake, which has server args but no checkpoint and would otherwise
# encode the raw setting while the model ran at the checkpoint's.
_RESOLVED: int | None = None


def resolved_shift(server_args=None) -> int:
    """The read point this process serves at.

    Once a model has been loaded this is what `resolve_shift` decided, which is the authority: it
    saw the loaded config and it warned about whatever it had to warn about.

    Before that -- the loader hooks run before any model exists, and they decide what is
    allocated -- it is read from the checkpoint's own config file, which is the same source
    `resolve_shift` would read and the same answer. Passing `server_args` is what makes that
    possible; a caller with neither a resolution nor args is asking a question that has no answer
    yet, and gets a refusal rather than a 0.
    """
    if _RESOLVED is not None:
        return _RESOLVED
    if server_args is None:
        raise RuntimeError(
            "the read point has not been resolved yet and no server args were given to read it "
            "from. Whoever asked is running before the model is loaded and holds nothing that "
            "says which checkpoint."
        )
    req = requested_shift(server_args)
    if req is not None:
        return int(req)
    return int(stated_shift(effective_model_path(server_args)) or 0)


def requested_shift(server_args):
    """What was ASKED for: the host's own flag, or failing that the pool's pushed word.

    The flag wins only because a contradiction between the two was already refused at
    adoption, so by here they agree or only one exists. Callers that used to read the raw
    flag come through this instead, which is what lets a host that was given nothing serve
    the arrangement the pool pushed -- and the resolution below stays the one place the
    checkpoint has its say."""
    if (
        server_args is not None
        and getattr(server_args, "afd_query_shift_layers", None) is not None
    ):
        return getattr(server_args, "afd_query_shift_layers", None)
    try:
        from sglang.srt.afd_query_shift.pushed import adopted_layers
    except ImportError:  # the derived package is absent: nothing can have been adopted
        return None

    return adopted_layers()


def effective_model_path(server_args=None) -> str:
    """The checkpoint path this process actually serves.

    `ServerArgs` carries the user's RAW input; what resolution decided lives in
    the config bag, and on an AFD host the two differ BY DESIGN -- a host whose
    path does not exist is given `pool://IP:PORT` so its config and tokenizer
    come from the pool rather than a filesystem. Reading the record in a
    published process therefore answers with a path that was never loaded, which
    is how the identity guard came to refuse a host for serving the checkpoint it
    was actually serving.

    So: the bag where one exists (every published process), and the record where
    one does not -- the resolution pipeline itself, and the tests that hand one in.
    """
    from sglang.srt.runtime_context import get_model

    try:
        return get_model().model_path
    except ValueError:  # this process has not published; resolution is still running
        if server_args is None:
            raise
        return server_args.model_path


def effective_model_path(server_args=None) -> str:
    """The checkpoint path this process actually serves.

    `ServerArgs` carries the user's RAW input; what resolution decided lives in
    the config bag, and on an AFD host the two differ BY DESIGN -- a host whose
    path does not exist is given `pool://IP:PORT` so its config and tokenizer
    come from the pool rather than a filesystem. Reading the record in a
    published process therefore answers with a path that was never loaded, which
    is how the identity guard came to refuse a host for serving the checkpoint it
    was actually serving.

    So: the bag where one exists (every published process), and the record where
    one does not -- the resolution pipeline itself, and the tests that hand one in.
    """
    from sglang.srt.runtime_context import get_model

    try:
        return get_model().model_path
    except ValueError:  # this process has not published; resolution is still running
        if server_args is None:
            raise
        return server_args.model_path


def stated_shift(model_path: str):
    """The read point the checkpoint at `model_path` declares, or None.

    A FILE read, not a loaded config, because this is asked before the model exists.

    Both spellings the resolution accepts, and the text config wins for the same reason -- on a
    vision-language wrapper the text stack is what was converted. Unreadable, absent or malformed
    means None: a checkpoint that cannot say says nothing.
    """
    import json
    import os

    try:
        with open(os.path.join(model_path, "config.json")) as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    for scope in (config.get("text_config"), config):
        if isinstance(scope, dict) and scope.get(SHIFT_KEY) is not None:
            return scope[SHIFT_KEY]
    return None


def states_a_read_point(model_path: str) -> bool:
    """Whether the checkpoint at `model_path` declares a read point of its own.

    A FILE read, not a loaded config, because this is asked before the model exists: the arm has
    to be chosen before anything is loaded, and choosing it is the same question as whether the
    checkpoint was repaired for it.

    Both spellings the resolution accepts, and the text config wins for the same reason -- on a
    vision-language wrapper the text stack is what was converted. Unreadable, absent or malformed
    means no: a checkpoint that cannot say says nothing, and the standard arrangement is what a
    checkpoint that says nothing gets.
    """
    return stated_shift(model_path) is not None


def serves_linear_layers(model_path: str) -> bool:
    """Whether the checkpoint at `model_path` is of the family this arm's cut serves.

    A FILE read, before the model exists, for the same reason `states_a_read_point`
    reads one: the arrangement is chosen before anything is loaded. The marker is the
    config's own `layer_types` naming a linear attention -- the same declaration the
    span groups by once the model is up. Unreadable, absent or malformed means no, and
    a checkpoint that is not of the family gets the base arrangement untouched.
    """
    import json
    import os

    try:
        with open(os.path.join(model_path, "config.json")) as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(config, dict):
        return False
    for scope in (config.get("text_config"), config):
        if isinstance(scope, dict) and isinstance(scope.get("layer_types"), list):
            return "linear_attention" in scope["layer_types"]
    return False


def resolve_shift(requested, hf_config) -> int:
    """The read point to serve at, from the checkpoint and whatever the caller asked for."""
    global _RESOLVED
    _refuse_retired_keys(hf_config)
    stated = _from_config(hf_config, SHIFT_KEY)
    if stated is not None and not isinstance(stated, int):
        raise TypeError(
            f"the checkpoint states {SHIFT_KEY}={stated!r}, which is not a layer count. A "
            f"checkpoint that cannot say where its query is read from should say nothing."
        )

    if requested is None:
        if stated is None:
            _RESOLVED = 0
            return _RESOLVED
        logger.info(
            "afd: serving at the checkpoint's own read point, %s layer(s) back "
            "(%.1f layers, %s half-layers)",
            stated,
            stated - 0.5 if stated else 0.0,
            max(2 * stated - 1, 0),
        )
        _RESOLVED = stated
        return _RESOLVED

    if stated is None:
        if requested == 0:
            _RESOLVED = 0
            return _RESOLVED
        logger.warning(
            "afd: a requested read point of %s on a checkpoint that states none. Its query "
            "projection was trained to read x_l and is being given h_(l-%s), so this serves a "
            "model nothing has repaired -- valid as a measurement of what the rewiring costs "
            "before repair, and wrong as a deployment. Serving at %s.",
            requested,
            requested,
            requested,
        )
        _RESOLVED = requested
        return _RESOLVED

    if requested == stated:
        _RESOLVED = requested
        return _RESOLVED

    logger.warning(
        "afd: a requested read point of %s OVERRIDES the checkpoint's own %s. Two things look like "
        "this and only one is intended: measuring a shift the checkpoint was not repaired for, "
        "or serving a repaired checkpoint at the wrong read point -- in which case its query "
        "projection is being given an input it was not trained on, and the output will read "
        "fluently and be wrong. Serving at %s.",
        requested,
        stated,
        requested,
    )
    _RESOLVED = requested
    return _RESOLVED


def resolve_coverage(requested, hf_config) -> str:
    """Which layers the shift reaches, resolved the same way."""
    _refuse_retired_keys(hf_config)
    stated = _from_config(hf_config, COVERAGE_KEY)
    if stated is not None and stated not in ("all", "softmax"):
        raise ValueError(
            f'the checkpoint states {COVERAGE_KEY}={stated!r}; it is "all" or "softmax"'
        )
    if requested is None:
        return stated if stated is not None else "all"
    if stated is not None and requested != stated:
        logger.warning(
            "afd: --afd-coverage=%s overrides the checkpoint's own %s. The two coverages cost "
            "differently -- the study measured +0.0181 bits per byte at the softmax layers alone "
            "against +0.0211 at all of them -- so a repaired checkpoint served at the other one "
            "is not the model that was repaired. Serving at %s.",
            requested,
            stated,
            requested,
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
