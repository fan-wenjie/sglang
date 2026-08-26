"""The query-shift arm's half of the pool's pushed configuration.

Transport, not resolution: the pool folds its REQUEST -- the shift layers its flag asked
for -- into the HELLO, and a host adopts the same request
before anything is installed. Each end then resolves through the checkpoint exactly as
before, from inputs the handshake has made identical, which is why this file may read the
raw setting: it moves the request between the two resolutions and serves at neither. The
arrangement word, computed from the RESOLVED value on both ends, remains the proof that the
two resolutions agreed.
"""

from __future__ import annotations

# The pool's word for the shift layers, once adopted. Module state rather than a write into
# server_args because server_args is the pristine startup record and is frozen after
# resolution -- the sanctioned reader is `checkpoint.requested_shift`, which prefers the
# host's own flag (already checked against this for contradiction) and falls back here.
_ADOPTED_LAYERS: list = [None]


def adopted_layers():
    """What the pool pushed for the shift layers, or None before any adoption."""
    return _ADOPTED_LAYERS[0]


def pushed_config(server_args) -> dict:
    """This arm's settings as the pool pushes them, raw, for the host to resolve itself."""
    return {
        "query_shift_layers": getattr(server_args, "afd_query_shift_layers", None),
    }


def adopt_config(cfg: dict, server_args) -> None:
    """Take the pool's word for the shift layers, on the host.

    A host that was ALSO given the setting, explicitly and differently, is refused by
    name: its flag is not configuration, it is a contradiction with the end that owns the
    configuration. An explicitly equal setting passes -- redundant, not wrong.
    """
    layers = cfg.get("query_shift_layers")
    mine = getattr(server_args, "afd_query_shift_layers", None)
    if mine is not None and mine != layers:
        raise RuntimeError(
            f"the pool serves --afd-query-shift-layers={layers} and this host was "
            f"started with {mine}. The pool owns the configuration; drop the host's flag."
        )
    _ADOPTED_LAYERS[0] = layers
