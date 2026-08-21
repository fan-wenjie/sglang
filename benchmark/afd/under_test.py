"""The model every arm of a comparison runs, named once and checked before it is used.

Eight measurement tools each wrote the checkpoint path down. That is eight places to update when
the model changes and eight chances to update seven of them -- and the failure mode is the worst
kind, because a tool still left on the old checkpoint produces a complete set of numbers that
nobody can tell apart from a correct set. A baseline taken on a quantised checkpoint against an
experiment taken on an unquantised one is not a comparison of arrangements; it is a comparison of
checkpoints wearing an arrangement's name.

So the path comes from here, and this module refuses to hand back a checkpoint whose config
carries a `quantization_config` unless the caller has explicitly said it wants one.

## Why quantisation is refused rather than warned about

The whole subject of these measurements is a small difference: a rewiring that costs a few
hundredths of a bit and a schedule that costs milliseconds. Quantisation moves both by more than
the effect being measured. Nothing downstream would notice -- an FP8 run and a bfloat16 run both
produce fluent text and plausible timings -- so the only place it can be caught is before the
weights load.

    AFD_MODEL=/path/to/checkpoint    what to serve; no default, because a default here is a
                                     silent answer to the question the comparison is about
    AFD_ALLOW_QUANTISED=1            serve a quantised checkpoint anyway. For measurements that
                                     are ABOUT quantisation, where both arms carry it.
"""

from __future__ import annotations

import json
import os
import pathlib


class QuantisedCheckpoint(RuntimeError):
    """Raised when a comparison is about to run on weights that carry their own error."""


def model_path() -> str:
    """The checkpoint under test, from the environment, verified to be what it claims.

    No default. A default would let a tool run to completion against whichever checkpoint happened
    to be written down when it was authored, and report it under the name of the one the run was
    supposed to use.
    """
    path = os.environ.get("AFD_MODEL")
    if not path:
        raise RuntimeError(
            "AFD_MODEL is not set. Every arm of a comparison has to serve the same weights, and "
            "this used to be written into each tool separately -- so name the checkpoint once, "
            "in the environment, and let every tool read it from there."
        )
    config = pathlib.Path(path) / "config.json"
    if not config.exists():
        raise RuntimeError(f"no config.json under AFD_MODEL={path!r}")

    quantisation = json.loads(config.read_text()).get("quantization_config")
    if quantisation is not None and not os.environ.get("AFD_ALLOW_QUANTISED"):
        raise QuantisedCheckpoint(
            f"{path} carries a quantization_config ({', '.join(sorted(quantisation))}). The "
            f"effects these tools measure are smaller than what quantisation moves, and a "
            f"quantised arm against an unquantised one compares checkpoints rather than "
            f"arrangements. Set AFD_ALLOW_QUANTISED=1 if the measurement is about quantisation "
            f"and every arm carries it."
        )
    return path


def describe() -> dict:
    """What a run record should carry about the weights it served."""
    path = model_path()
    config = json.loads((pathlib.Path(path) / "config.json").read_text())
    text = config.get("text_config", config)
    return {
        "path": path,
        "quantised": config.get("quantization_config") is not None,
        "layers": text.get("num_hidden_layers"),
        "hidden_size": text.get("hidden_size"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "head_dim": text.get("head_dim"),
    }
