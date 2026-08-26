"""The pool's word on the arrangement, pushed to every host at the HELLO.

A host does not configure the arrangement and reads no file for it. It connects, the pool
pushes its configuration -- stamped with a schema version and with the CODE version of the
build that produced it -- and the host validates both stamps and takes the settings from
the frame. One end owns the configuration, so the two ends cannot drift: the alternative
was lived twice on this branch. A flag renamed between two builds left the host honouring
an old default against a pool honouring a new one, and the flight measured a configuration
nobody had asked for; and a whole family of accuracy arms once compared two ends whose
code disagreed about what a setting meant. Both were mismatches a startup handshake can
refuse by name.

The shared half here moves a dict and checks the stamps. What is IN the dict beyond the
stamps belongs to the arms: each contributes its settings through `pushed_config` on its
factory and takes them back through `adopt_config`, the same shape `arrangement_word`
already uses. The arrangement word stays -- after adoption the two words must agree, so the
word is the proof the adoption worked, not a competitor to it.

The payload crosses as JSON bytes widened to int64, one byte a column, because the wire
moves 2-D tensors of its four dtypes and nothing else. A kilobyte of configuration is 8 KiB
once, at startup, on a wire that moves megabytes a token.
"""

from __future__ import annotations

import json
import logging
import pathlib
import subprocess

import torch

logger = logging.getLogger(__name__)

# the schema stamp. Bump it when the meaning of a field changes -- a host reading a stamp it
# does not know refuses the pool rather than guessing at the fields.
CONFIG_VERSION = 1

_code_version: str | None = None


def code_version() -> str:
    """One string that names the code this process runs: package version plus git head.

    Both ends compute it the same way and the host refuses a pool whose string differs. The
    git head is what makes it sharp -- two checkouts of one package version are exactly the
    drift this exists to catch -- and a build outside a repository degrades to the package
    version alone, which still catches a release mismatch.
    """
    global _code_version
    if _code_version is None:
        import sglang

        version = getattr(sglang, "__version__", "unknown")
        head = ""
        try:
            head = subprocess.run(
                [
                    "git",
                    "-C",
                    str(pathlib.Path(sglang.__file__).resolve().parent),
                    "rev-parse",
                    "HEAD",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except Exception:  # noqa: BLE001 -- no git is a configuration, not an error
            head = ""
        _code_version = f"{version}+{head[:12]}" if head else str(version)
    return _code_version


def pool_config(server_args, model=None) -> dict:
    """Everything a host must agree with this pool about, stamped."""
    from sglang.srt.afd import arms
    from sglang.srt.afd.checkpoint import effective_model_path

    cfg = {
        "config_version": CONFIG_VERSION,
        "code": code_version(),
        "model": pathlib.Path(str(effective_model_path(server_args))).name,
        # None is "unset": the pool resolves it to the default here, so the host adopts a
        # decided value and never re-derives the default on its own
        "transfer": getattr(server_args, "afd_transfer_backend", None) or "nccl",
        # what the host BUILDS: the family's class, or the attention-service
        # skeleton that knows only the layer kinds and their widths
        "host_model": (
            "skeleton" if getattr(server_args, "afd_host_skeleton", False) else "family"
        ),
    }
    if model is not None:
        # what a host must know to BUILD, named without any model's words
        from sglang.srt.afd.manifest import MANIFEST_KEY, manifest_of

        cfg[MANIFEST_KEY] = manifest_of(model)
    for _name, factory in sorted(arms._ARMS.items()):
        describe = getattr(factory, "pushed_config", None)
        if describe is not None:
            cfg.update(describe(server_args))
    return cfg


def encode_config(cfg: dict) -> torch.Tensor:
    payload = json.dumps(cfg, sort_keys=True).encode()
    return torch.tensor([list(payload)], dtype=torch.int64)


def decode_config(tensor: torch.Tensor) -> dict:
    return json.loads(bytes(int(x) for x in tensor.reshape(-1).tolist()).decode())


_ADOPTED = [False]
_ADOPTED_TRANSFER = ["tcp"]
_ADOPTED_CFG: list = [None]


def adopted_value(key: str, default=None):
    """One setting from the adopted configuration, or the default before adoption."""
    cfg = _ADOPTED_CFG[0]
    return default if cfg is None else cfg.get(key, default)


def adopted_transfer() -> str:
    """The transfer backend the pool announced. "tcp" before any adoption."""
    return _ADOPTED_TRANSFER[0]


def adopt_from_the_pool(timeout_s: float = 60.0) -> None:
    """Fetch the pool's word over one short-lived connection and adopt it. Idempotent.

    Called from the LOADER, before any weight is allocated, because the loader is the first
    consumer of the settings being adopted -- which classes to build without storage is
    itself configuration. The model transform and the install path call it again and it
    no-ops; a pool that is still starting is waited for inside the timeout, and one that
    cannot be reached fails the host at load, which is minutes earlier and one process
    quieter than the first frame's failure.
    """
    if _ADOPTED[0]:
        return
    import socket
    import time

    from sglang.srt.afd.arms import load
    from sglang.srt.afd.protocol import OP_HELLO, Frame, decode, send_frame
    from sglang.srt.runtime_context import get_disagg, get_server_args

    # the arms must exist before the dict is dispatched to them: this runs in a process
    # sglang spawned, where a registry populated in the parent is empty, and an adoption
    # over an empty registry validates the stamps and hands the settings to nobody
    load()
    disagg = get_disagg()
    if not (disagg.afd_mode == "host" and disagg.afd_pool_addr):
        return
    address = disagg.afd_pool_addr
    host, _, port = address.rpartition(":")
    deadline = time.monotonic() + timeout_s
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=10.0) as sock:
                send_frame(sock, Frame(0, 0, (torch.zeros(1, 1),), OP_HELLO))
                frame = decode(sock)
            break
        except OSError as e:
            last = e
            time.sleep(2.0)
    else:
        raise RuntimeError(
            f"no pool answered at {address} within {timeout_s:.0f}s ({last!r}). The host "
            f"adopts its configuration from the pool, so the pool must be up first."
        )
    if frame is None or frame.op != OP_HELLO or len(frame.tensors) < 2:
        raise RuntimeError(
            f"the pool at {address} pushed no configuration at the HELLO. It runs a build "
            f"from before the pool owned the configuration; align the two ends' code."
        )
    adopt(decode_config(frame.tensors[1]), get_server_args())
    _ADOPTED[0] = True


def adopt(cfg: dict, server_args) -> None:
    """Validate the stamps, then let every arm take the pool's word for its own settings.

    Order matters: the stamps first, because a field's meaning is only defined once the two
    ends are known to run the same schema and the same code. An arm that finds the host was
    ALSO given a setting, explicitly and differently, refuses by name rather than silently
    preferring either end -- the host's flag is not configuration, it is a contradiction.
    """
    from sglang.srt.afd import arms

    stamp = cfg.get("config_version")
    if stamp != CONFIG_VERSION:
        raise RuntimeError(
            f"the pool pushes configuration schema {stamp} and this host reads "
            f"{CONFIG_VERSION}. The two ends run builds whose handshakes disagree; align the "
            f"code before the settings can even be compared."
        )
    mine = code_version()
    theirs = cfg.get("code")
    if theirs != mine:
        raise RuntimeError(
            f"the pool runs {theirs} and this host runs {mine}. Two ends of one model must "
            f"run the same code -- a flight across that difference once measured a "
            f"configuration nobody had asked for. Check out the same commit on both ends."
        )
    from sglang.srt.afd.checkpoint import effective_model_path
    from sglang.srt.afd.model_files import is_pool_path

    mine = effective_model_path(server_args)
    my_model = pathlib.Path(str(mine)).name
    if is_pool_path(mine):
        # a pool-provisioned host has no checkpoint of its own to disagree with:
        # its papers, its weights, and now its name are all the pool's
        logger.info(
            "afd host: serving %r -- the pool's checkpoint, adopted along with "
            "everything else",
            cfg.get("model"),
        )
    elif cfg.get("model") != my_model:
        raise RuntimeError(
            f"the pool serves {cfg.get('model')!r} and this host loaded {my_model!r}. "
            f"The host's attention would run one checkpoint's layers against another's "
            f"feed-forward and the output would be fluent garbage rather than an error."
        )
    transfer = cfg.get("transfer", "tcp")
    mine_transfer = getattr(server_args, "afd_transfer_backend", None)
    if mine_transfer is not None and mine_transfer != transfer:
        raise RuntimeError(
            f"the pool serves --afd-transfer-backend={transfer} and this host was started "
            f"with {mine_transfer}. The pool owns the configuration; drop the host's flag."
        )
    _ADOPTED_TRANSFER[0] = transfer
    _ADOPTED_CFG[0] = dict(cfg)
    for _name, factory in sorted(arms._ARMS.items()):
        take = getattr(factory, "adopt_config", None)
        if take is not None:
            take(cfg, server_args)
    logger.info(
        "afd host: configuration adopted from the pool -- %s",
        {k: v for k, v in cfg.items() if k not in ("config_version",)},
    )
