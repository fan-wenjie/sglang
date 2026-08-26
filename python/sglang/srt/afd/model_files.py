"""The model's papers, served by the pool and never touching the host's storage.

A weightless host still needs the checkpoint's PAPERS -- config.json, the
tokenizer, the generation defaults -- but it does not need a filesystem to hold
them: a host whose `model_path` is `pool://IP:PORT` fetches them as raw bytes
(a JSON manifest, then one uint8 tensor per file; no pickle) and builds its
config and tokenizer objects IN MEMORY, through the two loader seams that
recognise the scheme. Nothing is written anywhere: a host process runs with no
write permission at all, which is the point -- the machine's provisioning is a
pool address, and the least privilege a host can hold is the privilege it needs.

Fetched once per PROCESS and cached in this module: sglang spawns its workers,
a parent's cache does not cross, and a fetch-on-miss in whichever process asks
is what makes that boundary a non-event (~a few MiB per process, once).
"""

from __future__ import annotations

import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

POOL_SCHEME = "pool://"

# what a host's skeleton and tokenizer can need; never a weight
FILE_PATTERNS = (
    ".json",
    ".txt",
    ".jinja",
    ".model",
    ".tiktoken",
)
SIZE_CAP = 256 * 1024 * 1024

# per-process: {pool address: {file name: bytes}}
_PAPERS: dict = {}


def is_pool_path(path) -> bool:
    return isinstance(path, str) and path.startswith(POOL_SCHEME)


def servable_files(model_path: str) -> list[str]:
    """The small files under the checkpoint root, in sorted order. Never weights."""
    names = []
    total = 0
    for name in sorted(os.listdir(model_path)):
        full = os.path.join(model_path, name)
        if not os.path.isfile(full):
            continue
        if not any(name.endswith(sfx) for sfx in FILE_PATTERNS):
            continue
        if "safetensors" in name:
            continue
        size = os.path.getsize(full)
        if size == 0:
            continue
        if total + size > SIZE_CAP:
            logger.warning(
                "afd pool: %s left out of the file push -- the %d MiB cap is "
                "reached. A host bootstrapping from this pool will not have it.",
                name,
                SIZE_CAP // 2**20,
            )
            continue
        total += size
        names.append(name)
    return names


def files_reply(model_path: str) -> tuple:
    """The manifest and the bytes, as the wire's tensors. The pool's half."""
    names = servable_files(model_path)
    # (1, n): a frame carries (rows, columns), and bytes are one row of them
    manifest = torch.frombuffer(
        bytearray(json.dumps(names).encode()), dtype=torch.uint8
    ).clone()
    out = [manifest.view(1, -1)]
    for name in names:
        with open(os.path.join(model_path, name), "rb") as f:
            out.append(
                torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
                .clone()
                .view(1, -1)
            )
    return tuple(out)


def papers_from_reply(tensors) -> dict:
    """A files reply, decoded to {name: bytes}. No name may look like a path."""
    manifest = json.loads(bytes(tensors[0].numpy().tobytes()).decode())
    if len(tensors) != len(manifest) + 1:
        raise RuntimeError(
            f"the pool's file push carried {len(tensors) - 1} file(s) for a "
            f"manifest of {len(manifest)}. The reply is torn; refusing to build a "
            f"model from half its papers."
        )
    papers = {}
    for name, tensor in zip(manifest, tensors[1:]):
        if os.sep in name or name.startswith("."):
            raise RuntimeError(f"the pool pushed a file named {name!r}; refused.")
        papers[name] = tensor.numpy().tobytes()
    return papers


def papers_for(path: str) -> dict:
    """The pool's papers for a `pool://` path, fetched once per process."""
    addr = path[len(POOL_SCHEME) :]
    if addr in _PAPERS:
        return _PAPERS[addr]
    from sglang.srt.afd.pool_client import PoolClient
    from sglang.srt.afd.protocol import OP_FILES

    client = PoolClient(addr, 30.0, reconnect=False)
    try:
        got = client.collect_frame(
            client.issue_frame(0, 0, (torch.zeros(1, 1),), OP_FILES), "cpu"
        )
    finally:
        client.close()
    papers = papers_from_reply(got)
    _PAPERS[addr] = papers
    logger.info(
        "afd host: %d model file(s) fetched from the pool into memory -- this "
        "process holds no checkpoint and writes nothing",
        len(papers),
    )
    return papers


def _paper_json(path: str, name: str, required: bool = False):
    papers = papers_for(path)
    if name not in papers:
        if required:
            raise RuntimeError(
                f"the pool's file push has no {name}; a host cannot build its "
                f"skeleton without it."
            )
        return None
    return json.loads(papers[name].decode())


def pool_config_dict(path: str) -> dict:
    """The raw config.json dict for a `pool://` path."""
    return _paper_json(path, "config.json", required=True)


def pool_config(path: str, model_override_args=None):
    """The hf config for a `pool://` path, built from the pushed dict in memory."""
    from transformers import AutoConfig, PretrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    config_dict = _paper_json(path, "config.json", required=True)
    if model_override_args:
        config_dict.update(model_override_args)
    model_type = config_dict.get("model_type")
    try:
        # sglang's own families register into this mapping too (AutoConfig.register)
        cls = CONFIG_MAPPING[model_type]
    except KeyError:
        cls = PretrainedConfig
    del AutoConfig  # imported for the registration side effect only
    return cls.from_dict(config_dict)


def pool_generation_config(path: str):
    """The generation defaults for a `pool://` path, or None when none travelled."""
    from transformers import GenerationConfig

    got = _paper_json(path, "generation_config.json")
    return None if got is None else GenerationConfig.from_dict(got)


def pool_tokenizer(path: str, **kwargs):
    """The tokenizer for a `pool://` path, built in memory from the pushed JSON."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    papers = papers_for(path)
    if "tokenizer.json" not in papers:
        raise RuntimeError(
            "the pool's file push has no tokenizer.json; only fast tokenizers can "
            "be built without a filesystem, and this checkpoint did not carry one."
        )
    init_kwargs = _paper_json(path, "tokenizer_config.json") or {}
    init_kwargs.pop("tokenizer_class", None)
    init_kwargs.pop("added_tokens_decoder", None)
    special = _paper_json(path, "special_tokens_map.json") or {}
    for key, value in special.items():
        if isinstance(value, dict):
            value = value.get("content")
        init_kwargs.setdefault(key, value)
    init_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(papers["tokenizer.json"].decode()),
        **init_kwargs,
    )


def pool_processor(path: str):
    """The multimodal processor for a `pool://` path, assembled in memory.

    The same public pieces `AutoProcessor.from_pretrained` would read off disk --
    the image and video processor dicts, the tokenizer, the chat template --
    built from the pushed papers instead and handed to the family's processor
    class from the auto mapping. Byte-identical inputs, so an identical
    processor; nothing is fetched from a hub and nothing lands on a filesystem.
    """
    import inspect

    from transformers.models.auto import image_processing_auto, video_processing_auto
    from transformers.models.auto.processing_auto import PROCESSOR_MAPPING

    config = pool_config(path)
    if type(config) not in PROCESSOR_MAPPING:
        raise RuntimeError(
            f"no processor class is registered for {type(config).__name__}; a "
            f"host cannot assemble a multimodal processor for this family from "
            f"the pool's papers."
        )
    cls = PROCESSOR_MAPPING[type(config)]

    # through the loader seam, so the serving-side tokenizer patches apply here too
    from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

    parts: dict = {"tokenizer": get_tokenizer(path)}
    # transformers has spelled these helpers both ways across versions
    ip_from_name = getattr(
        image_processing_auto, "get_image_processor_class_from_name", None
    ) or getattr(image_processing_auto, "image_processor_class_from_name")
    vp_from_name = getattr(
        video_processing_auto, "get_video_processor_class_from_name", None
    ) or getattr(video_processing_auto, "video_processor_class_from_name")
    ip = _paper_json(path, "preprocessor_config.json")
    if ip and ip.get("image_processor_type"):
        parts["image_processor"] = ip_from_name(ip["image_processor_type"]).from_dict(
            ip
        )
    vp = _paper_json(path, "video_preprocessor_config.json")
    if vp and vp.get("video_processor_type"):
        parts["video_processor"] = vp_from_name(vp["video_processor_type"]).from_dict(
            vp
        )
    papers = papers_for(path)
    if "chat_template.jinja" in papers:
        parts["chat_template"] = papers["chat_template.jinja"].decode()

    accepted = inspect.signature(cls.__init__).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        parts = {k: v for k, v in parts.items() if k in accepted}
    return cls(**parts)
