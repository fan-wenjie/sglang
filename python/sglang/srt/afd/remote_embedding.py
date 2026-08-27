"""The embedding lookup, moved to the pool, without touching sglang's shared route.

`embed_tokens` is 248,320 x 5120 -- 2.368 GiB in bf16, and with `lm_head` it is 89% of everything
the host still holds after #67. It is also the one weight the host reads exactly once a step,
whatever the context, so moving it costs nothing that scales.

## Why the forward is replaced rather than the caller changed

Two paths reach the embedding and BOTH go through this module:

    qwen3_5.py:1456     hidden_states = self.embed_tokens(input_ids)      the text path
    mm_utils.py:470     input_embeds = input_embedding(input_ids)         via get_input_embeddings()

Editing either caller would mean editing sglang's shared multimodal route, which every model takes.
Replacing one module's `forward` is what this arrangement does everywhere else, and it catches both
paths at once because both end at the same object.

## Why a placeholder, and why it is NaN

The ids are needed at `OP_SPAN_ENTER`, which is issued from inside the layer loop -- after this
returns. So this records them and returns something of the right shape for the loop to carry until
the first span replaces it.

Under the group cut nothing on the host reads that tensor: layers 0..2 are passengers whose forward
is a pass-through, and the first span sends ids instead of it. "Nothing reads it" is exactly the
kind of claim that rots silently, so the placeholder is NaN rather than zeros. A zero placeholder
that something did read would produce a plausible answer and a wrong one; NaN propagates and shows
up in the first token, which is where a wrong assumption should surface.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class PendingIds:
    """What the embedding recorded, waiting for the first span to send it.

    One slot, not a queue: the layer loop is single-threaded per forward, and the first span
    consumes what the embedding just recorded. A second forward before the span would be a bug
    elsewhere, so `take` refuses rather than returning stale ids.
    """

    def __init__(self) -> None:
        self.ids: torch.Tensor | None = None

    def record(self, ids: torch.Tensor) -> None:
        self.ids = ids.detach().reshape(-1, 1).to(torch.int64)

    def take(self, rows: int) -> torch.Tensor:
        if self.ids is None:
            raise RuntimeError(
                "the first span asked for token ids and the embedding recorded none. The host is "
                "configured to send ids rather than hidden states, so something reached the layer "
                "loop without passing through `embed_tokens` -- an `input_embeds` argument, or a "
                "multimodal path that builds the embedding itself."
            )
        ids, self.ids = self.ids, None
        if ids.shape[0] != rows:
            raise RuntimeError(
                f"the embedding recorded {ids.shape[0]} token id(s) and the span carries {rows} "
                f"row(s). They are the same tokens or the pool embeds the wrong ones, and nothing "
                f"downstream would say so."
            )
        return ids


def send_ids_instead(model, pending: PendingIds) -> None:
    """Make `embed_tokens` record its input and return a placeholder. Returns nothing to undo.

    Called once at install. The module keeps its weight for now -- `absent_classes` is what stops
    it being allocated, and that is a separate decision so this can be measured on its own.
    """
    embed = model.model.embed_tokens
    hidden = (
        model.config.text_config.hidden_size
        if hasattr(model.config, "text_config")
        else model.config.hidden_size
    )

    def record(input_ids: torch.Tensor) -> torch.Tensor:
        pending.record(input_ids)
        return torch.full(
            (input_ids.reshape(-1).shape[0], hidden),
            float("nan"),
            device=input_ids.device,
            dtype=model.dtype if hasattr(model, "dtype") else torch.bfloat16,
        )

    embed.forward = record
    logger.info(
        "afd host: the embedding runs on the pool. This side records token ids and carries a NaN "
        "placeholder to the first span, which sends the ids instead of it."
    )
