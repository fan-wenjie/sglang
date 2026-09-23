"""Model geometry VestigeKV runs against, derived once from the HF config.

Two distinct widths hide behind "the sidecar" in the Kimi Linear case and
must stay distinct for any other model: `side_dim` is the un-roped branch
stored inside the latent row and scored by attention (the tier-2 exact
summand), `sigma_dim` is the salience channel tier-1 ranks rows by. On Kimi
Linear both are the same 64-dim row tail; a rope-less MLA (side_dim == 0)
takes its salience channel from a separate per-layer pool instead.
"""

from __future__ import annotations

import msgspec

from sglang.srt.layers.attention.vestigekv import defaults as D


class Geometry(msgspec.Struct, frozen=True, kw_only=True):
    kv_lora_rank: int
    """Content half of the latent row; also the value vector."""
    side_dim: int
    """Un-roped branch inside the latent row, scored by attention (0 = none)."""
    qk_head_dim: int
    """Per-head q.k width the model's softmax scale is derived from."""
    sigma_dim: int
    """Width of the tier-1 salience channel."""
    sigma_in_row: bool
    """True: sigma reads the latent row tail; False: a separate side pool."""

    @property
    def latent_dim(self) -> int:
        """Latent row width as stored in the KV pool; also the expanded query width."""
        return self.kv_lora_rank + self.side_dim

    @property
    def attn_scale(self) -> float:
        return self.qk_head_dim**-0.5

    @property
    def sigma_offset(self) -> int:
        """Column of the sigma channel inside its source buffer."""
        return self.kv_lora_rank if self.sigma_in_row else 0

    @classmethod
    def from_hf_config(cls, cfg) -> Geometry:
        side = int(cfg.qk_rope_head_dim)
        if side > 0:
            # The decoupled branch is the salience channel. Kimi Linear leaves
            # it unrotated (mla_use_nope), so the row tail itself is scored;
            # a RoPE-MLA model (DeepSeek) rotates it by position, so the
            # pre-rotation copy the model files (RopeSalienceKey) is held in
            # a per-layer side pool and scored there. The certificate reads
            # the rotated tail off the row in both cases: that is the logit.
            return cls(
                kv_lora_rank=int(cfg.kv_lora_rank),
                side_dim=side,
                qk_head_dim=int(cfg.qk_nope_head_dim) + side,
                sigma_dim=side,
                # DeepSeek-family configs rotate the branch; Kimi Linear's is
                # NoPE unless its config says otherwise (mla_use_nope=False).
                sigma_in_row=(
                    getattr(cfg, "model_type", None) not in ("deepseek_v2", "deepseek_v3")
                    and getattr(cfg, "mla_use_nope", True) is not False
                ),
            )
        # Rope-less MLA: the DSA indexer key (never rotated when rope is absent)
        # is the salience channel, held in its own per-layer pool.
        index_head_dim = getattr(cfg, "index_head_dim", None)
        if not index_head_dim:
            raise ValueError(
                "vestigekv needs a salience channel: qk_rope_head_dim == 0 and "
                "the config declares no index_head_dim"
            )
        return cls(
            kv_lora_rank=int(cfg.kv_lora_rank),
            side_dim=0,
            qk_head_dim=int(cfg.qk_nope_head_dim),
            sigma_dim=int(index_head_dim),
            sigma_in_row=False,
        )


KIMI_LINEAR = Geometry(
    kv_lora_rank=D.KV_LORA_RANK,
    side_dim=D.SIDECAR_DIM,
    qk_head_dim=192,
    sigma_dim=D.SIDECAR_DIM,
    sigma_in_row=True,
)
"""The validated configuration; the default wherever a caller passes none."""
