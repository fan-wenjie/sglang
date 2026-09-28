"""Tier-2 with no sketch: the build with the basis and its work deleted.

Branch-only recall scores on the sidecar and bounds the content block by
Cauchy-Schwarz, so its basis is the zero matrix. The env-flag form arrives
there by FITTING a rank-r PCA basis and then overwriting it with zeros, which
is the most expensive way to spell zero: a cusolver eigendecomposition per
layer per calibrated build, measured at ~4 ms/layer and ~20 ms/request on the
token path, plus a one-shot ~230 ms cusolver initialisation (recall_tier's own
note on the provisional path).

That cost is per REQUEST and does not grow with context, which is exactly the
term that sets where short-decode breaks even against dense: at short context
the archive is small, the scan is cheap, and what VestigeKV pays is this fixed
build. Deleting it moves the crossover left, which is the point of this file.

The basis still has to EXIST -- the pack stacks it into [P, r, 512] and the
certificate's residual norm is defined relative to it -- so what is deleted is
the factorisation, not the tensor. A zero basis makes ||(I - V'V)q|| the whole
query norm, which is the Cauchy-Schwarz bound branch-only recall certifies
against, so the numbers are the flag form's exactly.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier


class BranchRecallTier(RecallTier):
    """A RecallTier whose basis is zero by construction, never by overwrite."""

    def _operand_builder(self):
        from sglang.srt.layers.attention.vestigekv.branch_operand import (
            build_operands_branch,
        )

        return build_operands_branch

    def build(
        self,
        kbuf: torch.Tensor,
        row_slots: torch.Tensor,
        keep: torch.Tensor,
        q_cal: torch.Tensor,
        q_pos: torch.Tensor,
        conservative: bool = False,
        v_init: torch.Tensor | None = None,
        operands_from: RecallTier | None = None,
        diag: bool = False,
    ) -> dict:
        # Hand the zero basis in as v_init and build() takes its `v_init is not
        # None` arm, so neither the eigendecomposition nor the provisional
        # identity is ever constructed. An adopting build inherits the donor's
        # basis, which is this same zero matrix, so it needs nothing here.
        if v_init is None and operands_from is None:
            v_init = torch.zeros(
                self.r, D.KV_LORA_RANK, device=kbuf.device, dtype=torch.float32
            )
        return super().build(
            kbuf,
            row_slots,
            keep,
            q_cal,
            q_pos,
            conservative=conservative,
            v_init=v_init,
            operands_from=operands_from,
            diag=diag,
        )
