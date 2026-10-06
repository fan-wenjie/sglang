"""The dsa arm's tier-1 hot set: which archived rows stay resident.

Separated from `vestigekv.eviction` because the two arms select on different
signals and, now, with different policies. The mla arm keeps `select_kept`'s mask
over sidecar sigma with a proportional m; this arm keeps a CONSTANT number of
index-k groups, ranked by group-space sigma, and needs the count to be a host-side
expression because the prefill path runs it per layer per chunk.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.vestigekv.dsa import defaults as D


def kept_rows(
    sigma: torch.Tensor,
    row_slots: torch.Tensor,
    rho: float,
    closed: int,
    sinks: int = D.SINKS,
    return_pos: bool = False,
    m_fixed: int | None = None,
):
    """The hot set's rows, with a size the host already knows.

    `row_slots[:closed][mask.nonzero()]` costs a device sync per call, because
    nonzero's output shape is data-dependent and the caller then reads it (to
    size the table write, and the bound the CSR pack is built from). On the
    prefill path that call runs per MLA layer per chunk -- 704 syncs for a 32k
    prompt at a 512-token chunk, each draining whatever the chunk has queued.

    So the sinks are taken out of the ranking rather than unioned into it: the
    count becomes exactly `sinks + m + tail`, a host-side expression. The row set
    is a SUPERSET of the mask's -- the two differ only when a sink would have won
    a top-m place, where this keeps the displaced non-sink row as well, at most
    `sinks` rows more out of ~m.

    `m_fixed` is the constant hot set: the count stays
    `n_sink + min(m, closed - n_sink) + tail`, so it is still host-side and still
    needs no sync, but it no longer tracks the context the way rho * closed does.
    """
    m = m_fixed if m_fixed is not None else max(1, round(rho * closed))
    n_sink = min(sinks, closed)
    ranked = sigma[:closed].clone()
    ranked[:n_sink] = float("-inf")  # kept anyway; do not let them win a place
    m = min(m, closed - n_sink)
    picked = (
        ranked.topk(m).indices.sort().values
        if m > 0
        else ranked.new_empty(0, dtype=torch.long)
    )
    pos = torch.cat([torch.arange(n_sink, device=sigma.device), picked])
    tail = torch.arange(closed, row_slots.shape[0], device=row_slots.device)
    rows = torch.cat([row_slots[:closed][pos], row_slots[closed:]])
    if return_pos:
        # The POSITIONS, not just the rows: a consumer that has to get from a
        # kept row back to its request-relative position would otherwise invert
        # the page table, and the position is free here because the ranking
        # produced it.
        return rows, torch.cat([pos, tail])
    return rows
