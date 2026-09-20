"""Debug-only telemetry helpers (SGLANG_DEBUG_VESTIGEKV_STATS)."""


def hist_percentiles(counts, qs):
    """Nearest-rank percentiles of a histogram whose bin i counts value i.

    Rank ceil(q * total) over the cumulative counts; an empty histogram
    answers 0 for every q."""
    total = sum(counts)
    out = []
    for q in qs:
        if total == 0:
            out.append(0)
            continue
        rank = max(
            1, -(-int(q * total * 1_000_000) // 1_000_000)
        )  # ceil without float drift
        acc = 0
        value = len(counts) - 1
        for i, c in enumerate(counts):
            acc += c
            if acc >= rank:
                value = i
                break
        out.append(value)
    return out
