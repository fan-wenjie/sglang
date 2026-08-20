"""Run a cache pool: histories, sweeps, appends. No weights, no model, no loader.

    python -m sglang.srt.afd.cache_pool_main --model <path> --port 9200 --max-context 262144

It reads the model's CONFIG for five numbers -- query heads, key-value heads, head dimension and
the softmax scale -- and never opens a checkpoint. That is the point of the split: the service
that remembers is not the service that computes, so it can be a different machine, a cheaper one,
or several of them, and restarting the one that holds weights costs it nothing.

What it answers:

    SWEEP_Q   q -> o, lse            over this request's history at this layer, and NOT this
                                     step's own token, which has not been appended yet
    APPEND    k, v -> how many held  posted after the join, off the critical path
    RELEASE   -> how many dropped    a slot starting a new sequence

The sweep is answered on the connection thread rather than queued for a departure: it reads the
caller's own cache, so there is no shared weight read for a departure to amortise, and queueing it
would add a departure's latency to buy batching it cannot use.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

import torch
from sglang.srt.afd.pool_attention import CachePool, KVHolder, LayerGeometry
from sglang.srt.afd.pool_server import serve

logger = logging.getLogger(__name__)


def _refuse_feed_forward(batch: torch.Tensor, layer: int) -> torch.Tensor:
    raise RuntimeError(
        f"a feed-forward frame reached a CACHE pool at layer {layer}. This service holds no "
        f"weights; the caller has its two pool addresses the wrong way round, and answering with "
        f"anything at all would be answering with something it did not compute."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="read for its config; never loaded")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--max-context", type=int, required=True)
    ap.add_argument("--device", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[cache-pool] %(message)s")

    from transformers import AutoConfig

    text = AutoConfig.from_pretrained(a.model).get_text_config()
    geometry = LayerGeometry.from_config(text)
    logger.info(
        "geometry from the config alone: %s query heads over %s key-value heads, head_dim %s, "
        "scale %.6f. No checkpoint was opened.",
        geometry.q_heads, geometry.kv_heads, geometry.head_dim, geometry.scaling,
    )
    pool = CachePool(KVHolder(a.device, a.max_context), geometry)
    ready = threading.Event()
    threading.Thread(
        target=serve,
        kwargs=dict(forward=_refuse_feed_forward, host="0.0.0.0", port=a.port, min_batch=1,
                    max_wait_s=0.005, device=a.device, ready=ready, cache=pool),
        daemon=True,
    ).start()
    if not ready.wait(timeout=30):
        raise SystemExit("  the cache pool did not bind")
    print(f"CACHE POOL READY on {a.port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
