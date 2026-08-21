"""Run a cache pool: histories, sweeps, appends. No weights, no model, no loader.

    python -m sglang.srt.afd.cache_pool_main --model <path> --port 9200 --max-context 262144

It reads the model's CONFIG for five numbers -- query heads, key-value heads, head dimension and
the softmax scale -- and never opens a checkpoint. That is the point of the split: the service
that remembers is not the service that computes, so it can be a different machine, a cheaper one,
or several of them, and restarting the one that holds weights costs it nothing.

`--model` therefore wants a directory containing config.json and nothing else is read from it.
Deploying a cache pool does not mean copying a checkpoint: the unquantised 27B is 52 GiB and its
config is 4 KiB, and the pool needs the second one. A directory holding only config.json is a
complete and correct argument here.

What the pool DOES need memory for is the histories, and that requirement is set by context times
concurrency rather than by the model: at batch 8 and 32k context this model's cache is 16 GiB in
bfloat16, which is half of a 32 GiB card. The service with no weights is not the service with no
memory -- it is the one whose memory grows with traffic instead of with the model.

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
from sglang.srt.afd.slotted_kv import SlottedKV
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
    ap.add_argument("--slots", type=int, required=True,
                    help="how many requests may hold a history at once. A slot reserves "
                         "max-context positions whether they are used or not, so this times "
                         "max-context is the pool's memory. No default: sized small the pool "
                         "refuses traffic it could serve, sized large it reserves a machine's "
                         "memory for histories that will never exist.")
    ap.add_argument("--park-timeout", type=float, required=True,
                    help="seconds a sweep may wait for the append whose positions it names. It "
                         "has no default: too short expires correct traffic, too long turns a "
                         "dead peer into a stall, and only the deployment knows its own wire.")
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
    # One buffer a layer with a slot a request, so a frame's rows are one contraction rather than
    # one pass each. The per-request holder is still what the unit tests pin the arithmetic with;
    # this is the same arithmetic in a shape a batch can be swept in.
    holder = SlottedKV(slots=a.slots, kv_heads=geometry.kv_heads, head_dim=geometry.head_dim,
                       v_head_dim=geometry.v_head_dim, max_context=a.max_context,
                       device=a.device, dtype=torch.bfloat16)
    pool = CachePool(holder, geometry)
    reserved = a.slots * geometry.kv_heads * a.max_context * (
        geometry.head_dim + geometry.v_head_dim) * 2
    softmax_layers = text.layer_types.count("full_attention") if hasattr(
        text, "layer_types") and text.layer_types else text.num_hidden_layers
    total = reserved * softmax_layers
    free_bytes = (torch.cuda.mem_get_info()[0] if a.device.startswith("cuda") else None)
    logger.info(
        "slotted cache: %s slot(s) x %s position(s) reserves %.2f GiB a layer; %s layer(s) sweep "
        "on this model, so %.1f GiB if every one is touched",
        a.slots, a.max_context, reserved / 1024 ** 3, softmax_layers, total / 1024 ** 3,
    )
    if free_bytes is not None and total > free_bytes:
        # Refused here rather than discovered at the layer that does not fit. The allocation is
        # lazy, so without this the pool serves the first few layers, reports success, and dies
        # partway through a measurement -- taking the host's connection with it and leaving a
        # BrokenPipe as the only evidence of a sizing mistake.
        raise SystemExit(
            f"  {a.slots} slot(s) x {a.max_context} position(s) needs "
            f"{total / 1024 ** 3:.1f} GiB across {softmax_layers} layer(s) and this device has "
            f"{free_bytes / 1024 ** 3:.1f} GiB free. Lower --slots or --max-context: a slot "
            f"reserves its whole context whether the request uses it or not, so the product is "
            f"what has to fit, not the traffic you expect."
        )
    ready = threading.Event()
    threading.Thread(
        target=serve,
        kwargs=dict(forward=_refuse_feed_forward, host="0.0.0.0", port=a.port, min_batch=1,
                    max_wait_s=0.005, device=a.device, ready=ready, cache=pool,
                    park_timeout_s=a.park_timeout),
        daemon=True,
    ).start()
    if not ready.wait(timeout=30):
        raise SystemExit("  the cache pool did not bind")
    print(f"CACHE POOL READY on {a.port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
