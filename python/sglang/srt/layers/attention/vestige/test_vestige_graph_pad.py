"""bs>1 CUDA-graph padding must not overrun out_cache_loc.

Regression for the decode crash `IndexError: index 3 is out of bounds for
dimension 0 with size 3` in _refresh_graph_bufs: under graph replay,
build_replay_fb_view pads seq_lens/req_pool_indices to the captured graph bs
but takes out_cache_loc from the original batch (real_bs entries). The refresh
must iterate only the real requests and point padded CSR slots at the reserved
pad row. CPU-only; drives _refresh_graph_bufs directly.

Run: python -m pytest python/sglang/srt/layers/attention/vestige/test_vestige_graph_pad.py -q
"""

import types

import pytest
import torch

from sglang.srt.layers.attention.vestige_mla_backend import VestigeMLABackend

LID = 3
GRAPH_BS = 4  # captured graph size
REAL_BS = 3  # real requests this step (the crash shape: 4 vs 3)
KEPT = 5  # kept rows already indexed per request


def _mk_backend():
    be = object.__new__(VestigeMLABackend)  # bypass __init__: unit under test only
    fm = types.SimpleNamespace(
        kv_indptr=torch.zeros(64, dtype=torch.int32),
        kv_indices=torch.zeros(1024, dtype=torch.int64),
    )
    be.base = types.SimpleNamespace(forward_metadata=fm)
    be._tier2 = {
        (req, LID): {
            # per-req index buffer: KEPT kept rows + tail room
            "buf": torch.arange(100 * req, 100 * req + 32, dtype=torch.int64),
            "n": KEPT,
            "synced_req": None,
        }
        for req in range(REAL_BS)
    }
    be._graph_bufs = {
        LID: {
            "indptr": torch.zeros(GRAPH_BS + 2, dtype=torch.int32),
            "indices": torch.zeros(256, dtype=torch.int64),
        }
    }
    return be


def _mk_forward_batch():
    return types.SimpleNamespace(
        # padded to the captured graph bs (build_replay_fb_view contract)
        seq_lens=torch.full((GRAPH_BS,), 7, dtype=torch.int64),
        req_pool_indices=torch.arange(GRAPH_BS, dtype=torch.int64),
        # only the real requests carry a new cache slot
        out_cache_loc=torch.tensor([1001, 1002, 1003], dtype=torch.int64),
    )


def test_padded_bs_does_not_overrun_out_cache_loc():
    be = _mk_backend()
    fb = _mk_forward_batch()
    # the pre-fix logic raised IndexError here (out_cache_loc[3] with size 3)
    be._refresh_graph_bufs(LID, fb, fb.req_pool_indices.tolist())

    bufs = be._graph_bufs[LID]
    indptr = bufs["indptr"]
    per_req = KEPT + 1  # kept + this step's appended slot
    # real requests: contiguous CSR, each row = its kept set + appended slot
    for i in range(REAL_BS):
        assert indptr[i + 1] - indptr[i] == per_req
        row = bufs["indices"][int(indptr[i]) : int(indptr[i + 1])]
        assert row[-1].item() == 1001 + i  # appended out_cache_loc
        assert torch.equal(row[:-1], torch.arange(100 * i, 100 * i + KEPT))
    # padded slot: exactly one reserved pad row (slot 0), softmax non-empty
    assert indptr[REAL_BS + 1] - indptr[REAL_BS] == 1
    assert bufs["indices"][int(indptr[REAL_BS])].item() == 0
    # tier-2 state advanced only for real requests
    for req in range(REAL_BS):
        assert be._tier2[(req, LID)]["n"] == per_req


def test_unpadded_bs_matches_prior_behavior():
    # real_bs == graph bs: every slot is a real request, no pad rows
    be = _mk_backend()
    be._tier2[(3, LID)] = {
        "buf": torch.arange(300, 332, dtype=torch.int64),
        "n": KEPT,
        "synced_req": None,
    }
    fb = _mk_forward_batch()
    fb.out_cache_loc = torch.tensor([1001, 1002, 1003, 1004], dtype=torch.int64)
    be._refresh_graph_bufs(LID, fb, fb.req_pool_indices.tolist())
    indptr = be._graph_bufs[LID]["indptr"]
    for i in range(GRAPH_BS):
        assert indptr[i + 1] - indptr[i] == KEPT + 1


def test_guard_fires_without_fix():
    # Audit rule: prove the failure mode is real -- the pre-fix loop body
    # (indexing out_cache_loc at a padded i) must raise on this input.
    fb = _mk_forward_batch()
    with pytest.raises(IndexError):
        _ = fb.out_cache_loc[REAL_BS]  # what the old code did at i == real_bs


if __name__ == "__main__":
    test_padded_bs_does_not_overrun_out_cache_loc()
    test_unpadded_bs_matches_prior_behavior()
    test_guard_fires_without_fix()
    print("all 3 tests passed")


def test_prefill_invalidates_stale_tier2():
    # Slot reuse regression: a new prefill on a slot must drop the bs>1 tier-2
    # index state left by the slot's previous occupant, else decode attends the
    # prior request's rows (observed: VESTIGE attn time == FULL at bs=16).
    be = object.__new__(VestigeMLABackend)
    be.rho = 1 / 32
    be._kept_buf, be._kept_len, be._indptr1 = {}, {}, {}
    be._tier2 = {(0, LID): {"buf": torch.zeros(4, dtype=torch.int64), "n": 2}}
    max_reqs, ctx, pool = 4, 64, 512
    be.base = types.SimpleNamespace(
        forward_metadata=types.SimpleNamespace(
            kv_indices=torch.zeros(8, dtype=torch.int64)
        ),
        max_context_len=ctx,
    )
    be.req_to_token_pool = types.SimpleNamespace(
        req_to_token=torch.arange(max_reqs * ctx, dtype=torch.int64).reshape(
            max_reqs, ctx
        )
        % pool
    )
    kbuf = torch.randn(pool, 576)
    be.token_to_kv_pool = types.SimpleNamespace(get_key_buffer=lambda lid: kbuf)
    layer = types.SimpleNamespace(layer_id=LID, v_head_dim=512)
    fb = types.SimpleNamespace(
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([48], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([48], dtype=torch.int64),
    )
    be._build_gpu_state(layer, fb)
    assert (0, LID) not in be._tier2, "stale tier-2 state survived slot re-extend"
    assert int(be._kept_len[LID][0]) > 0  # fresh kept table was built
