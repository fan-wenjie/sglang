"""Unit tests for the VestigeKV MLA attention backend (vestige_mla).

Each test pins a black-box behavior that once regressed in the port:
graph-replay batch padding must not overrun the unpadded out_cache_loc,
slot reuse must not leak the previous request's kept state, and the
SGLANG_VESTIGE_CHECK row invariant must both pass and fail correctly.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.vestige_mla_backend import VestigeMLABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LID = 3
GRAPH_BS = 4  # captured graph size
REAL_BS = 3  # real requests this step (the crash shape: 4 vs 3)
KEPT = 5  # kept rows already indexed per request
CAP = 64  # kept_buf row capacity in the fake


def _mk_backend():
    be = VestigeMLABackend.__new__(VestigeMLABackend)
    be.base = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            kv_indptr=torch.zeros(64, dtype=torch.int32),
            kv_indices=torch.zeros(1024, dtype=torch.int64),
        )
    )
    # prefill-built kept tables: request req holds rows 100*req .. 100*req+KEPT-1
    kept_buf = torch.zeros(GRAPH_BS, CAP, dtype=torch.int64)
    kept_len = torch.zeros(GRAPH_BS, dtype=torch.int64)
    for req in range(GRAPH_BS):
        kept_buf[req, :KEPT] = torch.arange(100 * req, 100 * req + KEPT)
        kept_len[req] = KEPT
    be._kept_buf, be._kept_len = {LID: kept_buf}, {LID: kept_len}
    be._tier2 = {}
    be._qbuf, be._fetch_buf, be._fetch_len, be._recall = {}, {}, {}, {}
    be._graph_bufs = {
        LID: {
            "indptr": torch.zeros(GRAPH_BS + 2, dtype=torch.int32),
            "indices": torch.zeros(256, dtype=torch.int64),
        }
    }
    return be


def _mk_forward_batch():
    return SimpleNamespace(
        # padded to the captured graph bs (build_replay_fb_view contract)
        seq_lens=torch.full((GRAPH_BS,), 7, dtype=torch.int64),
        req_pool_indices=torch.arange(GRAPH_BS, dtype=torch.int64),
        # only the real requests carry a new cache slot
        out_cache_loc=torch.tensor([1001, 1002, 1003], dtype=torch.int64),
    )


class TestGraphReplayPadding(CustomTestCase):
    def test_padded_bs_does_not_overrun_out_cache_loc(self):
        be = _mk_backend()
        fb = _mk_forward_batch()
        # the pre-fix logic raised IndexError here (out_cache_loc[3], size 3)
        be._refresh_graph_bufs(LID, fb, fb.req_pool_indices.tolist())

        bufs = be._graph_bufs[LID]
        indptr = bufs["indptr"]
        per_req = KEPT + 1  # kept + this step's appended slot
        for i in range(REAL_BS):
            self.assertEqual(int(indptr[i + 1] - indptr[i]), per_req)
            row = bufs["indices"][int(indptr[i]) : int(indptr[i + 1])]
            self.assertEqual(row[-1].item(), 1001 + i)
            self.assertTrue(
                torch.equal(row[:-1], torch.arange(100 * i, 100 * i + KEPT))
            )
        # padded slot: exactly one reserved pad row (slot 0)
        self.assertEqual(int(indptr[REAL_BS + 1] - indptr[REAL_BS]), 1)
        self.assertEqual(bufs["indices"][int(indptr[REAL_BS])].item(), 0)
        # kept tables advanced only for real requests
        for req in range(REAL_BS):
            self.assertEqual(int(be._kept_len[LID][req]), per_req)
        self.assertEqual(int(be._kept_len[LID][REAL_BS]), KEPT)

    def test_unpadded_bs_matches_padded_semantics(self):
        be = _mk_backend()
        fb = _mk_forward_batch()
        fb.out_cache_loc = torch.tensor([1001, 1002, 1003, 1004], dtype=torch.int64)
        be._refresh_graph_bufs(LID, fb, fb.req_pool_indices.tolist())
        indptr = be._graph_bufs[LID]["indptr"]
        for i in range(GRAPH_BS):
            self.assertEqual(int(indptr[i + 1] - indptr[i]), KEPT + 1)

    def test_pre_fix_failure_mode_is_real(self):
        # audit rule: the guarded failure must be demonstrable, not assumed
        fb = _mk_forward_batch()
        with self.assertRaises(IndexError):
            _ = fb.out_cache_loc[REAL_BS]


class TestFetchSplice(CustomTestCase):
    def test_fired_rows_splice_after_kept_segment(self):
        be = _mk_backend()
        W = 8
        be._fetch_buf = {LID: torch.zeros(GRAPH_BS, W, dtype=torch.int64)}
        be._fetch_len = {LID: torch.zeros(GRAPH_BS, dtype=torch.int64)}
        # request 1 fires two recall rows
        be._fetch_buf[LID][1, :2] = torch.tensor([901, 902])
        be._fetch_len[LID][1] = 2
        fb = _mk_forward_batch()
        be._refresh_graph_bufs(LID, fb, fb.req_pool_indices.tolist())
        bufs, indptr = be._graph_bufs[LID], be._graph_bufs[LID]["indptr"]
        per_req = KEPT + 1
        # request 0: kept segment only
        self.assertEqual(int(indptr[1] - indptr[0]), per_req)
        # request 1: kept + 2 fired, fired rows present in its slice
        self.assertEqual(int(indptr[2] - indptr[1]), per_req + 2)
        row1 = bufs["indices"][int(indptr[1]) : int(indptr[2])]
        self.assertIn(901, row1.tolist())
        self.assertIn(902, row1.tolist())
        # request 2 unaffected; padded lane still one pad row
        self.assertEqual(int(indptr[3] - indptr[2]), per_req)
        self.assertEqual(int(indptr[REAL_BS + 1] - indptr[REAL_BS]), 1)


class TestSlotReuseInvalidation(CustomTestCase):
    def test_prefill_invalidates_stale_tier2(self):
        be = VestigeMLABackend.__new__(VestigeMLABackend)
        be.rho = 1 / 32
        be._kept_buf, be._kept_len, be._indptr1 = {}, {}, {}
        be._qbuf, be._fetch_buf, be._fetch_len, be._recall = {}, {}, {}, {}
        be._q_heads, be._q_dim, be._fetch_w = 4, 576, 8
        be._tier2 = {(0, LID): {"buf": torch.zeros(4, dtype=torch.int64), "n": 2}}
        max_reqs, ctx, pool = 4, 64, 512
        be.base = SimpleNamespace(
            forward_metadata=SimpleNamespace(
                kv_indices=torch.zeros(8, dtype=torch.int64)
            ),
            max_context_len=ctx,
        )
        be.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(max_reqs * ctx, dtype=torch.int64).reshape(
                max_reqs, ctx
            )
            % pool
        )
        kbuf = torch.randn(pool, 576)
        be.token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda lid: kbuf)
        layer = SimpleNamespace(layer_id=LID, v_head_dim=512)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([48], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([48], dtype=torch.int64),
        )
        be._build_gpu_state(layer, fb)
        self.assertNotIn((0, LID), be._tier2)
        self.assertGreater(int(be._kept_len[LID][0]), 0)


class TestRowInvariantCheck(CustomTestCase):
    """SGLANG_VESTIGE_CHECK semantics: the check runs before this step's
    append, so the FULL arm requires kept_len >= seq_len - 1; the VESTIGE arm
    requires kept_len well below seq_len; a missing table or kept_len == 0
    always aborts."""

    def _mk(self, kept_lens):
        be = VestigeMLABackend.__new__(VestigeMLABackend)
        be._local_mla_lids = [LID]
        be._kept_len = {LID: torch.tensor(kept_lens, dtype=torch.int64)}
        be._fetch_len = {}
        fb = SimpleNamespace(
            out_cache_loc=torch.tensor([100, 101], dtype=torch.int64),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([2049, 2049], dtype=torch.int64),
        )
        return be, fb

    def _run(self, be, fb, full_arm):
        with patch.object(VestigeMLABackend, "_full_arm", return_value=full_arm):
            be._check_row_invariant(fb)

    def test_vestige_arm_compressed_passes(self):
        be, fb = self._mk([320, 320])
        self._run(be, fb, full_arm=False)

    def test_vestige_arm_uncompressed_raises(self):
        be, fb = self._mk([2048, 2048])
        with self.assertRaisesRegex(AssertionError, "compression not applied"):
            self._run(be, fb, full_arm=False)

    def test_full_arm_pending_append_passes(self):
        be, fb = self._mk([2048, 2048])
        self._run(be, fb, full_arm=True)

    def test_full_arm_short_raises(self):
        be, fb = self._mk([320, 320])
        with self.assertRaisesRegex(AssertionError, "FULL arm"):
            self._run(be, fb, full_arm=True)

    def test_missing_table_raises(self):
        be, fb = self._mk([2048, 2048])
        be._kept_len = {}
        with self.assertRaisesRegex(AssertionError, "never built"):
            self._run(be, fb, full_arm=False)


if __name__ == "__main__":
    unittest.main()
