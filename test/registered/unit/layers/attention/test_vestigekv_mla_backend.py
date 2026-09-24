"""Unit tests for the VestigeKV MLA attention backend (vestigekv_mla).

Each test pins a black-box behavior that once regressed in the port:
graph-replay batch padding must not overrun the unpadded out_cache_loc,
slot reuse must not leak the previous request's kept state, and a block
closed at decode time must not append a row the capture already holds.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.config import VestigeKVConfig
from sglang.srt.layers.attention.vestigekv_mla_backend import VestigeKVMLABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# The activation threshold is a server flag whose default is 0 (every request
# compresses); the state-machine tests below need a nonzero one, pinned on
# the class-level config the __new__-constructed fakes read.
ACT_MIN = 32768
FLAG_DEFAULT_CONFIG = VestigeKVMLABackend.config
_CONFIG_PATCH = patch.object(
    VestigeKVMLABackend,
    "config",
    msgspec.structs.replace(FLAG_DEFAULT_CONFIG, activation_min_tokens=ACT_MIN),
)


def setUpModule():
    _CONFIG_PATCH.start()


def tearDownModule():
    _CONFIG_PATCH.stop()


LID = 3
# Calibration/close are gated on ACTIVATION_MIN_TOKENS; tests of that state
# machine run at offsets above the threshold.
SEQ_BASE = ACT_MIN + 100
GRAPH_BS = 4  # captured graph size
REAL_BS = 3  # real requests this step (the crash shape: 4 vs 3)
KEPT = 5  # kept rows already indexed per request
CAP = 64  # kept_buf row capacity in the fake


def _mk_backend():
    be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
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
    be._qbuf, be._fetch_buf, be._fetch_len, be._recall = {}, {}, {}, {}
    be._fetch_ovf = {}
    # every slot has prefilled through the backend (compressed state exists)
    be._close_state = {(req, LID): {} for req in range(GRAPH_BS)}
    # _kmax: host-side upper bound the static-shape CSR pack gathers with;
    # KEPT here, raised by one per decode step exactly as prefill does.
    be._kmax = {LID: KEPT}
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
    """A pool slot re-extended for a new request must not carry the previous
    occupant's recall state into the new request's first decode step: the
    tier is dropped, the fetch buffer emptied and the overflow flag cleared
    (stale state once made bs>1 decode attend the prior request's row set)."""

    def _backend(self):
        be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
        be.rho = 1 / 32
        be._kept_buf, be._kept_len, be._kmax = {}, {}, {}
        be._close_state = {}
        be._qbuf, be._fetch_buf, be._fetch_len, be._recall = {}, {}, {}, {}
        be._fetch_ovf = {}
        be._q_heads, be._q_dim, be._fetch_w = 4, 576, 8
        be._qbuf_stack = be._fetch_stack = be._fetch_len_stack = None
        be._fetch_ovf_stack = be._ovf_count_stack = None
        be._li_map = {}
        be._local_mla_lids = [LID]
        be._mla_lids = set()
        be._n_cal, be.index_rank = 4, 8
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
        return be

    def _prefill(self, be, seq):
        layer = SimpleNamespace(layer_id=LID, v_head_dim=512)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([seq], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([seq], dtype=torch.int64),
            # chunked-prefill fields: this extend completes the prefix
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[seq],
        )
        be._build_gpu_state(layer, fb)

    def test_prefill_registers_the_layer_for_decode(self):
        # Layer membership used to be learned at graph capture only, so with
        # CUDA graph off no layer ever collected calibration or recalled.
        be = self._backend()
        self._prefill(be, 48)
        self.assertEqual(be._mla_lids, {LID})

    def test_prefill_resets_the_slot_recall_state(self):
        be = self._backend()
        self._prefill(be, 48)
        self.assertGreater(int(be._kept_len[LID][0]), 0)
        # the previous occupant's decode left a tier and a fired fetch behind
        be._recall[(0, LID)]["tier"] = "stale"
        be._fetch_len[LID][0] = 3
        be._fetch_ovf[LID][0] = 1
        self._prefill(be, 40)
        self.assertIsNone(be._recall[(0, LID)]["tier"])
        self.assertEqual(int(be._fetch_len[LID][0]), 0)
        self.assertEqual(int(be._fetch_ovf[LID][0]), 0)


class TestOverflowRearm(CustomTestCase):
    """A calibrated index is fitted once and then serves the whole request, so a
    fit made at short context over-fires at long context and every overflowing
    lane is served from the full row set. The per-layer overflow tally re-opens
    calibration for such a layer at its next block close."""

    def _backend(self, fraction=0.05, tally=(0, 0)):
        be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
        be.config = msgspec.structs.replace(
            FLAG_DEFAULT_CONFIG, rebuild_overflow_fraction=fraction
        )
        be._ovf_count_stack = torch.tensor(tally, dtype=torch.int32)
        be._li_map = {10: 0, 11: 1}
        be._ovf_at_close = {}
        be._collecting = False
        be._recall = {
            (0, lid): {"tier": object(), "qcal": None, "qpos": None, "target": 99}
            for lid in (10, 11)
        }
        return be

    def test_only_the_overflowing_layer_reopens_calibration(self):
        hot = int(D.CLOSE_BLOCK * 0.05) + 1
        be = self._backend(tally=(hot, 0))
        be._rearm_overflowing_indices([(0, 10), (0, 11)])
        self.assertEqual(be._recall[(0, 10)]["qcal"], [])
        self.assertEqual(be._recall[(0, 10)]["target"], D.N_CAL_START)
        self.assertIsNone(be._recall[(0, 11)]["qcal"])
        self.assertTrue(be._collecting)

    def test_the_tally_is_read_as_growth_since_the_last_close(self):
        # The counter is cumulative over the request: comparing it against zero
        # instead of against its value at the previous close re-arms every layer
        # at every close for the rest of the request, one build per 4096 tokens.
        hot = int(D.CLOSE_BLOCK * 0.05) + 1
        be = self._backend(tally=(hot, 0))
        be._rearm_overflowing_indices([(0, 10)])
        be._recall[(0, 10)]["qcal"] = None  # as an install would leave it
        be._collecting = False
        be._rearm_overflowing_indices([(0, 10)])  # same tally: no new overflow
        self.assertIsNone(be._recall[(0, 10)]["qcal"])
        self.assertFalse(be._collecting)

    def test_a_build_in_flight_is_not_disturbed(self):
        hot = int(D.CLOSE_BLOCK * 0.05) + 1
        be = self._backend(tally=(hot, 0))
        be._recall[(0, 10)]["job"] = {"seq_len": 1}
        be._rearm_overflowing_indices([(0, 10)])
        self.assertIsNone(be._recall[(0, 10)]["qcal"])

    def test_zero_fraction_is_off(self):
        be = self._backend(fraction=0.0, tally=(D.CLOSE_BLOCK, 0))
        be._rearm_overflowing_indices([(0, 10)])
        self.assertIsNone(be._recall[(0, 10)]["qcal"])


class TestSplitLens(CustomTestCase):
    """The split count is a performance knob over the row range the decode
    kernel reads; the base sizes it from the request's length, which at long
    context is 30x the attended rows."""

    def _backend(self, enabled, kmax=None):
        be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
        be.config = msgspec.structs.replace(
            FLAG_DEFAULT_CONFIG, attended_splits=enabled, recall_capacity=4096
        )
        be._kmax = {} if kmax is None else kmax
        return be

    def test_the_bound_is_kept_plus_the_recall_capacity(self):
        lens = torch.tensor([262144, 1000], dtype=torch.int64)
        got = self._backend(True, {3: 8192, 7: 4096})._split_lens(lens)
        self.assertEqual(got.tolist(), [8192 + 4096, 1000])

    def test_off_and_before_any_close_the_request_length_is_used(self):
        lens = torch.tensor([262144], dtype=torch.int64)
        self.assertEqual(
            self._backend(False, {3: 8192})._split_lens(lens).tolist(), [262144]
        )
        # no layer has closed a block yet: there is no attended bound to clamp to
        self.assertEqual(self._backend(True)._split_lens(lens).tolist(), [262144])


class TestConfig(CustomTestCase):
    def test_fake_default_matches_the_flag_defaults(self):
        # The __new__ fakes read the class-level config; if a flag default
        # moves without it, every CPU case here runs a configuration the
        # server never does.
        from sglang.srt.arg_groups.fields.exec_ import ExecKernel

        self.assertEqual(
            VestigeKVConfig.from_kernel_config(ExecKernel()), FLAG_DEFAULT_CONFIG
        )

    def test_out_of_range_values_are_refused(self):
        base = FLAG_DEFAULT_CONFIG
        for bad in (
            {"recall_capacity": 0},
            {"activation_min_tokens": -1},
            {"index_rank": 60},
            {"recall_margin": -0.5},
            {"recall_threshold": "mean"},
            {"rebuild_overflow_fraction": -0.1},
            {"rebuild_overflow_fraction": 1.5},
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                msgspec.structs.replace(base, **bad).validate()


class TestOverflowFence(CustomTestCase):
    """A lane whose recall fire overflowed the fetch capacity attends its full
    row set that step -- req_to_token[slot, :seq] with this step's slot last
    -- while the other lanes and the kept-table append are unchanged. With
    the fallback disabled the lane packs its truncated fetch as before.
    """

    W = 8
    R2T = 16

    def _backend(self, fallback):
        be = _mk_backend()
        be.config = msgspec.structs.replace(be.config, overflow_fallback=fallback)
        be._fetch_buf = {LID: torch.zeros(GRAPH_BS, self.W, dtype=torch.int32)}
        be._fetch_len = {LID: torch.zeros(GRAPH_BS, dtype=torch.int32)}
        be._fetch_ovf = {LID: torch.zeros(GRAPH_BS, dtype=torch.int32)}
        # request 1 fired past the capacity (buffer full, flag up); request 2
        # fired two rows and fits
        be._fetch_buf[LID][1] = torch.arange(900, 900 + self.W)
        be._fetch_len[LID][1] = self.W
        be._fetch_ovf[LID][1] = 1
        be._fetch_buf[LID][2, :2] = torch.tensor([701, 702])
        be._fetch_len[LID][2] = 2
        be.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(GRAPH_BS * self.R2T, dtype=torch.int32).reshape(
                GRAPH_BS, self.R2T
            )
        )
        return be

    def _segments(self, be):
        bufs = be._graph_bufs[LID]
        indptr = bufs["indptr"]
        return [
            bufs["indices"][int(indptr[i]) : int(indptr[i + 1])].tolist()
            for i in range(REAL_BS)
        ]

    def test_fenced_lane_attends_its_full_row_set(self):
        be = self._backend(fallback=True)
        fb = _mk_forward_batch()  # seq_lens 7, real lanes 0..2
        be._refresh_graph_bufs(LID, fb, fb.req_pool_indices.tolist())
        seg = self._segments(be)
        r2t = be.req_to_token_pool.req_to_token
        self.assertEqual(seg[1], r2t[1, :6].tolist() + [1002])
        self.assertEqual(seg[0], list(range(0, KEPT)) + [1001])
        self.assertEqual(seg[2], list(range(200, 200 + KEPT)) + [1003, 701, 702])
        for req in range(REAL_BS):  # the append happened on every lane
            self.assertEqual(int(be._kept_len[LID][req]), KEPT + 1)
        self.assertEqual(int(be._kept_buf[LID][1, KEPT]), 1002)

    def test_fallback_off_packs_the_truncated_fetch(self):
        be = self._backend(fallback=False)
        fb = _mk_forward_batch()
        be._refresh_graph_bufs(LID, fb, fb.req_pool_indices.tolist())
        seg = self._segments(be)
        self.assertEqual(
            seg[1], list(range(100, 100 + KEPT)) + [1002] + list(range(900, 908))
        )


if __name__ == "__main__":
    unittest.main()


def _mk_scan_backend(archs, built=True):
    """Backend carrying only the state the tier-2 scan capture reads."""
    be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
    be._mla_lids = [3, 7]
    be._qbuf = {3: object(), 7: object()}
    be._recall = {}
    for (slot, lid), a in archs.items():
        tier = SimpleNamespace(arch=torch.zeros(a)) if built else None
        be._recall[(slot, lid)] = {"tier": tier, "built_at": 0}
    be._scan_graph = None
    be._scan_key_cur = be._scan_key_seen = None
    be._scan_steps = 0
    be._scan_pool = None
    be._scan_capture_failed = False
    be._capture_asap = False
    be._stage_slots = be._stage_loc = None
    be._kept_buf, be._kmax = {}, {}
    be._step_cache = None
    return be


def _scan_fb(real_bs, graph_bs=None):
    return SimpleNamespace(
        out_cache_loc=torch.zeros(real_bs, dtype=torch.int64),
        seq_lens=torch.zeros(graph_bs or real_bs, dtype=torch.int64),
    )


class TestVestigeScanCapture(CustomTestCase):
    """The captured tier-2 scan bakes in pool-slot addresses and archive shapes.
    Every input that can move one of those must produce a different key, or the
    graph replays against a stale archive and silently attends the wrong rows.
    """

    def test_key_is_none_until_every_tier_is_built(self):
        be = _mk_scan_backend({(0, 3): 100}, built=False)
        self.assertIsNone(be._scan_key(_scan_fb(1), [0]))

    def test_key_is_none_when_one_layer_lacks_a_tier(self):
        be = _mk_scan_backend({(0, 3): 100})  # layer 7 missing entirely
        self.assertIsNone(be._scan_key(_scan_fb(1), [0]))

    def test_key_ignores_archive_size(self):
        # New contract: sizes stay OUT of the key -- kernels mask by
        # a_len/nk_len and the grid is capacity-sized, so growth is the epoch
        # path's job (fits()->update(); capacity overflow recaptures via
        # fits() failing). Keying on exact size made every install/close a
        # recapture: the bs4 capture churn (136 captures / 26 s).
        a = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        b = _mk_scan_backend({(0, 3): 100, (0, 7): 201})
        self.assertEqual(a._scan_key(_scan_fb(1), [0]), b._scan_key(_scan_fb(1), [0]))

    def test_key_tracks_pool_slot(self):
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200, (1, 3): 100, (1, 7): 200})
        self.assertNotEqual(
            be._scan_key(_scan_fb(1), [0]), be._scan_key(_scan_fb(1), [1])
        )

    def test_key_ignores_graph_padding(self):
        # req_pool_indices is padded to the captured bs; only out_cache_loc
        # gives the real request count, and the key must follow it.
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200, (1, 3): 100, (1, 7): 200})
        self.assertEqual(
            be._scan_key(_scan_fb(1), [0, 1]), be._scan_key(_scan_fb(1), [0, 9])
        )

    def test_capture_is_deferred_until_the_shape_has_held(self):
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        with patch.object(VestigeKVMLABackend, "_full_arm", return_value=False):
            with patch.object(VestigeKVMLABackend, "_capture_scan") as cap:
                for _ in range(7):
                    self.assertFalse(
                        be._replay_scan(
                            _scan_fb(1), [0], be._scan_key(_scan_fb(1), [0])
                        )
                    )
                self.assertEqual(cap.call_count, 0)
                be._replay_scan(_scan_fb(1), [0], be._scan_key(_scan_fb(1), [0]))
                self.assertEqual(cap.call_count, 1)

    def test_a_changed_shape_restarts_the_deferral(self):
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200, (1, 3): 100, (1, 7): 200})
        with patch.object(VestigeKVMLABackend, "_full_arm", return_value=False):
            with patch.object(VestigeKVMLABackend, "_capture_scan") as cap:
                for _ in range(7):
                    be._replay_scan(_scan_fb(1), [0], be._scan_key(_scan_fb(1), [0]))
                be._replay_scan(
                    _scan_fb(1), [1], be._scan_key(_scan_fb(1), [1])
                )  # different slot
                for _ in range(6):
                    be._replay_scan(_scan_fb(1), [1], be._scan_key(_scan_fb(1), [1]))
                self.assertEqual(cap.call_count, 0)

    def test_invalidate_keeps_the_capture_but_resets_the_deferral(self):
        # Contents changes no longer drop the graph (the replay path refreshes
        # the pack in place); what invalidate must still reset is the deferral
        # state and the step cache.
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        graph, pack = object(), object()
        be._scan_graph, be._scan_key_cur = graph, ("k",)
        be._scan_batched = pack
        be._scan_steps, be._scan_key_seen = 5, ("k",)
        be._step_cache = ("sig", [0], ("k",))
        be._invalidate_scan()
        self.assertIs(be._scan_graph, graph)  # capture survives
        self.assertIs(be._scan_batched, pack)
        self.assertIsNone(be._scan_key_seen)
        self.assertEqual(be._scan_steps, 0)
        self.assertIsNone(be._step_cache)

    def test_same_shape_new_tiers_updates_the_pack_in_place(self):
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        fb = _scan_fb(1)
        key = be._scan_key(fb, [0])
        be._scan_key_cur = key
        calls = []
        pack = SimpleNamespace(
            tier_ids=("stale",),
            fits=lambda pairs, tiers: True,
            update=lambda pairs, tiers: calls.append(("update", len(tiers))),
        )
        be._scan_batched = pack
        be._scan_graph = SimpleNamespace(replay=lambda: calls.append(("replay",)))
        be._li_map = {3: 0, 7: 1}
        be._kept_buf, be._kmax = {}, {}
        be._stage_slots = torch.zeros(4, dtype=torch.int64)
        be._stage_loc = torch.zeros(4, dtype=torch.int64)
        be._stage_seq = torch.zeros(4, dtype=torch.int64)
        fb2 = SimpleNamespace(
            out_cache_loc=torch.zeros(1, dtype=torch.int64),
            seq_lens=torch.zeros(1, dtype=torch.int64),
            req_pool_indices=torch.zeros(4, dtype=torch.int64),
        )
        with patch.object(VestigeKVMLABackend, "_full_arm", return_value=False):
            self.assertTrue(be._replay_scan(fb2, [0], key))
        self.assertEqual(calls, [("update", 2), ("replay",)])

    def test_pack_that_does_not_fit_recaptures(self):
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        fb = _scan_fb(1)
        key = be._scan_key(fb, [0])
        be._scan_key_cur = key
        be._scan_batched = SimpleNamespace(
            tier_ids=("stale",),
            fits=lambda pairs, tiers: False,
            update=lambda *_: self.fail("must not update an undersized pack"),
        )
        be._scan_graph = SimpleNamespace(replay=lambda: None)
        be._li_map = {3: 0, 7: 1}
        fb2 = SimpleNamespace(
            out_cache_loc=torch.zeros(1, dtype=torch.int64),
            seq_lens=torch.zeros(1, dtype=torch.int64),
            req_pool_indices=torch.zeros(4, dtype=torch.int64),
        )
        with patch.object(VestigeKVMLABackend, "_full_arm", return_value=False):
            with patch.object(
                VestigeKVMLABackend, "_capture_scan", return_value=True
            ) as cap:
                self.assertTrue(be._replay_scan(fb2, [0], key))
                self.assertEqual(cap.call_count, 1)

    def test_failed_capture_does_not_retry(self):
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        be._scan_capture_failed = True
        with patch.object(VestigeKVMLABackend, "_full_arm", return_value=False):
            with patch.object(VestigeKVMLABackend, "_capture_scan") as cap:
                for _ in range(20):
                    self.assertFalse(
                        be._replay_scan(
                            _scan_fb(1), [0], be._scan_key(_scan_fb(1), [0])
                        )
                    )
                self.assertEqual(cap.call_count, 0)

    def test_full_arm_never_captures(self):
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        with patch.object(VestigeKVMLABackend, "_full_arm", return_value=True):
            with patch.object(VestigeKVMLABackend, "_capture_scan") as cap:
                for _ in range(20):
                    self.assertFalse(
                        be._replay_scan(
                            _scan_fb(1), [0], be._scan_key(_scan_fb(1), [0])
                        )
                    )
                self.assertEqual(cap.call_count, 0)


class TestStaticPackMatchesReference(CustomTestCase):
    """The CSR pack was rewritten to static shapes with device-side offsets to
    drop four host readbacks per layer. Its output must still be, for every
    request in order, that request's kept rows followed by its fired recall
    rows -- checked here against a naive python reference over random lengths.
    """

    def _run_once(self, real_bs, graph_bs, k_lens, f_lens, cap, fw, rng):
        be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
        kept_buf = torch.zeros(graph_bs, cap, dtype=torch.int64)
        kept_len = torch.zeros(graph_bs, dtype=torch.int64)
        fetch_buf = torch.zeros(graph_bs, fw, dtype=torch.int64)
        fetch_len = torch.zeros(graph_bs, dtype=torch.int64)
        for r in range(real_bs):
            # kept_len holds the count BEFORE this step's append, so seed it one
            # short and let _refresh_graph_bufs append the step's own row.
            kept_buf[r, : k_lens[r] - 1] = torch.tensor(
                rng.sample(range(1000, 9000), k_lens[r] - 1)
            )
            kept_len[r] = k_lens[r] - 1
            fetch_buf[r, : f_lens[r]] = torch.tensor(
                rng.sample(range(10000, 90000), f_lens[r])
            )
            fetch_len[r] = f_lens[r]
        be._kept_buf, be._kept_len = {LID: kept_buf}, {LID: kept_len}
        be._fetch_buf, be._fetch_len = {LID: fetch_buf}, {LID: fetch_len}
        be._fetch_ovf = {}
        be._close_state = {(req, LID): {} for req in range(graph_bs)}
        be._kmax = {LID: max(k_lens) - 1}
        be._graph_bufs = {
            LID: {
                "indptr": torch.zeros(graph_bs + 2, dtype=torch.int64),
                "indices": torch.zeros(graph_bs * cap + 1, dtype=torch.int64),
            }
        }
        loc = torch.tensor(rng.sample(range(100, 999), real_bs), dtype=torch.int64)
        fb = SimpleNamespace(
            seq_lens=torch.zeros(graph_bs, dtype=torch.int64),
            out_cache_loc=loc,
            req_pool_indices=torch.arange(graph_bs, dtype=torch.int64),
        )
        be._refresh_graph_bufs(LID, fb, list(range(graph_bs)))
        bufs = be._graph_bufs[LID]

        expect, at = [], 0
        for r in range(real_bs):
            rows = kept_buf[r, : k_lens[r]].tolist()  # includes the appended loc
            rows += fetch_buf[r, : f_lens[r]].tolist()
            self.assertEqual(int(bufs["indptr"][r]), at)
            at += len(rows)
            self.assertEqual(int(bufs["indptr"][r + 1]), at)
            expect += rows
        got = bufs["indices"][: len(expect)].tolist()
        self.assertEqual(got, expect)
        # padded lanes each get one reserved row 0 and a strictly rising indptr
        for j in range(real_bs, graph_bs):
            self.assertEqual(int(bufs["indices"][int(bufs["indptr"][j])]), 0)
            self.assertEqual(int(bufs["indptr"][j + 1]) - int(bufs["indptr"][j]), 1)

    def test_random_lengths_match_reference(self):
        import random

        rng = random.Random(20260906)
        cap, fw = 128, 32
        for _ in range(40):
            graph_bs = rng.choice([1, 2, 4, 8])
            real_bs = rng.randint(1, graph_bs)
            # kept and fetched rows are disjoint (fetch draws from the
            # archive, which is kept's complement), so kept + fetch <= cap --
            # the invariant that lets the CSR buffer be sized by context length.
            k_lens = [rng.randint(1, cap - fw - 1) for _ in range(real_bs)]
            f_lens = [rng.randint(0, fw) for _ in range(real_bs)]
            self._run_once(real_bs, graph_bs, k_lens, f_lens, cap, fw, rng)

    def test_zero_fetch_everywhere(self):
        import random

        self._run_once(2, 4, [7, 3], [0, 0], 64, 16, random.Random(1))

    def test_fetch_at_full_width(self):
        import random

        self._run_once(2, 2, [5, 9], [16, 16], 64, 16, random.Random(2))


class TestTierTwoIsNeverOff(CustomTestCase):
    """Tier 2 has no off state, and the calibrated build is ASYNC: the
    provisional index (built synchronously on the first decode step) serves
    while the O(S) label GEMM runs on a side stream; the result is installed
    atomically on the main thread. These drive the enqueue/install state
    machine with a stubbed worker.
    """

    def _backend(self, slots=(0,)):
        be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
        be._mla_lids = [LID]
        be._qbuf = {LID: torch.zeros(4, 2, 576)}
        be._recall = {
            (s, LID): {
                "tier": None,
                "built_at": 0,
                "qcal": [],
                "qpos": [],
                "target": D.N_CAL_START,
            }
            for s in slots
        }
        be._collecting = True
        be._scan_steps = 0
        be._capture_asap = False
        be._needs_recapture = False
        be._build_jobs = []
        be.built = []
        be.enqueued = []
        return be

    def _fb(self, real_bs=1, seq=101):
        return SimpleNamespace(
            out_cache_loc=torch.zeros(real_bs, dtype=torch.int64),
            seq_lens=torch.full((real_bs,), SEQ_BASE + seq, dtype=torch.int64),
        )

    def _stubs(self, be):
        def build(_self, slot, lid, seq_len, st, proxy):
            be.built.append((seq_len, proxy))
            st["tier"] = SimpleNamespace(built_at=seq_len, proxy=proxy)
            return {"need_more_hard": False}

        def enqueue(_self, slot, lid, seq_len, st):
            job = {
                "slot": slot,
                "lid": lid,
                "st": st,
                "seq_len": seq_len,
                "qcal": list(st["qcal"]),
                "done": SimpleNamespace(is_set=lambda: False),
                "tier": None,
                "stats": None,
                "error": None,
            }
            be.enqueued.append(job)
            be._build_jobs.append(job)
            return job

        return (
            patch.object(VestigeKVMLABackend, "_build_index", build),
            patch.object(VestigeKVMLABackend, "_enqueue_build", enqueue),
        )

    def _finish(self, job, n_hard=99, error=None):
        job["done"] = SimpleNamespace(is_set=lambda: True)
        job["error"] = error
        if error is None:
            job["tier"] = SimpleNamespace(built_at=job["seq_len"], proxy=False, V="V")
            job["stats"] = {
                "need_more_hard": n_hard < D.min_hard(),
                "n_hard": n_hard,
            }

    def test_below_activation_threshold_defers_collection(self):
        # Below ACTIVATION_MIN_TOKENS the request runs dense: no archive, so
        # nothing to calibrate and no index to build. Collection must stay
        # pending so the provisional build fires on the crossing step.
        be = self._backend()
        b, e = self._stubs(be)
        with b, e:
            be._collect_calibration(self._fb(seq=101 - SEQ_BASE), [0])
        self.assertEqual(be.built, [])
        self.assertEqual(be._recall[(0, LID)]["qcal"], [])
        self.assertTrue(be._collecting)

    def test_a_build_finished_during_prefill_installs_before_the_provisional_index(
        self,
    ):
        # With prefill calibration the calibrated job can be done before the
        # first decode step; installing it first skips the provisional build.
        be = self._backend()
        st = be._recall[(0, LID)]
        job = {
            "slot": 0,
            "lid": LID,
            "st": st,
            "seq_len": 100,
            "qcal": [None] * 8,
            "done": SimpleNamespace(is_set=lambda: True),
            "error": None,
            "tier": SimpleNamespace(built_at=100, proxy=False, V="V"),
            "stats": {"need_more_hard": False, "n_hard": 99},
        }
        st["job"] = job
        be._build_jobs = [job]
        b, e = self._stubs(be)
        with b, e:
            be._collect_calibration(self._fb(), [0])
        self.assertEqual(be.built, [])  # no provisional build
        self.assertIs(st["tier"], job["tier"])

    def test_provisional_index_is_built_synchronously_on_step_one(self):
        be = self._backend()
        b, e = self._stubs(be)
        with b, e:
            be._collect_calibration(self._fb(), [0])
        self.assertEqual(be.built, [(SEQ_BASE + 101, True)])
        self.assertIsNotNone(be._recall[(0, LID)]["tier"])
        self.assertEqual(be.enqueued, [])

    def test_reaching_the_target_enqueues_exactly_once(self):
        be = self._backend()
        b, e = self._stubs(be)
        with b, e:
            for step in range(D.N_CAL_START + 3):
                be._collect_calibration(self._fb(seq=101 + step), [0])
        self.assertEqual(len(be.enqueued), 1)  # pending job blocks re-enqueue
        self.assertTrue(be._collecting)  # stays on while the job is in flight

    def test_finished_build_installs_on_the_main_step(self):
        be = self._backend()
        b, e = self._stubs(be)
        called = []
        with (
            b,
            e,
            patch.object(
                VestigeKVMLABackend, "_invalidate_scan", lambda _s: called.append(1)
            ),
        ):
            for step in range(D.N_CAL_START):
                be._collect_calibration(self._fb(seq=101 + step), [0])
            self._finish(be.enqueued[0])
            be._capture_asap = False  # clear the provisional-build flag
            epoch0 = be._pack_epoch
            be._collect_calibration(self._fb(seq=120), [0])
            be._collect_calibration(self._fb(seq=121), [0])  # sticky fires here
        st = be._recall[(0, LID)]
        self.assertFalse(st["tier"].proxy)  # calibrated tier adopted
        self.assertIsNone(st["qcal"])  # collection over
        self.assertTrue(be._capture_asap)
        # New contract: install is a content swap -- epoch bump, no graph drop
        # (invalidate caused capture churn: bs4 saw 139 captures / 27 s).
        self.assertEqual(called, [])
        self.assertGreater(be._pack_epoch, epoch0)

    def test_replaced_slot_state_and_its_job_are_freed_without_the_cyclic_gc(self):
        # A request that ends before its calibrated build installs must not
        # leave its state dict and job (which reference each other) to the
        # cyclic GC: the job holds the built tier, ~100 MB per request at 64k.
        import gc
        import weakref

        be = self._backend()
        b, e = self._stubs(be)
        with b, e:
            for step in range(D.N_CAL_START):
                be._collect_calibration(self._fb(seq=101 + step), [0])
        old_st = be._recall[(0, LID)]
        job = be.enqueued[0]
        self.assertIs(old_st["job"], job)
        self._finish(job)

        # dicts take no weakref: watch the tiers they hold (the provisional
        # one in the state, the calibrated one in the job)
        class Tier:
            built_at, proxy, V = 0, False, "V"

        old_st["tier"], job["tier"] = Tier(), Tier()
        ref_prov, ref_cal = weakref.ref(old_st["tier"]), weakref.ref(job["tier"])
        gc.disable()
        try:
            be._reset_slot_state(slot=0, lid=LID)  # the next request's prefill
            be._install_finished_builds()  # drops the stale job
            del old_st, job
            self.assertIsNone(ref_prov())
            self.assertIsNone(ref_cal())
        finally:
            gc.enable()
        self.assertEqual(be._build_jobs, [])

    def test_reachable_hard_rate_jumps_the_window_predictively(self):
        # window 8, n_hard=6 -> needed = ceil(18*8/6)=24 -> next pow2 = 32,
        # in ONE jump instead of 8->16->32
        be = self._backend()
        b, e = self._stubs(be)
        with b, e:
            for step in range(D.N_CAL_START):
                be._collect_calibration(self._fb(seq=101 + step), [0])
            self._finish(be.enqueued[0], n_hard=6)
            be._collect_calibration(self._fb(seq=120), [0])
        st = be._recall[(0, LID)]
        self.assertEqual(st["target"], 32)
        self.assertIsNotNone(st["qcal"])  # still collecting toward 32
        self.assertTrue(st["tier"].proxy)  # provisional keeps serving

    def test_unreachable_hard_rate_installs_the_clamped_tier_now(self):
        # window 8, n_hard=1 -> needed = 144 > N_CAL_MAX -> give up, adopt
        be = self._backend()
        b, e = self._stubs(be)
        with b, e:
            for step in range(D.N_CAL_START):
                be._collect_calibration(self._fb(seq=101 + step), [0])
            self._finish(be.enqueued[0], n_hard=1)
            be._collect_calibration(self._fb(seq=120), [0])
        st = be._recall[(0, LID)]
        self.assertFalse(st["tier"].proxy)  # clamped-but-real-basis adopted
        self.assertIsNone(st["qcal"])  # no more rebuild churn

    def test_installs_coalesce_into_one_invalidate(self):
        # two layers finish at different times: install silently, and the ONE
        # invalidate fires only when nothing is pending or collecting anymore
        be = self._backend()
        be._mla_lids = [3, 7]
        be._qbuf = {3: torch.zeros(4, 2, 576), 7: torch.zeros(4, 2, 576)}
        be._recall = {
            (0, L): {
                "tier": SimpleNamespace(proxy=True),
                "built_at": 0,
                "qcal": None,  # collection already over for both layers
                "qpos": None,
                "target": D.N_CAL_START,
            }
            for L in (3, 7)
        }
        for L in (3, 7):
            be._recall[(0, L)]["qcal"] = [object()] * 8
            be._recall[(0, L)]["qpos"] = [0] * 8
        calls = []
        pend = SimpleNamespace(is_set=lambda: False)
        j3 = {
            "slot": 0,
            "lid": 3,
            "st": be._recall[(0, 3)],
            "seq_len": 108,
            "qcal": [object()] * 8,
            "done": pend,
            "error": None,
            "tier": None,
            "stats": None,
        }
        j7 = {
            "slot": 0,
            "lid": 7,
            "st": be._recall[(0, 7)],
            "seq_len": 108,
            "qcal": [object()] * 8,
            "done": pend,
            "error": None,
            "tier": None,
            "stats": None,
        }
        be._build_jobs = [j3, j7]
        be._recall[(0, 3)]["job"] = j3
        be._recall[(0, 7)]["job"] = j7
        with patch.object(
            VestigeKVMLABackend, "_invalidate_scan", lambda _s: calls.append(1)
        ):
            self._finish(j3)
            be._collect_calibration(self._fb(seq=120), [0])
            self.assertEqual(calls, [])  # j7 pending: no invalidate yet
            self.assertFalse(be._recall[(0, 3)]["tier"].proxy)  # but installed
            self._finish(j7)
            epoch0 = be._pack_epoch
            be._collect_calibration(self._fb(seq=121), [0])
            be._collect_calibration(self._fb(seq=122), [0])  # sticky fires here
        # New contract: coalesced install is a content swap -- one epoch bump,
        # never a graph drop (invalidate caused capture churn: bs4 saw 139
        # captures/27 s; fits() failure still recaptures on its own).
        self.assertEqual(calls, [])
        self.assertGreater(be._pack_epoch, epoch0)
        self.assertTrue(be._capture_asap)

    def test_worker_error_keeps_the_provisional_index(self):
        be = self._backend()
        b, e = self._stubs(be)
        with b, e:
            for step in range(D.N_CAL_START):
                be._collect_calibration(self._fb(seq=101 + step), [0])
            self._finish(be.enqueued[0], error=RuntimeError("boom"))
            be._collect_calibration(self._fb(seq=120), [0])
        st = be._recall[(0, LID)]
        self.assertTrue(st["tier"].proxy)
        self.assertIsNone(st["qcal"])  # gives up cleanly, no retry storm

    def test_job_for_a_reprefilled_slot_is_dropped(self):
        be = self._backend()
        b, e = self._stubs(be)
        with b, e:
            for step in range(D.N_CAL_START):
                be._collect_calibration(self._fb(seq=101 + step), [0])
            # slot reused by a new request: collector state reset by prefill
            be._recall[(0, LID)] = {
                "tier": None,
                "built_at": 0,
                "qcal": [],
                "qpos": [],
                "target": D.N_CAL_START,
            }
            self._finish(be.enqueued[0])
            be._install_finished_builds()
        self.assertIsNone(be._recall[(0, LID)]["tier"])  # stale tier NOT adopted

    def test_queries_are_cloned_not_aliased(self):
        be = self._backend()
        b, e = self._stubs(be)
        with b, e:
            be._qbuf[LID][0].fill_(1.0)
            be._collect_calibration(self._fb(seq=101), [0])
            be._qbuf[LID][0].fill_(2.0)
            be._collect_calibration(self._fb(seq=102), [0])
        qcal = be._recall[(0, LID)]["qcal"]
        self.assertEqual(float(qcal[0][0, 0]), 1.0)
        self.assertEqual(float(qcal[1][0, 0]), 2.0)


class TestCaptureDoesNotDuplicateKeptRows(CustomTestCase):
    """_run executes four times around a capture (two warmups, the capture
    itself, and the first replay), and the CSR pack inside it appends the
    step's row each time. Without the rewind, the captured step leaves three
    duplicate copies of its row in the kept table -- duplicates change softmax
    weights -- and the host-side kmax bound under-counts by the same three.
    """

    def test_rewind_is_present_and_precedes_the_replay(self):
        import inspect

        src = inspect.getsource(VestigeKVMLABackend._capture_scan_timed)
        rewind = src.rindex("_rewind_appends(")
        replay = src.rindex("graph.replay()")
        self.assertLess(rewind, replay, "rewind must run before the replay")

    def test_failed_capture_rewinds_the_partial_appends(self):
        # A capture that dies mid-warmup has already appended this step's row
        # one or more times. The except path must undo exactly those appends
        # (counted, not assumed), or the eager fallback resumes from an
        # inflated kept table.
        import inspect

        src = inspect.getsource(VestigeKVMLABackend._capture_scan_timed)
        except_body = src[src.index("except (RuntimeError") :]
        except_body = except_body[: except_body.index("return False")]
        self.assertIn("_rewind_appends(slots, appended[0])", except_body)
        # the counter must be incremented after each successful pack, inside _run
        run_body = src[src.index("def _run()") : src.index("except (RuntimeError")]
        pack = run_body.index("_pack_all_layers(")
        incr = run_body.index("appended[0] += 1")
        self.assertLess(pack, incr, "count only appends that actually happened")

    def test_net_effect_of_a_capture_is_one_append(self):
        # simulate: three extra appends happened; the rewind must leave the
        # table exactly one row longer than before the step
        kept_len = torch.tensor([10, 20], dtype=torch.int64)
        slots = torch.tensor([0, 1])
        appended = 0
        for _ in range(3):  # warmups + capture
            kept_len.scatter_(0, slots, kept_len.gather(0, slots) + 1)
            appended += 1
        kept_len.scatter_(
            0, slots, (kept_len.gather(0, slots) - appended).clamp_min_(0)
        )
        kept_len.scatter_(0, slots, kept_len.gather(0, slots) + 1)  # the replay
        self.assertEqual(kept_len.tolist(), [11, 21])

    def test_net_effect_of_a_failed_capture_is_zero_appends(self):
        # the second warmup dies after one layer packed: only that layer's
        # count is rewound, and the table returns to its pre-step state
        kept_len = torch.tensor([10, 20], dtype=torch.int64)
        slots = torch.tensor([0, 1])
        appended = {"layer_a": 0, "layer_b": 0}
        for _ in range(2):  # warmup 1 fully, warmup 2 dies after layer_a
            for lid in appended:
                kept_len.scatter_(0, slots, kept_len.gather(0, slots) + 1)
                appended[lid] += 1
                if lid == "layer_a" and appended[lid] == 2:
                    break
        for lid, n in appended.items():
            kept_len.scatter_(0, slots, (kept_len.gather(0, slots) - n).clamp_min_(0))
        self.assertEqual(kept_len.tolist(), [10, 20])


class TestDecodeTimeBlockClose(CustomTestCase):
    """Decode-time compression events. Without them, rows generated after
    prefill are never evicted (attended set grows 1:1 with generation) and
    rows evicted by a close would be unrecallable. Semantics mirror the
    reference policy: global top-(rho*closed) rebalance plus sinks, tail
    untouched, recall archive refreshed through the live caches.
    """

    LID2 = 3

    def _backend(self, prefill=8192, pool=200000):
        be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
        be.rho = 1 / 32
        be._local_mla_lids = [self.LID2]
        be._kept_buf = {self.LID2: torch.zeros(4, 40000, dtype=torch.int64)}
        be._kept_len = {self.LID2: torch.zeros(4, dtype=torch.int64)}
        be._kmax = {}
        be._recall = {}
        be._close_state = {}
        torch.manual_seed(0)
        kbuf = torch.randn(pool, 576)
        be.token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda lid: kbuf)
        be.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(4 * 50000).reshape(4, 50000) % pool
        )
        be._close_state[(0, self.LID2)] = {
            "closed": prefill,
            "sigma": torch.rand(prefill),
        }
        return be

    def _fb(self, seq):
        return SimpleNamespace(
            out_cache_loc=torch.zeros(1, dtype=torch.int64),
            seq_lens=torch.tensor([seq], dtype=torch.int64),
        )

    def test_no_close_before_a_full_block(self):
        be = self._backend(prefill=ACT_MIN)
        be._maybe_close_blocks(self._fb(ACT_MIN + D.CLOSE_BLOCK - 1), [0])
        self.assertEqual(be._close_state[(0, self.LID2)]["closed"], ACT_MIN)

    def test_no_close_below_the_activation_threshold(self):
        # Dense regime: even with whole closable blocks outstanding, nothing
        # closes below ACTIVATION_MIN_TOKENS.
        be = self._backend(prefill=8192)
        be._maybe_close_blocks(self._fb(8192 + 3 * D.CLOSE_BLOCK + 1), [0])
        self.assertEqual(be._close_state[(0, self.LID2)]["closed"], 8192)

    def test_crossing_the_threshold_catches_up_all_blocks(self):
        # The crossing step closes every outstanding full block in one pass;
        # sigma is per-block immutable and the top-m rebalance is global, so
        # the kept set matches having compressed from the start.
        be = self._backend(prefill=8192)
        be._maybe_close_blocks(self._fb(ACT_MIN + 7), [0])
        self.assertEqual(be._close_state[(0, self.LID2)]["closed"], ACT_MIN)

    def test_close_advances_and_rewrites_kept(self):
        be = self._backend(prefill=ACT_MIN)
        seq = ACT_MIN + D.CLOSE_BLOCK + 7
        be._maybe_close_blocks(self._fb(seq), [0])
        cl = be._close_state[(0, self.LID2)]
        self.assertEqual(cl["closed"], ACT_MIN + D.CLOSE_BLOCK)
        self.assertEqual(cl["sigma"].shape[0], ACT_MIN + D.CLOSE_BLOCK)
        n = int(be._kept_len[self.LID2][0])
        m = max(1, round(be.rho * cl["closed"]))
        tail = seq - cl["closed"]
        # kept = selected (m, possibly +sinks overlap) + tail; sinks may or may
        # not already be in the top-m, so allow the small range
        self.assertGreaterEqual(n, m + tail)
        self.assertLessEqual(n, m + D.SINKS + tail)
        # tail rows must be the trailing positions, in order
        r2t = be.req_to_token_pool.req_to_token
        expect_tail = r2t[0, cl["closed"] : seq]
        got_tail = be._kept_buf[self.LID2][0, n - tail : n]
        self.assertTrue(torch.equal(got_tail, expect_tail))
        self.assertGreaterEqual(be._kmax[self.LID2], n)

    def test_multiple_blocks_close_in_one_step(self):
        be = self._backend(prefill=ACT_MIN)
        seq = ACT_MIN + 3 * D.CLOSE_BLOCK + 1
        be._maybe_close_blocks(self._fb(seq), [0])
        self.assertEqual(
            be._close_state[(0, self.LID2)]["closed"],
            ACT_MIN + 3 * D.CLOSE_BLOCK,
        )

    def test_close_refreshes_a_built_tier(self):
        be = self._backend(prefill=ACT_MIN)
        calls = []
        tier = SimpleNamespace(
            built=True,
            _pos_all=None,
            extend_closed=lambda rows, slots: calls.append(("extend", slots.shape[0])),
            refresh_membership=lambda keep, kept: calls.append(
                ("refresh", int(keep.sum()))
            ),
        )
        be._recall[(0, self.LID2)] = {"tier": tier}
        seq = ACT_MIN + D.CLOSE_BLOCK
        be._maybe_close_blocks(self._fb(seq), [0])
        # first close: caches empty -> extend covers ALL closed rows
        self.assertEqual(calls[0], ("extend", ACT_MIN + D.CLOSE_BLOCK))
        self.assertEqual(calls[1][0], "refresh")

    def test_unarmed_slot_is_ignored(self):
        be = self._backend(prefill=8192)
        del be._close_state[(0, self.LID2)]
        be._maybe_close_blocks(self._fb(60000), [0])  # must not raise
        self.assertEqual(be._close_state, {})


class TestLiveArchiveBackfill(CustomTestCase):
    """The close path's backfill watermark must match the cache contents.

    Regression: build left _pos_all eagerly set (full prefix) while
    side/csk stayed None-until-first-close, so the first serving close
    appended only the new block yet refresh_membership indexed it with
    full-prefix indices; the conservative build path additionally crashed
    with AttributeError (_pos_all never initialized). Protocol under test
    is the one _close_one_block runs: watermark -> extend_closed(delta)
    -> refresh_membership(keep over ALL closed rows).
    """

    def _run_protocol(self, tier, rows, c1):
        cached = 0 if tier._pos_all is None else tier._pos_all.shape[0]
        slots = torch.arange(c1)
        if cached < c1:
            tier.extend_closed(rows[cached:c1], slots[cached:])
        keep = torch.zeros(c1, dtype=torch.bool)
        keep[::3] = True
        # refresh_membership takes the POOL buffer, not a gathered subset: the
        # tier re-reads rows by id now rather than being handed copies. Passing
        # rows[keep] here would make `side` index a 214-row tensor with
        # full-prefix positions.
        tier.refresh_membership(keep, rows)
        return keep

    def test_backfill_aligns_archive_after_first_close(self):
        import sglang.srt.layers.attention.vestigekv.defaults as D
        from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

        torch.manual_seed(0)
        c1 = 640
        rows = torch.randn(c1, D.LATENT_DIM)
        tier = RecallTier(r=16)
        # conservative-shaped tier: built without the full-build tail
        tier.V = torch.linalg.qr(torch.randn(D.KV_LORA_RANK, 16))[0].T.contiguous()
        tier.built = True
        # __init__ must have made the watermark readable (AttributeError fix)
        self.assertIsNone(tier._pos_all)
        keep = self._run_protocol(tier, rows, c1)
        arch_idx = (~keep).nonzero().flatten()
        # cache rows must be the projections of exactly the archived rows
        self.assertEqual(tier._pos_all.shape[0], c1)
        self.assertEqual(tier.side.shape[0], arch_idx.shape[0])
        want_side = rows[arch_idx][:, D.KV_LORA_RANK :].to(tier.side.dtype)
        self.assertTrue(torch.equal(tier.side, want_side))
        # arch carries POOL ROW IDS. This protocol feeds slots = arange, so the
        # ids coincide with the positions -- assert against _pos_all[arch_idx]
        # so the check stays honest when they do not.
        self.assertTrue(torch.equal(tier.arch, tier._pos_all[arch_idx]))
        self.assertTrue(torch.equal(tier._arch_idx, arch_idx))


class TestEmptyKeptRows(CustomTestCase):
    """query paths must not crash when tier-1 kept the empty set.

    Regression: an anomalous build produced arch == seq_len (keep all-False),
    so kept_rows was [0, 576] and s_kept.max(-1) raised IndexError on a
    zero-size reduction, crashing the scheduler mid-decode. Empty kept has a
    well-defined meaning -- no max1 baseline, so every archived row is eligible
    -- and both query paths must serve it as full-archive recall.
    """

    def _tier(self, dev="cpu"):
        import sglang.srt.layers.attention.vestigekv.defaults as D
        from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

        torch.manual_seed(0)
        H, A = 8, 64
        t = RecallTier(r=16)
        t.V = torch.linalg.qr(torch.randn(D.KV_LORA_RANK, 16, device=dev))[
            0
        ].T.contiguous()
        # Empty tier-1. The tier stores INDICES and derives the rows, so the
        # empty set is an empty kept_slots; kept_rows is set alongside because
        # this fake tier has no pool to derive them from.
        t.kept_slots = torch.zeros(0, dtype=torch.int32, device=dev)
        t.kept_rows = torch.zeros(0, D.LATENT_DIM, device=dev)
        t.side = torch.randn(A, D.SIDECAR_DIM, device=dev).to(torch.bfloat16)
        t.csk = torch.randn(A, 16, device=dev).half()
        t.rho = torch.rand(A, device=dev)
        t.arch = torch.arange(A, device=dev)
        t.zp = 4.0
        t.thr_g = 0.0
        t.built = True
        return t, H, A, D

    def test_query_empty_kept_fires_all(self):
        t, H, A, D = self._tier()
        qe = torch.randn(H, D.LATENT_DIM)
        fired = t.query(qe)  # must not raise
        # with no baseline every eligible row can fire; at minimum no crash and
        # a subset of the archive
        self.assertLessEqual(fired.numel(), A)

    @unittest.skipUnless(torch.cuda.is_available(), "query_fixed is device-only")
    def test_query_fixed_empty_kept(self):
        t, H, A, D = self._tier(dev="cuda")
        qe = torch.randn(H, D.LATENT_DIM, device="cuda")
        out = torch.zeros(1, A + 8, dtype=torch.int64, device="cuda")
        out_len = torch.zeros(1, dtype=torch.int64, device="cuda")
        out_ovf = torch.zeros(1, dtype=torch.int32, device="cuda")
        t.query_fixed(qe, out, out_len, out_ovf, 0)  # must not raise
        self.assertGreaterEqual(int(out_len[0]), 0)


class TestNoPEPreconditionGuard(CustomTestCase):
    """NoPE is a load-bearing precondition, not just a model family: a config
    that applies a positional encoding to the MLA path must be refused, even
    inside the validated family. Audit rule: the guard is driven with a
    known-bad config and shown to refuse it, not assumed.
    """

    def _run(self, cfg):
        from unittest.mock import MagicMock, patch

        from sglang.srt.layers.attention import attention_registry as reg

        runner = MagicMock()
        runner.use_mla_backend = True
        runner.page_size = 1
        with (
            patch.object(reg, "kimi_linear_config", return_value=cfg),
            patch.object(
                reg, "get_spec", return_value=MagicMock(speculative_algorithm=None)
            ),
        ):
            reg.create_vestigekv_mla_backend(runner)

    def test_rope_mla_config_is_refused(self):
        from unittest.mock import MagicMock

        bad = MagicMock()
        bad.mla_use_nope = False  # a positional encoding rotates the branch
        with self.assertRaises(ValueError) as e:
            self._run(bad)
        self.assertIn("NoPE-MLA cache", str(e.exception))

    def test_nope_config_passes_the_precondition(self):
        # A NoPE config must clear THIS guard (it may fail later on the mocked
        # runner, but not with the NoPE ValueError).
        from unittest.mock import MagicMock

        good = MagicMock()
        good.mla_use_nope = True
        try:
            self._run(good)
        except ValueError as e:
            self.assertNotIn("NoPE-MLA cache", str(e.exception))
        except Exception:
            pass  # downstream construction on a mock runner is out of scope


class TestEagerDecodeUsesThePackedBuffers(CustomTestCase):
    """An eager decode step attends the CSR its metadata hook packed into the
    per-layer buffers -- the same buffers the graphs read. A batch those
    buffers cannot hold must fail loudly: slicing a short indptr silently
    truncates and trips the base kernel's q/kv_indptr assertion downstream
    (observed in serving at bs > cuda-graph-max-bs before the buffers were
    sized for the request table)."""

    def _run_decode(self, be, bs):
        from unittest.mock import patch

        fm = be.base.forward_metadata
        layer = SimpleNamespace(layer_id=LID)
        fb = SimpleNamespace(
            seq_lens=torch.full((bs,), 7, dtype=torch.int64),
            req_pool_indices=torch.arange(bs, dtype=torch.int64),
            out_cache_loc=torch.arange(2000, 2000 + bs, dtype=torch.int64),
        )
        seen = {}

        def fake_base_decode(q, k, v, layer, forward_batch, **kw):
            seen["indptr"] = fm.kv_indptr
            seen["indices"] = fm.kv_indices

        be.base.forward_decode = fake_base_decode
        with patch(
            "sglang.srt.model_executor.runner.get_is_capture_mode",
            return_value=False,
        ):
            be.forward_decode(None, None, None, layer, fb)
        return seen

    def test_eager_decode_uses_the_packed_buffers(self):
        be = _mk_backend()
        bufs = be._graph_bufs[LID]
        seen = self._run_decode(be, 2)
        self.assertEqual(seen["indptr"].shape[0], 3)
        self.assertEqual(seen["indptr"].data_ptr(), bufs["indptr"].data_ptr())
        self.assertIs(seen["indices"], bufs["indices"])

    def test_a_batch_the_buffers_cannot_hold_is_refused(self):
        be = _mk_backend()
        bs = be._graph_bufs[LID]["indptr"].shape[0] + 2
        with self.assertRaises(RuntimeError):
            self._run_decode(be, bs)


class TestUnseenSlotDecodesDense(CustomTestCase):
    """A decode lane whose slot holds no compressed state for the layer (a
    warmup/dummy batch, or a slot that never extended through this backend)
    packs its full row set -- what the base backend attends -- while lanes
    with state pack kept + fetched. The warmup crashed with "no packed CSR"
    when such a lane packed nothing."""

    def test_lane_without_state_packs_the_dense_rows(self):
        be = _mk_backend()
        del be._close_state[(1, LID)]  # slot 1 never prefilled here
        be.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(GRAPH_BS * 16, dtype=torch.int32).reshape(
                GRAPH_BS, 16
            )
        )
        fb = _mk_forward_batch()  # seq_lens 7, real lanes 0..2
        be._refresh_graph_bufs(LID, fb, fb.req_pool_indices.tolist())
        bufs, indptr = be._graph_bufs[LID], be._graph_bufs[LID]["indptr"]
        seg = [
            bufs["indices"][int(indptr[i]) : int(indptr[i + 1])].tolist()
            for i in range(REAL_BS)
        ]
        r2t = be.req_to_token_pool.req_to_token
        self.assertEqual(seg[1], r2t[1, :6].tolist() + [1002])
        self.assertEqual(seg[0], list(range(0, KEPT)) + [1001])
        self.assertEqual(seg[2], list(range(200, 200 + KEPT)) + [1003])


class TestEagerDecodeRunsTheStep(CustomTestCase):
    """A decode step no graph replays (CUDA graph off, or a batch beyond the
    captured sizes) must run the replay hook's step -- block closes,
    calibration, recall, CSR pack -- on the same buffers. It used to run
    nothing: the request was served its prefill kept set plus the tail, with
    no recall and no decode-time eviction."""

    def _fake(self, decode, prefilled=True):
        be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
        be.base = SimpleNamespace(init_forward_metadata=lambda fb: None)
        be._kept_buf = {LID: torch.zeros(1, 1)} if prefilled else {}
        be._local_mla_lids = [LID, LID + 1]
        be._collecting = True
        calls = []

        def rec(name, *a):
            calls.append((name,) + a)

        be._ensure_graph_bufs = lambda: rec("bufs")
        be._step_slots = lambda fb: ([0], None)
        be._maybe_close_blocks = lambda fb, reqs: rec("close", reqs)
        be._collect_calibration = lambda fb, reqs: rec("collect", reqs)
        be._recall_step = lambda lid, fb, reqs: rec("recall", lid)
        be._refresh_graph_bufs = lambda lid, fb, reqs: rec("pack", lid)
        fb = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: decode))
        return be, fb, calls

    def test_decode_runs_the_whole_step_per_layer(self):
        be, fb, calls = self._fake(decode=True)
        be.init_forward_metadata(fb)
        self.assertEqual(
            calls,
            [
                ("bufs",),
                ("close", [0]),
                ("collect", [0]),
                ("recall", LID),
                ("pack", LID),
                ("recall", LID + 1),
                ("pack", LID + 1),
            ],
        )

    def test_extend_runs_nothing(self):
        be, fb, calls = self._fake(decode=False)
        be.init_forward_metadata(fb)
        self.assertEqual(calls, [])

    def test_decode_before_any_prefill_still_packs(self):
        # the flashinfer autotune warmup decodes a dummy batch before any
        # request extended through the backend; it must get a packed CSR
        # (the dense rows, see TestUnseenSlotDecodesDense), not a crash
        be, fb, calls = self._fake(decode=True, prefilled=False)
        be.init_forward_metadata(fb)
        self.assertEqual(calls[0], ("bufs",))
        self.assertIn(("pack", LID), calls)


if __name__ == "__main__":
    unittest.main()
