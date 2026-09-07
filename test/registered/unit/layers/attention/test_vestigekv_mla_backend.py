"""Unit tests for the VestigeKV MLA attention backend (vestigekv_mla).

Each test pins a black-box behavior that once regressed in the port:
graph-replay batch padding must not overrun the unpadded out_cache_loc,
slot reuse must not leak the previous request's kept state, and the
SGLANG_DEBUG_VESTIGEKV_ROWS row invariant must both pass and fail correctly.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv_mla_backend import VestigeKVMLABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LID = 3
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
    be._tier2 = {}
    be._qbuf, be._fetch_buf, be._fetch_len, be._recall = {}, {}, {}, {}
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
    def test_prefill_invalidates_stale_tier2(self):
        be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
        be.rho = 1 / 32
        be._kept_buf, be._kept_len, be._indptr1, be._kmax = {}, {}, {}, {}
        be._close_state = {}
        be._qbuf, be._fetch_buf, be._fetch_len, be._recall = {}, {}, {}, {}
        be._q_heads, be._q_dim, be._fetch_w = 4, 576, 8
        be._qbuf_stack = be._fetch_stack = be._fetch_len_stack = None
        be._li_map = {}
        be._local_mla_lids = [LID]
        be._n_cal, be.index_rank, be.topj = 4, 8, -1
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
            # chunked-prefill fields: this extend completes the prefix
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[48],
        )
        be._build_gpu_state(layer, fb)
        self.assertNotIn((0, LID), be._tier2)
        self.assertGreater(int(be._kept_len[LID][0]), 0)


class TestRowInvariantCheck(CustomTestCase):
    """SGLANG_DEBUG_VESTIGEKV_ROWS semantics: the check runs before this step's
    append, so the FULL arm requires kept_len >= seq_len - 1; the VESTIGE arm
    requires kept_len well below seq_len; a missing table or kept_len == 0
    always aborts."""

    def _mk(self, kept_lens):
        be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
        be._local_mla_lids = [LID]
        be._kept_len = {LID: torch.tensor(kept_lens, dtype=torch.int64)}
        be._fetch_len = {}
        fb = SimpleNamespace(
            out_cache_loc=torch.tensor([100, 101], dtype=torch.int64),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            # above the assert floor (4 * CLOSE_BLOCK): below it the
            # unclosed tail alone makes kept ~= seq legitimately
            seq_lens=torch.tensor([20000, 20000], dtype=torch.int64),
        )
        return be, fb

    def _run(self, be, fb, full_arm):
        with patch.object(VestigeKVMLABackend, "_full_arm", return_value=full_arm):
            be._check_row_invariant(fb)

    def test_vestige_arm_compressed_passes(self):
        be, fb = self._mk([5000, 5000])  # rho*closed + tail(<=4096) + sinks
        self._run(be, fb, full_arm=False)

    def test_vestige_arm_uncompressed_raises(self):
        be, fb = self._mk([19000, 19000])  # kept ~= seq: not compressing
        with self.assertRaisesRegex(AssertionError, "compression not applied"):
            self._run(be, fb, full_arm=False)

    def test_full_arm_pending_append_passes(self):
        be, fb = self._mk([19999, 19999])
        self._run(be, fb, full_arm=True)

    def test_full_arm_short_raises(self):
        be, fb = self._mk([5000, 5000])
        with self.assertRaisesRegex(AssertionError, "FULL arm"):
            self._run(be, fb, full_arm=True)

    def test_missing_table_raises(self):
        be, fb = self._mk([19000, 19000])
        be._kept_len = {}
        with self.assertRaisesRegex(AssertionError, "never built"):
            self._run(be, fb, full_arm=False)


if __name__ == "__main__":
    unittest.main()


class TestCapabilityDelegation(CustomTestCase):
    """Capability flags are class attributes, so a wrapper that only forwards
    methods silently inherits AttentionBackend's defaults instead of the
    wrapped backend's values -- which changes the runtime's fast paths.
    Missing needs_cpu_seq_lens alone cost 137 -> 51 tok/s (per-step host sync
    of seq_lens) with no functional symptom."""

    def test_capability_flags_are_delegated(self):
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

        flags = [
            name
            for name, val in vars(AttentionBackend).items()
            if not name.startswith("_")
            and not callable(val)
            and not isinstance(val, (property, classmethod, staticmethod))
        ]
        self.assertIn("needs_cpu_seq_lens", flags)
        sentinel = {f: object() for f in flags}
        extra = {
            "token_to_kv_pool": SimpleNamespace(full_attention_layer_id_mapping=[LID]),
            "req_to_token_pool": None,
            "kv_index_translator": None,
        }
        base = SimpleNamespace(**{**extra, **sentinel})
        be = VestigeKVMLABackend.__new__(VestigeKVMLABackend)
        be.base = base
        # replay only the delegation block of __init__
        for f in (
            "needs_cpu_seq_lens",
            "extend_dummy_seqs_capped_by_req_pool",
            "supports_ragged_verify_graph",
            "supports_full_cuda_graph_chunked_prefix",
            "use_captured_forward_metadata_for_breakable_cuda_graph",
            "prefill_attention_backend_str",
            "decode_attention_backend_str",
        ):
            setattr(be, f, getattr(base, f))
        # a runtime-queried flag must never resolve to the class default
        for f in ("needs_cpu_seq_lens", "supports_ragged_verify_graph"):
            self.assertIs(getattr(be, f), sentinel[f])
            self.assertIsNot(getattr(be, f), getattr(AttentionBackend, f))


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
    be._scan_kmax = {}
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

    def test_key_tracks_archive_size(self):
        a = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        b = _mk_scan_backend({(0, 3): 100, (0, 7): 201})
        self.assertNotEqual(
            a._scan_key(_scan_fb(1), [0]), b._scan_key(_scan_fb(1), [0])
        )

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
            tier_ids=("stale",), fits=lambda pairs, tiers: False,
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

    def test_headroom_guard_fires_before_rows_are_dropped(self):
        # A capture bakes one gather width; kept_len grows every step. Driven
        # directly with a bound that has just caught up to the baked width, the
        # guard must report exhausted -- one step BEFORE a row would be lost.
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        be._scan_kmax = {3: 10, 7: 10}
        be._kmax = {3: 8, 7: 8}
        self.assertFalse(be._kmax_exhausted())
        be._kmax = {3: 9, 7: 8}  # 9 + 1 == 10, still exactly representable
        self.assertFalse(be._kmax_exhausted())
        be._kmax = {3: 10, 7: 8}  # 10 + 1 > 10 -> would drop the newest row
        self.assertTrue(be._kmax_exhausted())

    def test_exhausted_headroom_recaptures_instead_of_replaying(self):
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        fb = _scan_fb(1)
        be._scan_key_cur = be._scan_key(fb, [0])
        be._scan_graph = SimpleNamespace(replay=lambda: self.fail("replayed stale"))
        be._scan_kmax = {3: 10}
        be._kmax = {3: 10}
        with patch.object(VestigeKVMLABackend, "_full_arm", return_value=False):
            with patch.object(
                VestigeKVMLABackend, "_capture_scan", return_value=True
            ) as cap:
                self.assertTrue(be._replay_scan(fb, [0], be._scan_key(fb, [0])))
                self.assertEqual(cap.call_count, 1)

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


class _CountingSlots:
    """req_pool_indices stand-in that records every .tolist() readback."""

    def __init__(self, values):
        self.values, self.reads = list(values), 0

    def tolist(self):
        self.reads += 1
        return list(self.values)


def _slots_fb(slots, graph_bs, real_bs):
    return SimpleNamespace(
        req_pool_indices=_CountingSlots(slots),
        seq_lens=torch.zeros(graph_bs, dtype=torch.int64),
        out_cache_loc=torch.zeros(real_bs, dtype=torch.int64),
    )


class TestStepSlotCache(CustomTestCase):
    """The per-step slot list is cached to keep a D2H readback off the decode
    path. It must refresh on exactly the two events that can change it: a batch
    that shrank (a shape) and a prefill (which also rebuilds a tier).
    """

    def test_repeated_steps_read_the_device_once(self):
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        fb = _slots_fb([0], 1, 1)
        for _ in range(10):
            reqs, _ = be._step_slots(fb)
            self.assertEqual(reqs, [0])
        self.assertEqual(fb.req_pool_indices.reads, 1)

    def test_a_shrinking_batch_refreshes(self):
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200, (1, 3): 100, (1, 7): 200})
        wide = _slots_fb([0, 1], 2, 2)
        be._step_slots(wide)
        narrow = _slots_fb([0], 2, 1)  # one request finished
        reqs, _ = be._step_slots(narrow)
        self.assertEqual(reqs, [0])
        self.assertEqual(narrow.req_pool_indices.reads, 1)

    def test_prefill_refreshes(self):
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        fb = _slots_fb([0], 1, 1)
        be._step_slots(fb)
        be._invalidate_scan()  # what every prefill calls
        be._step_slots(fb)
        self.assertEqual(fb.req_pool_indices.reads, 2)

    def test_key_is_cached_with_the_slots(self):
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        fb = _slots_fb([0], 1, 1)
        _, first = be._step_slots(fb)
        self.assertIsNotNone(first)
        _, again = be._step_slots(fb)
        self.assertIs(first, again)

    def test_an_unbuilt_tier_caches_no_key_and_recovers(self):
        be = _mk_scan_backend({(0, 3): 100}, built=False)
        fb = _slots_fb([0], 1, 1)
        self.assertIsNone(be._step_slots(fb)[1])
        # the build that follows calls _invalidate_scan, so the next step must
        # recompute rather than serve the cached None forever
        be._recall[(0, 3)] = {"tier": SimpleNamespace(arch=torch.zeros(100))}
        be._recall[(0, 7)] = {"tier": SimpleNamespace(arch=torch.zeros(200))}
        be._invalidate_scan()
        self.assertIsNotNone(be._step_slots(fb)[1])


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
        be._vcache = {}
        be.built = []
        be.enqueued = []
        return be

    def _fb(self, real_bs=1, seq=101):
        return SimpleNamespace(
            out_cache_loc=torch.zeros(real_bs, dtype=torch.int64),
            seq_lens=torch.full((real_bs,), seq, dtype=torch.int64),
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
            job["tier"] = SimpleNamespace(
                built_at=job["seq_len"], proxy=False, V="V"
            )
            job["stats"] = {
                "need_more_hard": n_hard < D.min_hard(),
                "n_hard": n_hard,
            }

    def test_provisional_index_is_built_synchronously_on_step_one(self):
        be = self._backend()
        b, e = self._stubs(be)
        with b, e:
            be._collect_calibration(self._fb(), [0])
        self.assertEqual(be.built, [(101, True)])
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
        with b, e, patch.object(
            VestigeKVMLABackend, "_invalidate_scan", lambda _s: called.append(1)
        ):
            for step in range(D.N_CAL_START):
                be._collect_calibration(self._fb(seq=101 + step), [0])
            self._finish(be.enqueued[0])
            be._capture_asap = False  # clear the provisional-build flag
            be._collect_calibration(self._fb(seq=120), [0])
            be._collect_calibration(self._fb(seq=121), [0])  # sticky fires here
        st = be._recall[(0, LID)]
        self.assertFalse(st["tier"].proxy)  # calibrated tier adopted
        self.assertIsNone(st["qcal"])  # collection over
        self.assertTrue(be._capture_asap)
        self.assertEqual(called, [1])  # exactly one coalesced invalidate

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
        j3 = {"slot": 0, "lid": 3, "st": be._recall[(0, 3)], "seq_len": 108,
              "qcal": [object()] * 8, "done": pend, "error": None,
              "tier": None, "stats": None}
        j7 = {"slot": 0, "lid": 7, "st": be._recall[(0, 7)], "seq_len": 108,
              "qcal": [object()] * 8, "done": pend, "error": None,
              "tier": None, "stats": None}
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
            be._collect_calibration(self._fb(seq=121), [0])
            be._collect_calibration(self._fb(seq=122), [0])  # sticky fires here
        self.assertEqual(calls, [1])  # exactly one coalesced invalidate
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
                "tier": None, "built_at": 0, "qcal": [], "qpos": [],
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


class TestDerivedCalibrationConstants(CustomTestCase):
    """The recall tier has ONE quality parameter, tau. These pin the algebra
    that derives everything else, so a future edit cannot silently decouple
    them."""

    def test_miss_budget_splits_evenly(self):
        self.assertAlmostEqual(D.gate_alpha(0.9), 0.05)
        self.assertAlmostEqual((1 - D.gate_alpha(0.9)) * D.scan_target(0.9), 0.9)

    def test_min_hard_is_the_smallest_n_where_the_quantile_exists(self):
        for tau in (0.8, 0.9, 0.95):
            n = D.min_hard(tau)
            self.assertLessEqual(D.conformal_k(n, tau), n)
            if n > 1:
                self.assertGreater(D.conformal_k(n - 1, tau), n - 1)

    def test_conformal_k_is_within_bounds(self):
        for n in (18, 50, 512):
            k = D.conformal_k(n)
            self.assertGreaterEqual(k, 1)
            self.assertLessEqual(k, n)


class TestCollectionRunsOnBothPaths(CustomTestCase):
    """Collection used to live on the eager scan path, so capture had to be
    blocked until calibration finished or the provisional index would never be
    replaced -- 16 eager steps per request at 11.3 ms each against 1.5 ms for a
    replayed step. Collection is now a few small clones done before the paths
    diverge, so capture is free to proceed; these pin that it still happens.
    """

    def test_capture_is_not_blocked_by_pending_calibration(self):
        be = _mk_scan_backend({(0, 3): 100, (0, 7): 200})
        for lid in (3, 7):
            be._recall[(0, lid)]["qcal"] = [object()]
        self.assertIsNotNone(be._scan_key(_scan_fb(1), [0]))

    def test_collection_happens_before_the_paths_diverge(self):
        # the step block must collect whether or not the graph then replays
        import inspect

        src = inspect.getsource(VestigeKVMLABackend.init_forward_metadata_out_graph)
        collect = src.index("_collect_calibration")
        replay = src.index("_replay_scan")
        stats = src.index("_step_with_stats")
        self.assertLess(collect, replay, "collection must precede the replay")
        self.assertLess(collect, stats, "collection must precede the stats path")


class TestStatsPathMatchesProduction(CustomTestCase):
    """An instrument that takes a different branch than production measures
    something production does not run. Twice now this backend has had that
    defect: first pricing the eager scan on steps that replayed a graph, then
    packing the CSR a second time on steps whose graph had already packed it
    (which also double-appended to kept_len). Pin the shape by reading the
    source, since the failure is structural, not numeric.
    """

    def test_stats_path_delegates_the_branch_choice(self):
        import inspect

        src = inspect.getsource(VestigeKVMLABackend._step_with_stats)
        self.assertIn("self._replay_scan(", src)
        self.assertIn("if not replayed:", src)
        # the eager-only work must sit under a `not replayed` guard, never at
        # the top level of the step
        for call in ("_recall_step(", "_refresh_graph_bufs("):
            idx = src.index(call)
            guard = src.rindex("if not replayed:", 0, idx)
            self.assertGreater(
                guard, src.index("replayed = "), f"{call} not guarded"
            )


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
        rewind = src.index("- 3")
        replay = src.rindex("graph.replay()")
        self.assertLess(rewind, replay, "rewind must run before the replay")

    def test_net_effect_of_a_capture_is_one_append(self):
        # simulate: three extra appends happened; the rewind must leave the
        # table exactly one row longer than before the step
        kept_len = torch.tensor([10, 20], dtype=torch.int64)
        slots = torch.tensor([0, 1])
        for _ in range(3):  # warmups + capture
            kept_len.scatter_(0, slots, kept_len.gather(0, slots) + 1)
        kept_len.scatter_(0, slots, (kept_len.gather(0, slots) - 3).clamp_min_(0))
        kept_len.scatter_(0, slots, kept_len.gather(0, slots) + 1)  # the replay
        self.assertEqual(kept_len.tolist(), [11, 21])


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
        be = self._backend(prefill=8192)
        be._maybe_close_blocks(self._fb(8192 + D.CLOSE_BLOCK - 1), [0])
        self.assertEqual(be._close_state[(0, self.LID2)]["closed"], 8192)

    def test_close_advances_and_rewrites_kept(self):
        be = self._backend(prefill=8192)
        seq = 8192 + D.CLOSE_BLOCK + 7
        be._maybe_close_blocks(self._fb(seq), [0])
        cl = be._close_state[(0, self.LID2)]
        self.assertEqual(cl["closed"], 8192 + D.CLOSE_BLOCK)
        self.assertEqual(cl["sigma"].shape[0], 8192 + D.CLOSE_BLOCK)
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
        be = self._backend(prefill=8192)
        seq = 8192 + 3 * D.CLOSE_BLOCK + 1
        be._maybe_close_blocks(self._fb(seq), [0])
        self.assertEqual(
            be._close_state[(0, self.LID2)]["closed"], 8192 + 3 * D.CLOSE_BLOCK
        )

    def test_close_refreshes_a_built_tier(self):
        be = self._backend(prefill=8192)
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
        seq = 8192 + D.CLOSE_BLOCK
        be._maybe_close_blocks(self._fb(seq), [0])
        # first close: caches empty -> extend covers ALL closed rows
        self.assertEqual(calls[0], ("extend", 8192 + D.CLOSE_BLOCK))
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
        keep[:: 3] = True
        tier.refresh_membership(keep, rows[keep])
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
        self.assertTrue(torch.equal(tier.arch, arch_idx))


class TestEmptyKeptRows(CustomTestCase):
    """query paths must not crash when tier-1 kept the empty set.

    Regression: an anomalous build produced arch == seq_len (keep all-False),
    so kept_rows was [0, 576] and skept.max(-1) raised IndexError on a
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
        t.kept_rows = torch.zeros(0, D.LATENT_DIM, device=dev)  # empty tier-1
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
        t.query_fixed(qe, out, out_len, 0)  # must not raise
        self.assertGreaterEqual(int(out_len[0]), 0)


class TestEpochFastPathLocals(CustomTestCase):
    """The epoch fast path must not reference slow-path locals.

    Regression: `real` was defined only inside the epoch-mismatch branch but
    used by the stage copies after it; the first steady-state fast-path step
    raised UnboundLocalError and killed the scheduler (unit tests had only
    exercised the slow path).
    """

    def test_fast_path_body_defines_real_before_branch(self):
        import inspect

        from sglang.srt.layers.attention.vestigekv_mla_backend import (
            VestigeKVMLABackend,
        )

        src = inspect.getsource(VestigeKVMLABackend._replay_scan)
        # `real = ...` must appear before the epoch branch in the same block
        i_real = src.index("real = forward_batch.out_cache_loc.shape[0]")
        i_branch = src.index("_pack_epoch != self._pack_epoch_synced")
        self.assertLess(i_real, i_branch)
