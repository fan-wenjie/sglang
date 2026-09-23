"""Static per-layer roles answer the same way at capture and at replay.

The whole point of making the role a property of the layer rather than of the
step is that the Python branch runs once, during capture, and the replay
inherits it. A predicate that consulted anything per-step -- the overflow
probe, the graph variant, the batch -- would bake whatever happened to be
true at capture and then be wrong for every later step, silently, since a
replay reruns no Python. So what is pinned here is that the answer depends on
the layer id alone, and that an empty set leaves the dynamic rule untouched.
"""

import unittest
from unittest import mock

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as kpool
from sglang.srt.layers.attention.vestigekv_dsa_backend import VestigeKVDSABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DSA_LAYERS = frozenset({3, 19, 23, 27, 39})
ALL_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]


def _backend(dsa_only):
    # No engine: the predicate reads one attribute off the backend, and
    # building a real one needs a model runner and a KV pool.
    vk = object.__new__(VestigeKVDSABackend)
    vk.dsa_only_layers = dsa_only
    vk.lean_step = False
    return vk


class TestStaticLayerSplit(CustomTestCase):
    def _lean(self, *, lid, dsa_only, capture):
        with mock.patch.object(
            kpool, "_vestigekv_decode_backend", return_value=_backend(dsa_only)
        ), mock.patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.get_is_capture_mode",
            return_value=capture,
        ):
            return kpool._vestigekv_lean_step(lid)

    def test_capture_and_replay_agree_per_layer(self):
        # The captured answer is the one the replay runs with, so it is the
        # only answer; it must be a function of the layer id.
        for lid in ALL_LAYERS:
            self.assertEqual(
                self._lean(lid=lid, dsa_only=DSA_LAYERS, capture=True),
                lid not in DSA_LAYERS,
                f"layer {lid}",
            )

    def test_dsa_layers_keep_the_indexer(self):
        for lid in sorted(DSA_LAYERS):
            self.assertFalse(self._lean(lid=lid, dsa_only=DSA_LAYERS, capture=True))

    def test_eager_steps_keep_the_indexer_on_every_layer(self):
        # The eager path serves through the CSR pack, whose fenced fallback is
        # sized for DSA's selection; dropping the selection there would size
        # a dense lane by a budget meant for a sparse one.
        for lid in ALL_LAYERS:
            self.assertFalse(self._lean(lid=lid, dsa_only=DSA_LAYERS, capture=False))

    def test_empty_set_leaves_the_dynamic_rule(self):
        # Default: the step-global rule decides, so an eager step reads the
        # backend's stale-by-one flag (False on this stub) rather than the
        # layer id.
        for lid in ALL_LAYERS:
            self.assertFalse(self._lean(lid=lid, dsa_only=frozenset(), capture=False))


if __name__ == "__main__":
    unittest.main()
