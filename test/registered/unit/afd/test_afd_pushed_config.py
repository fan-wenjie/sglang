"""The pool owns the configuration; a host adopts it at the HELLO or is refused by name.

What these hold shut, in the order the handshake checks them: a payload that survives the
wire, a schema stamp a newer build refuses rather than guesses at, a CODE version that
catches two checkouts of one package, a checkpoint name that catches one model's attention
against another's feed-forward, and the arm hooks -- the pool folds every arm's settings in,
the host hands every arm its settings back, and a host that was ALSO given a setting
explicitly and differently is a contradiction, not a configuration.
"""

import socket
import threading
import unittest

import pytest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.afd import arms
from sglang.srt.afd.pool_client import PoolClient
from sglang.srt.afd.pool_server import Departure
from sglang.srt.afd.protocol import Frame, decode, send_frame
from sglang.srt.afd.pushed_config import (
    CONFIG_VERSION,
    adopt,
    code_version,
    decode_config,
    encode_config,
    pool_config,
)
from sglang.test.test_utils import CustomTestCase


@pytest.fixture(autouse=True)
def _no_inherited_runtime():
    """Every case here hands its own `server_args` in, and that only works with no bag published.

    `effective_model_path` answers from the config bag wherever one exists and falls back to the
    record only where none does -- which is right in a server and makes these cases depend on
    whatever ran before them. `afd_tiny_stack` publishes one for a throwaway checkpoint and
    cannot take it back (its four initialisations are process-global), so run after that file
    these cases were comparing the pool's name against `afd-tiny-stack-<tmpdir>` and failing on
    the identity guard several assertions before their own. Seven of them, green alone and red in
    the suite, which is the shape of a bug nobody chases because it looks like flakiness.
    """
    from sglang.srt.runtime_context import reset_context

    reset_context()
    yield
    reset_context()


def _args(**kw):
    base = {
        "model_path": "/models/Qwen3.8-27B",
        "afd_query_shift_layers": None,
        "afd_transfer_backend": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _stamped(**kw):
    cfg = {
        "config_version": CONFIG_VERSION,
        "code": code_version(),
        "model": "Qwen3.8-27B",
    }
    cfg.update(kw)
    return cfg


class TestThePayloadSurvivesTheWire(CustomTestCase):
    def test_encode_decode_round_trip(self):
        cfg = _stamped(query_shift_layers=1)
        tensor = encode_config(cfg)
        self.assertEqual(tensor.dtype, torch.int64)
        self.assertEqual(tensor.dim(), 2)
        self.assertEqual(decode_config(tensor), cfg)


class TestTheStampsAreCheckedFirst(CustomTestCase):
    def test_a_foreign_schema_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "schema"):
            adopt(_stamped(config_version=CONFIG_VERSION + 1), _args())

    def test_a_code_mismatch_is_refused_by_both_names(self):
        cfg = _stamped(code="0.0.0+deadbeef")
        with self.assertRaisesRegex(RuntimeError, "0.0.0\\+deadbeef"):
            adopt(cfg, _args())

    def test_a_different_checkpoint_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "Qwen3.8-27B"):
            adopt(_stamped(model="SomeOtherModel"), _args())

    def test_a_matching_word_is_adopted_silently(self):
        adopt(_stamped(), _args())

    def test_the_transfer_backend_rides_the_word(self):
        from sglang.srt.afd.pushed_config import adopted_transfer

        adopt(_stamped(transfer="nccl"), _args())
        self.assertEqual(adopted_transfer(), "nccl")
        adopt(_stamped(transfer="tcp"), _args())
        self.assertEqual(adopted_transfer(), "tcp")

    def test_a_contradicting_transfer_backend_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "afd-transfer-backend"):
            adopt(_stamped(transfer="tcp"), _args(afd_transfer_backend="nccl"))
        with self.assertRaisesRegex(RuntimeError, "afd-transfer-backend"):
            adopt(_stamped(transfer="nccl"), _args(afd_transfer_backend="tcp"))

    def test_an_unset_host_adopts_whatever_the_pool_decided(self):
        from sglang.srt.afd.pushed_config import adopted_transfer

        adopt(_stamped(transfer="nccl"), _args())
        self.assertEqual(adopted_transfer(), "nccl")


class TestTheArmsSupplyAndTakeTheirOwn(CustomTestCase):
    def setUp(self):
        self.pushed, self.taken = [], []

        class FakeArm:
            @staticmethod
            def pushed_config(server_args):
                self.pushed.append(server_args)
                return {"fake_setting": 7}

            @staticmethod
            def adopt_config(cfg, server_args):
                self.taken.append(cfg["fake_setting"])

        self._had = arms._ARMS.get("fake-config-arm")
        arms._ARMS["fake-config-arm"] = FakeArm

    def tearDown(self):
        if self._had is None:
            arms._ARMS.pop("fake-config-arm", None)
        else:
            arms._ARMS["fake-config-arm"] = self._had

    def test_the_pool_folds_every_arm_in(self):
        cfg = pool_config(_args())
        self.assertEqual(cfg["fake_setting"], 7)
        self.assertEqual(cfg["model"], "Qwen3.8-27B")

    def test_the_host_hands_every_arm_its_settings(self):
        adopt(_stamped(fake_setting=7), _args())
        self.assertEqual(self.taken, [7])


class TestTheHelloCarriesTheWord(CustomTestCase):
    """The frame itself, through a real socket and the real client."""

    def _pool(self, pushed):
        departure = Departure(
            lambda batch, layer: batch, 1, 0.005, "cpu", pushed=pushed
        )
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)

        def serve():
            try:
                conn, _ = listener.accept()
                while True:
                    frame = decode(conn)
                    if frame is None:
                        return
                    if not departure.answer_directly(frame, conn):
                        send_frame(
                            conn, Frame.one(frame.request_id, frame.layer, frame.tensor)
                        )
            except OSError:
                return

        threading.Thread(target=serve, daemon=True).start()
        return listener, listener.getsockname()[1]

    def test_a_pool_with_a_word_pushes_it(self):
        cfg = _stamped(query_shift_layers=1)
        listener, port = self._pool(encode_config(cfg))
        client = PoolClient(f"127.0.0.1:{port}", 5.0)
        try:
            self.assertEqual(client.fetch_config(), cfg)
        finally:
            client.close()
            listener.close()

    def test_a_pool_without_one_answers_none(self):
        listener, port = self._pool(None)
        client = PoolClient(f"127.0.0.1:{port}", 5.0)
        try:
            self.assertIsNone(client.fetch_config())
        finally:
            client.close()
            listener.close()


class TestTheQueryShiftArmAdopts(CustomTestCase):
    def setUp(self):
        pushed = pytest.importorskip(
            "sglang.srt.afd_query_shift.pushed",
            reason="the derived package is absent by design on the base tree",
        )

        self._pushed = pushed
        self._layers = pushed._ADOPTED_LAYERS[0]

    def tearDown(self):
        self._pushed._ADOPTED_LAYERS[0] = self._layers

    def test_an_unconfigured_host_takes_the_pools_word(self):
        from sglang.srt.afd.checkpoint import requested_shift
        from sglang.srt.afd_query_shift.installer import QueryShiftConfig

        args = _args()
        QueryShiftConfig.adopt_config({"query_shift_layers": 1}, args)
        # the flag stays None -- server_args is the pristine record -- and the sanctioned
        # reader answers with the pool's word
        self.assertIsNone(args.afd_query_shift_layers)
        self.assertEqual(requested_shift(args), 1)

    def test_an_explicitly_contradicting_flag_is_refused(self):
        from sglang.srt.afd_query_shift.installer import QueryShiftConfig

        with self.assertRaisesRegex(RuntimeError, "afd-query-shift-layers"):
            QueryShiftConfig.adopt_config(
                {"query_shift_layers": 1},
                _args(afd_query_shift_layers=3),
            )

    def test_an_explicitly_equal_setting_is_redundant_not_wrong(self):
        from sglang.srt.afd_query_shift.installer import QueryShiftConfig

        args = _args(afd_query_shift_layers=1)
        QueryShiftConfig.adopt_config({"query_shift_layers": 1}, args)
        self.assertEqual(self._pushed.adopted_layers(), 1)


if __name__ == "__main__":
    unittest.main()
