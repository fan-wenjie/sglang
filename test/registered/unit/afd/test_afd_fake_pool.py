"""A pool with no model behind it, for testing the host's half alone.

The fake pool owns a real socket and speaks the real wire -- frames in,
frames out, through the same encode/decode as production -- so anything a
host-side path does that the wire cannot carry fails HERE, not on a live
pair. What it does not own is a model: each op is answered by a handler the
test supplies, which is the point -- the host under test cannot tell.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import json
import os
import socket
import tempfile
import threading

import torch
import torch.nn as nn

from sglang.test.test_utils import CustomTestCase

from sglang.srt.afd import model_files
from sglang.srt.afd.model_files import files_reply
from sglang.srt.afd.pool_client import PoolClient
from sglang.srt.afd.protocol import OP_FILES, OP_WEIGHTS, Frame, decode, send_frame


class FakePool:
    """One listening socket, one serving thread, handlers instead of a model."""

    def __init__(self, handlers):
        self.handlers = handlers
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(4)
        self.port = self._server.getsockname()[1]
        self.address = f"127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            try:
                sock, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve_one, args=(sock,), daemon=True).start()

    def _serve_one(self, sock):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            frame = decode(sock)
            if frame is None:
                return
            reply = self.handlers[frame.op](frame)
            send_frame(sock, Frame(frame.request_id, frame.layer, reply, frame.op))

    def close(self):
        self._server.close()


def _checkpoint():
    d = tempfile.mkdtemp()
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tok = Tokenizer(WordLevel({"hello": 0, "world": 1, "[UNK]": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    open(os.path.join(d, "tokenizer.json"), "w").write(tok.to_str())
    open(os.path.join(d, "config.json"), "w").write(
        '{"model_type": "no_such_family", "hidden_size": 8}'
    )
    open(os.path.join(d, "tokenizer_config.json"), "w").write("{}")
    return d


class ToyLinearLayer(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.linear_attn = nn.Module()
        self.linear_attn.conv1d = nn.Conv1d(4, 4, 3, groups=4, bias=bias)


class ToyAttentionLayer(nn.Module):
    pass


def _toy_skeleton(bias=True):
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList(
        [ToyLinearLayer(bias), ToyAttentionLayer(), ToyLinearLayer(bias)]
    )
    model.model.norm = nn.Module()
    model.model.norm.weight = nn.Parameter(torch.zeros(8))
    return model


class TestTheHostAgainstAFakePool(CustomTestCase):
    def test_papers_provision_over_the_real_wire(self):
        d = _checkpoint()
        pool = FakePool({OP_FILES: lambda frame: files_reply(d)})
        try:
            model_files._PAPERS.pop(pool.address, None)
            path = "pool://" + pool.address
            cfg = model_files.pool_config(path)
            self.assertEqual(cfg.hidden_size, 8)
            tok = model_files.pool_tokenizer(path)
            self.assertEqual(tok.encode("hello world"), [0, 1])
        finally:
            model_files._PAPERS.pop(pool.address, None)
            pool.close()

    def test_residual_weights_arrive_and_land(self):
        self._weights_round_trip(bias=True)

    def test_a_meta_built_module_gains_its_storage(self):
        # the host builds the linear layers with NO storage; the push is what
        # materialises them
        donor = _toy_skeleton()
        for layer in (donor.model.layers[0], donor.model.layers[2]):
            nn.init.uniform_(layer.linear_attn.conv1d.weight, 1.0, 2.0)
            nn.init.uniform_(layer.linear_attn.conv1d.bias, 1.0, 2.0)
        nn.init.uniform_(donor.model.norm.weight, 1.0, 2.0)

        def weights(frame):
            out = []
            for layer in (donor.model.layers[0], donor.model.layers[2]):
                conv = layer.linear_attn.conv1d
                out.append(conv.weight.detach().reshape(1, -1))
                out.append(conv.bias.detach().reshape(1, -1))
            out.append(donor.model.norm.weight.detach().reshape(1, -1))
            return tuple(out)

        pool = FakePool({OP_WEIGHTS: weights})
        client = PoolClient(pool.address, 10.0, reconnect=False)
        try:
            from sglang.srt.afd.installer import _pull_residual_weights

            host = _toy_skeleton()
            with torch.device("meta"):
                meta_stack = _toy_skeleton()
            for i in (0, 2):
                host.model.layers[i] = meta_stack.model.layers[i]
            _pull_residual_weights(host, client)
            for i in (0, 2):
                got = host.model.layers[i].linear_attn.conv1d.weight
                self.assertFalse(got.is_meta)
                torch.testing.assert_close(
                    got.cpu(), donor.model.layers[i].linear_attn.conv1d.weight
                )
        finally:
            client.close()
            pool.close()

    def test_a_biasless_filter_still_rides(self):
        # a conv with no bias sends a (1, 0) placeholder; an empty tensor must
        # survive the wire, not crash the pool's encoder
        self._weights_round_trip(bias=False)

    def _weights_round_trip(self, bias):
        # the pool's half of the push, run against a toy stack of the same
        # shape, THROUGH the wire -- a tensor the frame cannot carry fails here
        donor = _toy_skeleton(bias)
        for layer in (donor.model.layers[0], donor.model.layers[2]):
            nn.init.uniform_(layer.linear_attn.conv1d.weight, 1.0, 2.0)
            if bias:
                nn.init.uniform_(layer.linear_attn.conv1d.bias, 1.0, 2.0)
        nn.init.uniform_(donor.model.norm.weight, 1.0, 2.0)

        def weights(frame):
            out = []
            for layer in (donor.model.layers[0], donor.model.layers[2]):
                conv = layer.linear_attn.conv1d
                out.append(conv.weight.detach().reshape(1, -1))
                out.append(
                    conv.bias.detach().reshape(1, -1)
                    if conv.bias is not None
                    else torch.zeros(1, 0)
                )
            out.append(donor.model.norm.weight.detach().reshape(1, -1))
            return tuple(out)

        pool = FakePool({OP_WEIGHTS: weights})
        client = PoolClient(pool.address, 10.0, reconnect=False)
        try:
            from sglang.srt.afd.installer import _pull_residual_weights

            host = _toy_skeleton(bias)
            _pull_residual_weights(host, client)
            for a, b in zip(host.parameters(), donor.parameters()):
                torch.testing.assert_close(a, b)
        finally:
            client.close()
            pool.close()

    def test_a_torn_weight_push_is_refused_by_name(self):
        pool = FakePool({OP_WEIGHTS: lambda frame: (torch.zeros(1, 4),)})
        client = PoolClient(pool.address, 10.0, reconnect=False)
        try:
            from sglang.srt.afd.installer import _pull_residual_weights

            with self.assertRaisesRegex(RuntimeError, "different checkpoints"):
                _pull_residual_weights(_toy_skeleton(), client)
        finally:
            client.close()
            pool.close()


if __name__ == "__main__":
    unittest.main()
