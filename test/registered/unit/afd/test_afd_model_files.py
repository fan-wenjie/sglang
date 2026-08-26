"""The pool serves the model's papers; the host builds from memory, never from disk.

Round trip pinned: the manifest and the bytes survive the wire's tensor form, a
weight never makes the list, a pushed name cannot pose as a path, and the config
comes back as a real PretrainedConfig without any filesystem underneath.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import json
import os
import tempfile

import torch

from sglang.test.test_utils import CustomTestCase

from sglang.srt.afd import model_files
from sglang.srt.afd.model_files import (
    files_reply,
    papers_from_reply,
    pool_config,
    servable_files,
)


class TestThePapersRoundTrip(CustomTestCase):
    def _checkpoint(self):
        d = tempfile.mkdtemp()
        open(os.path.join(d, "config.json"), "w").write(
            '{"model_type": "no_such_family", "hidden_size": 8}'
        )
        open(os.path.join(d, "tokenizer.json"), "w").write('{"vocab": {}}')
        open(os.path.join(d, "model.safetensors.json"), "w").write("{}")
        open(os.path.join(d, "model-00001.safetensors"), "wb").write(b"\0" * 64)
        return d

    def test_weights_never_make_the_list(self):
        names = servable_files(self._checkpoint())
        self.assertEqual(names, ["config.json", "tokenizer.json"])

    def test_bytes_survive_the_wire_form(self):
        # through the REAL wire encoding: 2-D uint8 must be packable
        from sglang.srt.afd.protocol import _payload_of

        reply = files_reply(self._checkpoint())
        for t in reply:
            _payload_of(t)
        papers = papers_from_reply(reply)
        self.assertEqual(sorted(papers), ["config.json", "tokenizer.json"])
        self.assertEqual(json.loads(papers["config.json"])["hidden_size"], 8)

    def test_a_pushed_name_cannot_pose_as_a_path(self):
        evil = torch.frombuffer(
            bytearray(json.dumps(["../evil"]).encode()), dtype=torch.uint8
        ).clone()
        with self.assertRaisesRegex(RuntimeError, "refused"):
            papers_from_reply((evil, torch.zeros(4, dtype=torch.uint8)))

    def test_the_config_is_built_in_memory(self):
        papers = papers_from_reply(files_reply(self._checkpoint()))
        model_files._PAPERS["fake:1"] = papers
        try:
            cfg = pool_config("pool://fake:1")
            self.assertEqual(cfg.hidden_size, 8)
        finally:
            model_files._PAPERS.pop("fake:1", None)


if __name__ == "__main__":
    unittest.main()
