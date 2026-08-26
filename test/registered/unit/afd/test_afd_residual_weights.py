"""Every weight byte a host multiplies by has one source: the pool.

The host's loader runs dummy and the residual tensors (convolution filters, the
final norm) arrive by OP_WEIGHTS in a fixed order both ends derive from the same
config -- the order IS the schema, pinned here so it cannot drift apart silently.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types

import torch

from sglang.test.test_utils import CustomTestCase


def _model(layers=4, taps=3, ch=6):
    def layer(kind):
        ly = types.SimpleNamespace()
        if kind != "full_attention":
            conv = types.SimpleNamespace(
                weight=torch.arange(ch * taps, dtype=torch.float32).reshape(
                    ch, 1, taps
                ),
                bias=torch.ones(ch),
            )
            ly.linear_attn = types.SimpleNamespace(conv1d=conv)
        return ly

    kinds = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
    m = types.SimpleNamespace(
        model=types.SimpleNamespace(
            layers=[layer(k) for k in kinds],
            norm=types.SimpleNamespace(weight=torch.full((8,), 2.0)),
        ),
        config=types.SimpleNamespace(layer_types=kinds),
    )
    return m, kinds


class TestTheOrderIsTheSchema(CustomTestCase):
    def test_the_pool_lists_filters_then_biases_then_the_norm(self):
        from sglang.srt.afd import pool_span

        model, kinds = _model()
        sent = {}

        def catch(sock, frame):
            sent["tensors"] = frame.tensors

        departure = types.SimpleNamespace(
            runner=types.SimpleNamespace(model=model),
            _wire_lock=__import__("threading").Lock(),
        )
        frame = types.SimpleNamespace(request_id=7, tensors=(torch.zeros(1, 1),))
        import unittest.mock as mock

        with mock.patch.object(pool_span, "send_frame", catch), mock.patch.object(
            pool_span, "layer_types_of", lambda m: kinds
        ):
            pool_span._depart_weights(departure, 0, 0, [(frame, object())])
        got = sent["tensors"]
        # 2 linear layers -> filter, bias, filter, bias, then the norm;
        # everything rides as one (1, n) row -- the host rebuilds the shapes
        self.assertEqual(len(got), 5)
        self.assertEqual(tuple(got[0].shape), (1, 18))
        self.assertEqual(tuple(got[1].shape), (1, 6))
        self.assertTrue(torch.equal(got[4], torch.full((1, 8), 2.0)))


if __name__ == "__main__":
    unittest.main()
