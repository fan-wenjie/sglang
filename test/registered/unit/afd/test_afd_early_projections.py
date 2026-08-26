"""The early view's projections, by serving method: slice, apply-and-slice, refuse.

The early view wants a PREFIX of a fused projection's outputs. Unquantised,
that is a smaller matmul against the weight's leading rows; quantised, the
weight is not the matrix, so the entry runs the method's own `apply` on the
whole projection and slices the OUTPUT -- always legitimate. Pinned here:
the two entries agree wherever both are defined, and a method the tables do
not name is refused rather than guessed.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types

import torch
import torch.nn as nn

from sglang.test.test_utils import CustomTestCase

from sglang.srt.afd.span import _project_prefix, _project_suffix


class _ApplyByLinear:
    """A stand-in quantised method whose apply IS the plain projection."""

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight)


class _NamedFp8(_ApplyByLinear):
    pass


_NamedFp8.__name__ = "Fp8LinearMethod"


def _proj(method=None):
    linear = nn.Module()
    linear.weight = nn.Parameter(torch.randn(12, 6), requires_grad=False)
    if method is not None:
        linear.quant_method = method
    return linear


class TestTheEarlyProjections(CustomTestCase):
    def test_the_two_entries_agree(self):
        x = torch.randn(3, 6)
        plain = _proj()
        quant = _proj(_NamedFp8())
        quant.weight = plain.weight
        torch.testing.assert_close(
            _project_prefix(plain, x, 5, "the rest"),
            _project_prefix(quant, x, 5, "the rest"),
        )
        torch.testing.assert_close(
            _project_suffix(plain, x, 5, "the front"),
            _project_suffix(quant, x, 5, "the front"),
        )

    def test_an_unnamed_method_is_refused(self):
        class MysteryMethod(_ApplyByLinear):
            pass

        with self.assertRaisesRegex(NotImplementedError, "MysteryMethod"):
            _project_prefix(_proj(MysteryMethod()), torch.randn(2, 6), 4, "v")


if __name__ == "__main__":
    unittest.main()
