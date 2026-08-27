"""One tensor kept out of a class that is otherwise built with no storage at all.

`absent_classes` names a whole class and the loader wraps the CLASS, not its instances, so every
`Qwen3_5GatedDeltaNet` is constructed inside `torch.device("meta")`. That is the point: the memory
is never taken rather than taken and released, and the construction peak is what fails on a card
smaller than the checkpoint.

Under `--afd-rings-on-host` the host holds the convolution ring the pool no longer does, and it
needs the one weight that filters it -- 3.75 MiB of an 11 GiB layer stack. Dropping the class from
the absent list to get it would cost the whole linear attention and end the small-card claim, so
the exemption is by parameter NAME within a class every instance of which is still built on meta.

The failure this guards is silent in the worst way. Without the exemption the weight stays on meta,
`_mix` raises `NotImplementedError: Cannot copy out of meta tensor` inside the client's receive
thread, the host answers nothing, and the pool reports a 30 s timeout naming the far end -- three
hops from the cause, with the server looking alive throughout. That is what it did.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.absent_ffn import BuildFeedForwardOnMeta
from sglang.test.test_utils import CustomTestCase


class Conv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(6, 4))
        self.bias = torch.nn.Parameter(torch.zeros(6))


class GatedDeltaNetLike(torch.nn.Module):
    """Stands in for the class an arm names whole: a small convolution and a large projection."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1d = Conv()
        self.in_proj_qkvz = torch.nn.Linear(64, 64)


CONVOLUTION = ("conv1d.weight", "conv1d.bias")


class TestAClassBuiltOnMetaCanStillKeepOneTensor(CustomTestCase):
    def test_without_the_exemption_everything_is_meta(self):
        """The control. If this ever fails, the class is not being built on meta at all and the
        case below proves nothing -- it would be passing on an ordinary construction."""
        with BuildFeedForwardOnMeta([GatedDeltaNetLike]):
            built = GatedDeltaNetLike()
        self.assertTrue(built.conv1d.weight.is_meta)
        self.assertTrue(built.in_proj_qkvz.weight.is_meta)

    def test_the_named_parameters_are_real_and_the_rest_are_not(self):
        with BuildFeedForwardOnMeta(
            [GatedDeltaNetLike], keep={GatedDeltaNetLike: CONVOLUTION}
        ) as ctx:
            built = GatedDeltaNetLike()

        self.assertFalse(
            built.conv1d.weight.is_meta, "the convolution weight stayed on meta"
        )
        self.assertFalse(
            built.conv1d.bias.is_meta, "the convolution bias stayed on meta"
        )
        self.assertTrue(
            built.in_proj_qkvz.weight.is_meta,
            "the projection was allocated too, so this bought the whole layer and not one tensor",
        )
        self.assertEqual(ctx.kept, 2)

    def test_the_kept_parameter_keeps_its_shape_and_dtype(self):
        """What the checkpoint lookup needs. sglang fills by name afterwards, and a parameter with
        the wrong shape is a load-time error while one with the wrong dtype is a silent cast.
        """
        with BuildFeedForwardOnMeta(
            [GatedDeltaNetLike], keep={GatedDeltaNetLike: CONVOLUTION}
        ):
            built = GatedDeltaNetLike()
        self.assertEqual(tuple(built.conv1d.weight.shape), (6, 4))
        self.assertEqual(built.conv1d.weight.dtype, torch.float32)

    def test_it_is_a_parameter_the_loader_will_find(self):
        """Restored as a Parameter, not a bare tensor: sglang looks the name up in
        `named_parameters()`, and a plain tensor is invisible there -- the weight would then be
        silently left at whatever `torch.empty` returned."""
        with BuildFeedForwardOnMeta(
            [GatedDeltaNetLike], keep={GatedDeltaNetLike: CONVOLUTION}
        ):
            built = GatedDeltaNetLike()
        found = dict(built.named_parameters())
        self.assertIn("conv1d.weight", found)
        self.assertFalse(found["conv1d.weight"].is_meta)

    def test_the_class_is_restored_afterwards(self):
        """The context patches __init__ on the class itself; leaving it patched would build every
        later instance on meta, including in another test."""
        with BuildFeedForwardOnMeta(
            [GatedDeltaNetLike], keep={GatedDeltaNetLike: CONVOLUTION}
        ):
            pass
        after = GatedDeltaNetLike()
        self.assertFalse(after.conv1d.weight.is_meta)
        self.assertFalse(after.in_proj_qkvz.weight.is_meta)


if __name__ == "__main__":
    unittest.main()
