"""A feed-forward that was never allocated must survive the loader, not be refused by it.

This file exists because of a failure no other test on this branch could see. Upstream added
`model_loader/post_load.py`, whose staging context raises on any meta tensor a module still
holds -- the right answer for a module whose load failed, and fatal for AFD, where a host's
feed-forward is deliberately built without storage because a pool computes it:

    RuntimeError: Cannot post-process meta tensor MergedColumnParallelLinear.weight

The host did not start. Every unit test was green while that was true, and stayed green, because
not one of them loads a model. The rebase that brought the check in replayed with no conflict.

So the guard is here, and it needs no GPU: `device_loading_context` decides before it touches a
device, so both branches are reachable on CPU with a target device that does not exist.
"""

import unittest

import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.model_loader.loader import device_loading_context
from sglang.test.test_utils import CustomTestCase


class _Absent(nn.Module):
    """A module whose weight was never given storage, the way `absent_ffn` builds one."""

    def __init__(self):
        super().__init__()
        with torch.device("meta"):
            self.weight = nn.Parameter(torch.empty(4, 4))


class TestAModuleBuiltWithoutStorageLoads(CustomTestCase):
    def test_an_unmarked_meta_module_is_still_refused(self):
        """The upstream check has to keep working: this is the failure it is FOR.

        Asserted first, and it is the half that makes the other half mean something. A fix that
        skipped staging for every meta module would pass the next case and silently swallow a
        module whose weights failed to arrive.
        """
        with self.assertRaises(RuntimeError) as caught:
            with device_loading_context(_Absent(), torch.device("cuda")):
                pass
        self.assertIn("meta tensor", str(caught.exception))

    def test_a_marked_module_passes_through(self):
        """`absent_ffn` marks what it builds, so the loader knows there is nothing to stage."""
        module = _Absent()
        for child in module.modules():
            child.afd_weights_absent = True
        with device_loading_context(module, torch.device("cuda")):
            pass

    def test_the_mark_reaches_the_children_the_loader_actually_stages(self):
        """The first fix marked the block and not its linears, and the host failed identically.

        The loader iterates `model.named_modules()` and stages each one carrying a `quant_method`
        -- inside a feed-forward those are the linears, not the block. A mark on the parent alone
        lands on the one module staging never looks at, so this case pins the descendants.
        """

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = _Absent()

        block = Block()
        block.afd_weights_absent = True  # the parent only, as the first attempt did

        with self.assertRaises(RuntimeError):
            with device_loading_context(block.inner, torch.device("cuda")):
                pass

        for child in block.modules():
            child.afd_weights_absent = True
        with device_loading_context(block.inner, torch.device("cuda")):
            pass


if __name__ == "__main__":
    unittest.main()
