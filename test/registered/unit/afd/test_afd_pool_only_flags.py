"""The pool owns the arrangement's flags; a host that sets one is refused at startup.

A host configures nothing about the arrangement and loads no configuration file: it
adopts what the pool pushes at the HELLO. A host-side arrangement flag could only agree
with the pushed value (redundant) or disagree (a second source of truth), so it is
refused by name before a model loads rather than adopted when it happens to match.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase


def _args(**kw):
    base = dict(
        afd_mode="host",
        afd_pool_addr="127.0.0.1:8999",
        afd_bootstrap_port=8999,
        afd_min_batch=2,
        afd_max_wait_ms=5,
        afd_transfer_backend=None,
        afd_query_shift_layers=None,
        model_path="/nonexistent",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _check(args):
    from sglang.srt.arg_groups.afd_hook import _refuse_pool_only_on_host

    _refuse_pool_only_on_host(args)


class TestTheHostSetsNoArrangementFlag(CustomTestCase):
    def test_a_clean_host_passes(self):
        _check(_args())

    def test_each_pool_flag_is_refused_on_a_host_by_name(self):
        for name, value in (
            ("afd_min_batch", 1),
            ("afd_max_wait_ms", 7),
            ("afd_transfer_backend", "nccl"),
            ("afd_bootstrap_port", 9001),
        ):
            with self.assertRaisesRegex(ValueError, name.replace("_", "-"), msg=name):
                _check(_args(**{name: value}))

    def test_the_same_flags_are_fine_on_the_pool(self):
        _check(_args(afd_mode="pool", afd_min_batch=1, afd_query_shift_layers=1))


if __name__ == "__main__":
    unittest.main()
