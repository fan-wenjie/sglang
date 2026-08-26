"""The arm's startup checks run against a REAL ServerArgs, not a stand-in.

Every other test of these checks builds a namespace with exactly the fields it means to exercise.
That is right for asking what a check decides, and blind to the question of whether the fields
still exist: a setting deleted from ServerArgs leaves the stand-ins passing and the server dead at
startup with an AttributeError naming a flag nobody has any more.

That happened. `--afd-coverage` was deleted and `_check_coverage` went on reading it; the unit
suite stayed green and a two-machine run died before the first token.

So this constructs the real thing and runs the real entry points over it. It asserts nothing about
what they decide -- the other files do that -- only that every field they reach for is a field.
"""

from sglang.test.test_utils import CustomTestCase


def _real_args(**kw):
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(model_path="/nonexistent")
    for name, value in kw.items():
        assert hasattr(
            args, name
        ), f"ServerArgs has no {name}; this test is out of date"
        setattr(args, name, value)
    return args


class TestTheArmReadsFieldsThatExist(CustomTestCase):
    def test_the_argument_check_runs_on_a_real_server_args(self):
        from sglang.srt.afd_query_shift.arg_checks import check

        for shift in (None, 0, 1):
            with self.subTest(shift=shift):
                check(_real_args(afd_query_shift_layers=shift))

    def test_the_arm_decides_whether_it_is_wanted_from_a_real_server_args(self):
        from sglang.srt.afd.installer import (
            remote_embedding_wanted,
            span_cut_wanted_for,
        )

        args = _real_args(afd_query_shift_layers=None)
        self.assertFalse(span_cut_wanted_for(args))
        self.assertTrue(span_cut_wanted_for(_real_args(afd_query_shift_layers=0)))
        remote_embedding_wanted(args)

    def test_the_loader_hooks_run_on_a_real_server_args(self):
        """These decide what is allocated, before any model exists, and each takes server_args."""
        from sglang.srt.afd.arms import load, resolve

        load()
        arm = resolve("span")()
        args = _real_args(afd_query_shift_layers=1)
        arm.absent_classes(args)
        arm.kept_parameters(args)
        word = arm.arrangement_word(args)
        arm.explain_arrangement(args, word, word)
