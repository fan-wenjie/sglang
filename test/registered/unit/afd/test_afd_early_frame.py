"""The early frame is cooked here, queued rather than waited for, and the lower rungs send nothing.

The properties are read from the SOURCE of `_send_early`, because each is invisible in any
output and each has already been got wrong once in this arrangement:

    it is sent            an installer gap left the issue call unreached in every configuration
                          this branch supported, so three separate shift-1-versus-shift-0
                          comparisons measured two builds with the mechanism inert on both
    nothing waits on it   the frame's deadline is the mix a feed-forward later; a wait here puts
                          the wire in front of the feed-forward instead of beside it
    the cook is here      the far end contracts and assembles, nothing else -- preparation on
                          the host is the schedule's max moving to the bottleneck card, priced
                          at 10.69 percentage points by the ladder
    shift 0 sends nothing shift 0 is standard AFD, and a frame on the wire is a difference
                          whatever the far end then does with it
"""

import ast
import pathlib

from sglang.test.test_utils import CustomTestCase

SRT = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"


def _method(path, name):
    for node in ast.walk(ast.parse((SRT / path).read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path} no longer defines {name}")


def _calls(node):
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Attribute):
                out.add(f.attr)
            elif isinstance(f, ast.Name):
                out.add(f.id)
    return out


class TheEarlySendIsWhatItSays(CustomTestCase):
    def setUp(self):
        self.send = _method("afd/span.py", "_send_early")

    def test_shift_zero_returns_before_any_work(self):
        src = ast.unparse(self.send)
        self.assertIn("self.query_shift == 0", src)

    def test_the_cook_happens_on_this_side(self):
        called = _calls(self.send)
        self.assertIn("cook_early_fused", called)
        self.assertIn("_early_materials", called)

    def test_it_is_sent_and_nothing_waits(self):
        called = _calls(self.send)
        self.assertIn("issue", called)
        # `.wait()` on the sent events would put the wire in front of the feed-forward. The
        # events are COLLECTED (for the old-op mix to order an inline send behind), not waited.
        self.assertNotIn("wait", called)

    def test_the_send_records_what_it_cooked(self):
        # `_linear_attention` chooses the ready op by this record; losing it would send raw
        # materials for a layer whose contraction the far end already holds
        src = ast.unparse(self.send)
        self.assertIn("early_cooked", src)


class TheHostHalfDoesNotCook(CustomTestCase):
    """THE RULE from the ladder, held statically: the host handlers contract and assemble.

    Any convolution, normalisation or coefficient in `early_contraction.py` is preparation on
    the card the schedule cannot afford it on. The runtime tripwire in `span_routing` catches
    the consequence (the max flipping to the host); this catches the cause, in review.
    """

    FORBIDDEN = (
        "convolve_with_ring",
        "convolve_partial",
        "normalise",
        "query_coefficient",
    )

    def test_no_preparation_in_the_host_handlers(self):
        text = (SRT / "afd_query_shift" / "early_contraction.py").read_text()
        tree = ast.parse(text)
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                called.add(
                    f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                )
        for name in self.FORBIDDEN:
            self.assertNotIn(
                name,
                called,
                f"the host's handlers call {name}; preparation belongs on the pool -- see "
                f"pool_cook's docstring for the ladder that priced this",
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
