"""The order inside a span's window, which is the whole schedule and is invisible in any output.

The pool answers a span in two halves: the query's source arrives one feed-forward before the
span's own output does. The host is supposed to spend that feed-forward sweeping its cache. Three
arrangements compute exactly the same tokens:

    collect q, sweep, collect k/v      the window. The pool finishes the span while this side reads
                                       its own cache
    collect q, collect k/v, sweep      nothing hidden. This is the synchronous arrangement with an
                                       extra reply
    collect k/v, collect q, sweep      the same, and it also throws away the reason the reply has
                                       two halves at all

Only the ORDER separates them, so the order is asserted here rather than inferred from a duration.
A benchmark can only report a smaller number, and a smaller number has many other explanations --
a warmer cache, a quieter link, a different batch. `test_afd_sweep_window.py` makes the same
argument for the per-layer cut, where the gap sits between issuing a feed-forward and collecting
it; this is the group cut's own gap, which is a different mechanism for the same idea.

The two leaf computations are stubbed. What is under test is `head`, and the sweep's POSITION in
it -- not what a sweep computes, which `test_afd_split_exactness.py` covers to 6e-16.
"""

import ast
import inspect
import textwrap
import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.span import group_layers
from sglang.srt.afd.span_routing import SpanRouting
from sglang.test.test_utils import CustomTestCase

TYPES = (
    ["linear_attention"] * 3
    + ["full_attention"]
    + ["linear_attention"] * 3
    + ["full_attention"]
)
H = 4


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = (
            object()
        )  # the head keeps its attention; the fake never calls into it

    def forward(self, positions, hidden_states, residual=None, **kwargs):
        return hidden_states, residual


class Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer() for _ in TYPES])


class Recorder:
    """A span client that writes down WHEN each half was collected."""

    def __init__(self, order):
        self.order = order
        # the raw pool client inside a SpanClient, which `forget_starting_requests` reaches for.
        # Never called here: this batch is a decode, `extend_prefix_lens_cpu` is absent, and
        # nothing is beginning
        self.client = types.SimpleNamespace()

    def issue(self, layer, o, row_ids, positions, op=None, windows=None):
        self.order.append(("issue", layer))
        return ("handle", layer)

    def kv_already_arrived(self, handle) -> bool:
        # the fixture's pool is never the loser: the k|v half is collected, not found waiting,
        # so the schedule's max stays with the pool and the tripwire stays quiet
        return False

    def collect_read_point(self, handle, device):
        self.order.append(("collect_q", handle[1]))
        return torch.zeros(1, H)

    def collect_kv(self, handle, device):
        self.order.append(("collect_kv", handle[1]))
        return torch.zeros(1, H), torch.zeros(1, H)

    def collect_output(self, handle, device):
        self.order.append(("collect_out", handle[1]))
        return (torch.zeros(1, H),)


def a_routing(order):
    """A routing with the heads installed and the two leaf computations recorded."""
    routing = SpanRouting.__new__(SpanRouting)
    routing.model = Stack()
    routing.client = Recorder(order)
    routing._undo = []
    routing._outstanding = None
    routing._pool_logits = None
    routing._head_on_pool = False
    routing._vocab = None
    routing._returned = None
    routing._rows = None
    routing.history = None
    routing.sweeps = 0
    routing.pool_was_the_max = 0
    routing.host_was_the_max = 0
    # shift 0: the fixture's spans attach no convolution windows
    routing.query_shift = 0
    routing.types = []
    routing._span_by_head = {}
    routing.fused = 0
    routing.fused_decodes = 0
    routing.head_calls = 0
    routing.report_every = 200
    routing.pending_ids = None
    routing.window_s = 0.0
    routing.windows = 0
    routing.spans = group_layers(TYPES)
    routing.heads = {s[0] for s in routing.spans if s[0] >= 0}
    routing._first_head = min(routing.heads)
    routing._index = None

    def sweep(attn, forward_batch, q):
        order.append(("sweep", None))
        routing.sweeps += 1
        return "state"

    def join(attn, forward_batch, k, v, state, q):
        order.append(("join", None))
        return torch.zeros(1, H)

    routing._sweep = sweep
    routing._join = join
    routing._check_untouched = lambda layer_id, hidden: None
    return routing


def a_batch():
    return types.SimpleNamespace(
        req_pool_indices=torch.tensor([3]),
        extend_seq_lens=None,
        extend_prefix_lens_cpu=None,  # a decode step begins no request
    )


class TestTheSweepSitsBetweenTheTwoHalves(CustomTestCase):
    def setUp(self):
        self.order = []
        self.routing = a_routing(self.order)
        heads = sorted(self.routing.heads)
        self.first = heads[0]
        layers = self.routing.model.model.layers
        self.routing._make_head(
            layers[self.first], self.first, opens=True, closes=False
        )

    def _run(self):
        layer = self.routing.model.model.layers[self.first]
        layer.forward(
            torch.zeros(1, 3), torch.zeros(1, H), None, forward_batch=a_batch()
        )

    def test_the_order_is_issue_collect_q_sweep_collect_kv_join(self):
        """The window, named step by step. Any other order computes the same tokens."""
        self._run()
        self.assertEqual(
            [name for name, _ in self.order],
            ["issue", "collect_q", "sweep", "collect_kv", "join", "issue"],
        )

    def test_the_sweep_is_before_the_key_and_value_arrive(self):
        """Stated separately because it is the property, and the list above is one spelling of it.
        If the sweep moved after `collect_kv`, the host would wait out the whole span and then
        read its cache -- correct, and with nothing overlapped."""
        self._run()
        names = [name for name, _ in self.order]
        self.assertLess(names.index("sweep"), names.index("collect_kv"))

    def test_the_sweep_is_after_the_query_arrives(self):
        """The other side of the same window: the sweep needs the query the pool sends early, so
        it cannot start before that half lands."""
        self._run()
        names = [name for name, _ in self.order]
        self.assertGreater(names.index("sweep"), names.index("collect_q"))

    def test_the_sweep_is_counted_so_the_report_cannot_assert_it(self):
        """`sweep_window_open` was a hardcoded True, then a hardcoded False. Neither could tell a
        window that stopped opening from one that never shut. It reads `sweeps` now, so this is
        what makes the report mean something."""
        self.assertEqual(self.routing.sweeps, 0)
        self._run()
        self.assertEqual(self.routing.sweeps, 1)


if __name__ == "__main__":
    unittest.main()


class TestTheFixtureDoesNotDriftFromTheConstructor(CustomTestCase):
    """`a_routing` builds a `SpanRouting` with `__new__` and lists its fields by hand.

    It has to: `__init__` ends in `_install()`, which rewrites a real model's layers, and these
    tests are about the ORDER of four calls rather than about a stack. The cost is that the list
    is a copy of the constructor kept up to date by nobody.

    Twice in one afternoon a field was added to `__init__` and not here -- `pending_ids` when the
    embedding moved to the pool, then `window_s`/`windows` when the window's width was counted --
    and both times the symptom was an `AttributeError` raised deep inside the head's forward,
    naming neither the fixture nor the field. Nothing pointed at the copy.

    So the copy is checked against the original rather than trusted. A field the fixture
    deliberately does not want is named in `NOT_NEEDED` with its reason, which makes leaving one
    out a decision instead of an omission.
    """

    NOT_NEEDED = {
        "passengers": (
            "these tests install a HEAD layer and drive its forward. A passenger's forward is a "
            "pass-through and opens no window, so the set is never read on this path."
        ),
    }

    def test_every_field_the_constructor_sets_is_in_the_fixture(self):
        routing = a_routing([])
        source = textwrap.dedent(inspect.getsource(SpanRouting.__init__))
        assigned = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    assigned.add(target.attr)
        missing = sorted(
            name
            for name in assigned
            if name not in self.NOT_NEEDED and not hasattr(routing, name)
        )
        self.assertEqual(
            missing,
            [],
            f"`a_routing` is missing {missing}, which `SpanRouting.__init__` sets. Add them to "
            f"the fixture, or name them in NOT_NEEDED with the reason they are not wanted.",
        )
