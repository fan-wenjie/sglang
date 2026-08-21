"""The host under the group cut: which layers still run, and what refuses when one that shouldn't.

Every case here guards a failure that produces fluent output. That is the whole difficulty of this
arrangement -- a layer running in the wrong place, a reply taken under the wrong key, a row sent
under the wrong request id -- none of them raise, and none of them show up in the text. They show
up in a throughput number that gets quoted as this arrangement's.
"""

import socket
import threading
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.pool_client import PoolClient, PoolClosed
from sglang.srt.afd.pool_server import Departure
from sglang.srt.afd.protocol import (
    OP_SPAN,
    OP_SPAN_ENTER,
    OP_SPAN_EXIT,
    OP_SPAN_Q,
    Frame,
    decode,
    send_frame,
)
from sglang.srt.afd.span_routing import SpanClient, SpanRouting
from sglang.test.test_utils import CustomTestCase

TYPES = ["linear_attention"] * 3 + ["full_attention"] + \
        ["linear_attention"] * 3 + ["full_attention"]


class Recorder:
    """A pool client that records what was asked and answers with shaped noise."""

    def __init__(self, short_kv=False):
        self.issued = []
        self.collected = []
        # a pool one protocol version behind, answering a span with one tensor where the key and
        # value belong
        self.short_kv = short_kv

    def issue_frame(self, request_id, layer, tensors, op):
        self.issued.append({"request_id": request_id, "layer": layer, "op": op,
                            "tensors": tensors})
        return SimpleNamespace(request_id=request_id, layer=layer, op=op,
                               _replace=lambda **k: SimpleNamespace(
                                   request_id=request_id, layer=layer, **k))

    def collect_frame(self, handle, device):
        """Answer by opcode, as the pool does: one tensor for the query, two for the key/value.

        A stub that answered the same shape to both would let a caller that mixed the two halves
        up pass, and mixing them up is the failure the opcodes exist to prevent.
        """
        self.collected.append((handle.layer, handle.op))
        if handle.op == OP_SPAN_Q:
            return (torch.zeros(1, 4),)
        if self.short_kv:
            return (torch.zeros(1, 4),)
        return (torch.zeros(1, 4), torch.zeros(1, 4))


def a_client(short_kv=False):
    return SpanClient(Recorder(short_kv), reply_timeout_s=5.0)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, hidden_states, residual=None, *args, **kwargs):
        self.forward_calls += 1
        return hidden_states * 2, residual


class Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer() for _ in TYPES])


class TestThePassThroughs(CustomTestCase):
    """Three layers in four do not run here, and a layer that still does is caught.

    The install this mirrors went wrong once already in this tree, in the other direction: a
    construction hook patched one of sixteen loaders and nothing said so. What made it invisible
    was that a layer running where it should not still produces a number.
    """

    def setUp(self):
        self.stack = Stack()
        # only the pass-throughs are installed here; the heads need an attention this fake has not
        # got, and what is under test is which layers stop running
        self.routing = SpanRouting.__new__(SpanRouting)
        self.routing.model = self.stack
        self.routing._undo = []
        from sglang.srt.afd.span import group_layers

        self.routing.spans = group_layers(TYPES)
        for span in self.routing.spans:
            for layer_id in span[1:]:
                self.routing._make_pass_through(self.stack.model.layers[layer_id], layer_id)

    def test_a_passenger_layer_returns_its_input_untouched(self):
        x = torch.randn(2, 4)
        out, residual = self.stack.model.layers[0].forward(x, None)
        self.assertIs(out, x)
        self.assertIsNone(residual)
        self.assertEqual(self.stack.model.layers[0].forward_calls, 0)

    def test_the_head_layers_are_left_alone(self):
        x = torch.randn(2, 4)
        out, _ = self.stack.model.layers[3].forward(x, None)
        self.assertEqual(self.stack.model.layers[3].forward_calls, 1)
        torch.testing.assert_close(out, x * 2)

    def test_removing_the_install_puts_every_layer_back(self):
        self.routing.remove()
        x = torch.randn(2, 4)
        self.stack.model.layers[0].forward(x, None)
        self.assertEqual(self.stack.model.layers[0].forward_calls, 1)

    def test_a_layer_that_ran_in_between_is_refused(self):
        """The guard that says the install took, rather than assuming it."""
        self.routing._returned = torch.randn(1, 4)
        with self.assertRaises(RuntimeError) as caught:
            self.routing._check_untouched(7, torch.randn(1, 4))
        self.assertIn("pass-through install missed it", str(caught.exception))

    def test_the_same_tensor_passes_the_guard(self):
        held = torch.randn(1, 4)
        self.routing._returned = held
        self.routing._check_untouched(7, held)


class TestTheTwoHalvesAreTakenUnderDifferentKeys(CustomTestCase):
    """Both halves of a span's reply share a request and a group and are the same shape.

    Taken under one key the caller gets whichever landed first, and the query source and the span
    output are interchangeable as tensors -- so the attention would run against a hidden state
    from the wrong end of the span and nothing would say so. The opcode is what separates them.
    """

    def test_the_read_point_is_collected_under_its_own_opcode(self):
        client = a_client()
        handle = client.issue(3, torch.zeros(1, 4), torch.tensor([0]), torch.tensor([5]))
        client.collect_read_point(handle, "cpu")
        client.collect_kv(handle, "cpu")
        self.assertEqual(client.client.collected, [(3, OP_SPAN_Q), (3, OP_SPAN)])

    def test_the_order_is_read_point_first(self):
        """It is the half that arrives early; collecting it second discards the head start."""
        client = a_client()
        handle = client.issue(3, torch.zeros(1, 4), torch.tensor([0]), torch.tensor([5]))
        client.collect(handle, "cpu")
        self.assertEqual(client.client.collected[0][1], OP_SPAN_Q)

    def test_a_reply_that_is_not_a_key_and_a_value_is_refused(self):
        """Unpacking anyway would append something that is not a key to the cache."""
        client = a_client(short_kv=True)
        handle = client.issue(3, torch.zeros(1, 4), torch.tensor([0]), torch.tensor([5]))
        with self.assertRaises(RuntimeError) as caught:
            client.collect_kv(handle, "cpu")
        self.assertIn("key and value", str(caught.exception))


class TestTheRowIdsTravel(CustomTestCase):
    """The pool keys both recurrent states by request; a row with no id advances the wrong one."""

    def test_a_row_count_mismatch_is_refused(self):
        client = a_client()
        with self.assertRaises(ValueError) as caught:
            client.issue(3, torch.zeros(4, 8), torch.tensor([0, 1]), torch.tensor([5]))
        self.assertIn("row id", str(caught.exception))

    def test_the_positions_ride_with_the_ids(self):
        """The pool rotates the key and the query by them; a span without them attends nowhere."""
        client = a_client()
        client.issue(3, torch.zeros(2, 8), torch.tensor([5, 9]), torch.tensor([40, 41]))
        pos = client.client.issued[0]["tensors"][2]
        self.assertEqual(tuple(pos.shape), (2, 1))
        self.assertEqual(pos.dtype, torch.int64)
        self.assertEqual(pos.reshape(-1).tolist(), [40, 41])

    def test_the_ids_ride_as_a_column_of_int64(self):
        client = a_client()
        client.issue(3, torch.zeros(2, 8), torch.tensor([5, 9]), torch.tensor([1, 2]))
        ids = client.client.issued[0]["tensors"][1]
        self.assertEqual(tuple(ids.shape), (2, 1))
        self.assertEqual(ids.dtype, torch.int64)
        self.assertEqual(ids.reshape(-1).tolist(), [5, 9])

    def test_every_call_gets_a_fresh_request_id(self):
        """Frames are keyed by (request, layer, op); a reused id crosses two answers."""
        client = a_client()
        for _ in range(3):
            client.issue(3, torch.zeros(1, 4), torch.tensor([0]), torch.tensor([5]))
        ids = [c["request_id"] for c in client.client.issued]
        self.assertEqual(len(set(ids)), 3)


class TestTheOpcodesNameTheThreeShapes(CustomTestCase):
    """The ends of the stack are different arrangements, not special cases of the middle one."""

    def test_they_are_distinct(self):
        self.assertEqual(len({OP_SPAN, OP_SPAN_Q, OP_SPAN_ENTER, OP_SPAN_EXIT}), 4)

    def test_the_opcode_travels_with_the_frame(self):
        client = a_client()
        client.issue(0, torch.zeros(1, 4), torch.tensor([0]), torch.tensor([0]), OP_SPAN_ENTER)
        self.assertEqual(client.client.issued[0]["op"], OP_SPAN_ENTER)


class Pool:
    """A pool that serves spans, or does not, answering HELLO and nothing else."""

    def __init__(self, span=None):
        self.departure = Departure(lambda b, l: b, 1, 0.005, "cpu")
        self.departure.span = span
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        try:
            while True:
                frame = decode(conn)
                if frame is None:
                    return
                if not self.departure.answer_directly(frame, conn):
                    send_frame(conn, Frame.one(frame.request_id, frame.layer, frame.tensor))
        except OSError:
            return

    def close(self):
        self.sock.close()


class TestBothEndsHaveToAgreeOnTheCut(CustomTestCase):
    """--afd-span-cut changes what each END holds, not only what they say to each other.

    A host under the group cut has no feed-forward weights and no recurrent state. Pointed at a
    per-layer pool it would not fail with a bad frame -- it would ask for spans nobody serves,
    which arrives as "closed mid-call", a message about a socket that names neither side's
    configuration. That is the failure this capability bit exists to replace, and it is the same
    failure the other three bits were added for.
    """

    def test_a_per_layer_pool_does_not_claim_spans(self):
        pool = Pool(span=None)
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)
        try:
            with self.assertRaises(PoolClosed) as caught:
                client.require(PoolClient.NEEDS_SPANS)
            self.assertIn("spans", str(caught.exception))
        finally:
            pool.close()

    def test_a_span_pool_satisfies_a_span_host(self):
        pool = Pool(span=object())
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)
        try:
            served = client.require(PoolClient.NEEDS_SPANS)
            self.assertEqual(served & PoolClient.NEEDS_SPANS, PoolClient.NEEDS_SPANS)
        finally:
            pool.close()

    def test_a_span_pool_still_serves_feed_forwards(self):
        """The bit is added to the others, not swapped for them: the weights are still here."""
        pool = Pool(span=object())
        client = PoolClient(f"127.0.0.1:{pool.port}", 5.0)
        try:
            client.require(PoolClient.NEEDS_FEED_FORWARD | PoolClient.NEEDS_SPANS)
        finally:
            pool.close()


class TestOneDepartureServesOneJob(CustomTestCase):
    """Riders queue by layer, so a feed-forward frame and a span frame can share a queue.

    They are different jobs with different replies. Dispatching on the first rider would answer
    one caller with the other's arithmetic, and both replies are tensors of plausible shape.
    """

    def test_a_mixed_queue_is_refused(self):
        departure = Departure(lambda b, l: b, 1, 0.005, "cpu")
        riding = [(Frame.one(1, 3, torch.zeros(1, 4), OP_SPAN), None),
                  (Frame.one(2, 3, torch.zeros(1, 4)), None)]
        with self.assertRaises(RuntimeError) as caught:
            departure._depart(3, riding)
        self.assertIn("mixes", str(caught.exception))

    def test_a_span_frame_at_a_pool_without_a_runner_is_refused_by_name(self):
        departure = Departure(lambda b, l: b, 1, 0.005, "cpu")
        riding = [(Frame(1, 3, (torch.zeros(1, 4), torch.zeros(1, 1)), OP_SPAN), None)]
        with self.assertRaises(RuntimeError) as caught:
            departure._depart(3, riding)
        self.assertIn("no span runner", str(caught.exception))


if __name__ == "__main__":
    unittest.main()


class TestRowIdsAreOneARowNotOneARequest(CustomTestCase):
    """A prefill of 122 tokens from one request is 122 rows and ONE request index.

    Sending that pair is how the deployment failed at its first token, twice: the row ids and the
    rows disagree, and the check that caught it is the only thing between that and 122 tokens all
    advancing slot zero's history. In decode the two coincide, which is why it survived every
    decode-shaped test.
    """

    def test_decode_is_already_one_a_row(self):
        got = SpanRouting._row_ids(
            SimpleNamespace(req_pool_indices=torch.tensor([3, 7]), extend_seq_lens=None))
        self.assertEqual(got.tolist(), [3, 7])

    def test_prefill_repeats_each_request_by_its_extend_length(self):
        got = SpanRouting._row_ids(SimpleNamespace(
            req_pool_indices=torch.tensor([3, 7]), extend_seq_lens=torch.tensor([4, 2])))
        self.assertEqual(got.tolist(), [3, 3, 3, 3, 7, 7])

    def test_a_batch_that_disagrees_with_itself_is_refused(self):
        with self.assertRaises(RuntimeError) as caught:
            SpanRouting._row_ids(SimpleNamespace(
                req_pool_indices=torch.tensor([3, 7]), extend_seq_lens=torch.tensor([4])))
        self.assertIn("disagrees with itself", str(caught.exception))
