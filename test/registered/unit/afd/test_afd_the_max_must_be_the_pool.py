"""The schedule's max must be the pool, and a host-shaped max must warn, not pass.

The arrangement prices itself on e = max(host arm, pool arm) resolving to the pool's
feed-forward. Work moved onto the host flipped that once, silently, and cost 10.7 percentage
points before a ladder found it. The tripwire that exists because of that must (a) see a reply
that pre-arrived without consuming it, and (b) warn on a sustained majority of host-won windows
while staying quiet under jitter-level ones.
"""

import threading
import time
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.pool_client import Handle, PoolClient
from sglang.srt.afd.span_routing import SpanRouting


def _client_with_reply_table() -> PoolClient:
    client = PoolClient.__new__(PoolClient)
    client._cond = threading.Condition()
    client._replies = {}
    return client


class TheProbeDoesNotTake(unittest.TestCase):
    def test_waiting_then_still_waiting(self):
        client = _client_with_reply_table()
        handle = Handle(request_id=3, layer=7, issued_at=time.perf_counter(), op=11)
        self.assertFalse(client.reply_waiting(handle))
        client._replies[handle.reply_key] = (torch.zeros(1),)
        self.assertTrue(client.reply_waiting(handle))
        # the probe must not have consumed it: a collect after a probe still finds the reply
        self.assertTrue(client.reply_waiting(handle))
        self.assertIn(handle.reply_key, client._replies)

    def test_another_op_is_another_reply(self):
        client = _client_with_reply_table()
        asked = Handle(request_id=3, layer=7, issued_at=0.0, op=11)
        client._replies[asked._replace(op=12).reply_key] = (torch.zeros(1),)
        self.assertFalse(client.reply_waiting(asked))


class AHostShapedMaxWarns(unittest.TestCase):
    def _routing(self, *, host_won: int, pool_won: int) -> SpanRouting:
        routing = SpanRouting.__new__(SpanRouting)
        routing.host_was_the_max = host_won
        routing.pool_was_the_max = pool_won
        return routing

    def test_a_sustained_host_max_warns(self):
        routing = self._routing(host_won=300, pool_won=212)
        with self.assertLogs("sglang.srt.afd.span_routing", "WARNING") as caught:
            routing._refuse_a_host_shaped_max()
        self.assertIn("HOST is the schedule's max", caught.output[0])

    def test_jitter_stays_quiet(self):
        routing = self._routing(host_won=40, pool_won=472)
        with self.assertNoLogs("sglang.srt.afd.span_routing", "WARNING"):
            routing._refuse_a_host_shaped_max()

    def test_mid_period_stays_quiet_even_at_a_majority(self):
        # the rule fires on period boundaries so one warning covers one period, not every window
        routing = self._routing(host_won=300, pool_won=211)
        with self.assertNoLogs("sglang.srt.afd.span_routing", "WARNING"):
            routing._refuse_a_host_shaped_max()


if __name__ == "__main__":
    unittest.main()
