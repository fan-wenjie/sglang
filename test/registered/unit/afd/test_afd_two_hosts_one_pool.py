"""Two hosts' row ids must not collide in the pool's tables.

Both hosts' schedulers hand out row ids from similar counters, so the same integer arrives on
two sockets meaning two different requests. Every pool-side table keyed by a bare id would let
one host's request continue from another's residual -- fluently, which is the failure mode this
whole arrangement is built to refuse by name. The namespace is per socket, assigned at first
sight, and never travels: callbacks and replies are built per rider from the rider's own
frames.
"""

import socket
import unittest

from sglang.srt.afd.pool_server import namespace_of
from sglang.test.test_utils import CustomTestCase


class TestTheNamespaceKeepsHostsApart(CustomTestCase):
    def test_same_socket_same_namespace(self):
        a, b = socket.socketpair()
        try:
            self.assertEqual(namespace_of(a), namespace_of(a))
        finally:
            a.close()
            b.close()

    def test_two_sockets_two_namespaces(self):
        a, b = socket.socketpair()
        try:
            self.assertNotEqual(namespace_of(a), namespace_of(b))
        finally:
            a.close()
            b.close()

    def test_the_same_bare_id_lands_in_two_keys(self):
        a, b = socket.socketpair()
        try:
            row = 12345
            self.assertNotEqual(namespace_of(a) | row, namespace_of(b) | row)
            # and the bare id survives inside its namespace, so nothing downstream that
            # formats or compares ids within one host's stream changes meaning
            self.assertEqual((namespace_of(a) | row) & ((1 << 40) - 1), row)
        finally:
            a.close()
            b.close()


if __name__ == "__main__":
    unittest.main()
