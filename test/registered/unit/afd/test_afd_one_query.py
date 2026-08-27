"""The operator has one query, and `s` takes the state's key.

    S_t = alpha S_(t-1) P(k) + beta v k^T          P(k) = I - beta k k^T
    o_t = S_t q = alpha S_(t-1) P(k) q + beta (k.q) v = alpha (S q~) + s v

Two things follow, and the implementation had neither.

`beta (k.q) v` and `beta v k^T` are the same term of the same product, so the key in `s` is the
one that advances the state -- this step's. Not the early key that may enter the output's copy of
`P(k)`.

And there is ONE q. Under this arrangement it is the early one, so the current projection has no
use for a query at all; it produced one anyway and `s` was computed from it, which is neither of
the pre-registered arms nor the operator.

These pin both halves: that `core_from_mixed` given a query uses it and the current key, and that
given none it computes exactly what it always did.
"""

import torch

from sglang.srt.afd.linear_history import (
    core_from_mixed,
    expand_to_value_heads,
    normalise,
    query_coefficient,
)
from sglang.test.test_utils import CustomTestCase

KEY_HEADS, VALUE_HEADS, HEAD_K, HEAD_V = 4, 8, 16, 16
ROWS = 3


def _inputs(seed=0):
    torch.manual_seed(seed)
    width = KEY_HEADS * HEAD_K
    mixed = torch.randn(ROWS, 2 * width + VALUE_HEADS * HEAD_V)
    alpha = torch.rand(ROWS, VALUE_HEADS)
    beta = torch.rand(ROWS, VALUE_HEADS)
    return mixed, alpha, beta


def _call(mixed, alpha, beta, *, query, reading):
    return core_from_mixed(
        mixed,
        alpha=alpha,
        beta=beta,
        key_heads=KEY_HEADS,
        value_heads=VALUE_HEADS,
        head_k_dim=HEAD_K,
        head_v_dim=HEAD_V,
        read_state=(lambda _q: reading) if reading is not None else (lambda q: q * 0.0),
        query=query,
    )


class TestWhereSComesFrom(CustomTestCase):
    def test_without_a_query_it_is_what_it_always_was(self):
        """The no-handler path is standard AFD's arithmetic and must not have moved."""
        mixed, alpha, beta = _inputs()
        width = KEY_HEADS * HEAD_K
        q = mixed[:, :width].reshape(ROWS, KEY_HEADS, HEAD_K)
        k = mixed[:, width : 2 * width].reshape(ROWS, KEY_HEADS, HEAD_K)
        v = mixed[:, 2 * width :].reshape(ROWS, VALUE_HEADS, HEAD_V)
        q, k = normalise(q, k, scale=HEAD_K**-0.5)
        q = expand_to_value_heads(q, VALUE_HEADS)
        k = expand_to_value_heads(k, VALUE_HEADS)
        q_tilde, s = query_coefficient(q, k, beta)
        want = alpha.unsqueeze(-1) * (q_tilde * 0.0) + s.unsqueeze(-1) * v.float()

        core, _ = _call(mixed, alpha, beta, query=None, reading=None)
        self.assertTrue(
            torch.allclose(core, want.reshape(ROWS, -1), atol=1e-6),
            "the path with no query given no longer computes what it did before",
        )

    def test_with_a_query_s_takes_that_query_and_this_steps_key(self):
        mixed, alpha, beta = _inputs(1)
        width = KEY_HEADS * HEAD_K
        k = mixed[:, width : 2 * width].reshape(ROWS, KEY_HEADS, HEAD_K)
        v = mixed[:, 2 * width :].reshape(ROWS, VALUE_HEADS, HEAD_V)
        _q, k = normalise(
            mixed[:, :width].reshape(ROWS, KEY_HEADS, HEAD_K), k, scale=HEAD_K**-0.5
        )
        k = expand_to_value_heads(k, VALUE_HEADS)

        query = torch.randn(ROWS, VALUE_HEADS, HEAD_K)
        reading = torch.randn(ROWS, VALUE_HEADS, HEAD_V)
        want = (
            alpha.unsqueeze(-1) * reading
            + (beta * (k * query).sum(-1)).unsqueeze(-1) * v.float()
        )

        core, _ = _call(mixed, alpha, beta, query=query, reading=reading)
        self.assertTrue(
            torch.allclose(core, want.reshape(ROWS, -1), atol=1e-6),
            "`s` is not beta (k.q) with THIS step's key and the query it was given",
        )

    def test_the_given_query_actually_changes_the_answer(self):
        """A guard that cannot fail is worse than none: if the query were ignored, the case above
        would still pass whenever the ignored value happened to be close."""
        mixed, alpha, beta = _inputs(2)
        reading = torch.randn(ROWS, VALUE_HEADS, HEAD_V)
        a, _ = _call(
            mixed,
            alpha,
            beta,
            query=torch.randn(ROWS, VALUE_HEADS, HEAD_K),
            reading=reading,
        )
        b, _ = _call(
            mixed,
            alpha,
            beta,
            query=torch.randn(ROWS, VALUE_HEADS, HEAD_K),
            reading=reading,
        )
        self.assertFalse(
            torch.allclose(a, b),
            "two different queries produced the same core, so the query is being ignored",
        )
