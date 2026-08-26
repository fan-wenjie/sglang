"""Relocating the preparation must keep the arithmetic within its declared regime.

The cooked protocol moves the convolution, the normalisation and the coefficient from the host
to the pool. A move that changes the arithmetic is not a move -- the accuracy figures were
measured against the served arrangement, and they carry over exactly as far as the arithmetic
does. So each relocated piece is run both ways on the same inputs and compared exactly:

    convolve_partial      against `convolve_with_ring`'s one-row read-only path
    cook_early            against the host sequence it replaced, spelled inline
    mix_ready's assembly  against `core_from_mixed`'s external-query branch
    the ring advance      against the write `convolve_with_ring` performs

The regime is allclose since the partial-sum change: the history's taps are summed by the
ring's owner and the newest is added by the cook, a deliberate reassociation shared with the
fused kernel. The ring-advance comparison stays exact -- nothing there reassociates.
"""

import torch

from sglang.srt.afd.linear_history import (
    convolve_with_ring,
    core_from_mixed,
    expand_to_value_heads,
    normalise,
    query_coefficient,
)
from sglang.srt.afd_query_shift.pool_cook import (
    convolve_partial,
    cook_early,
    cook_mix,
)
from sglang.test.test_utils import CustomTestCase

KEY_HEADS, VALUE_HEADS, HEAD_K, HEAD_V = 4, 8, 16, 16
KEY_DIM = KEY_HEADS * HEAD_K
WIDTH = 2 * KEY_DIM + VALUE_HEADS * HEAD_V
TAPS = 4


def _pieces(rows=3, seed=11):
    torch.manual_seed(seed)
    ring = torch.randn(rows + 2, WIDTH, TAPS, dtype=torch.bfloat16) * 0.1
    x = torch.randn(rows, WIDTH, dtype=torch.bfloat16)
    weight = torch.randn(WIDTH, TAPS) * 0.1
    slots = list(range(rows))
    runs = [(100 + i, i, 1) for i in range(rows)]
    return ring, x, weight, slots, runs


def _partial(ring, slots, weight):
    window = ring[torch.tensor(slots)][..., 1:]
    return (window * weight[:, :-1]).sum(-1)


class ThePartialConvolutionIsTheRings(CustomTestCase):
    def test_finished_partial_agrees(self):
        ring, x, weight, slots, runs = _pieces()
        with_ring = convolve_with_ring(
            ring.clone(), x, weight, slots=slots, runs=runs, write=False
        )
        finished = convolve_partial(_partial(ring, slots, weight), x, weight[:, -1])
        # the history's sum is reassociated (three taps then the newest, against all four
        # together), so this is allclose, not bits -- the same regime as the fused cook
        torch.testing.assert_close(finished, with_ring, rtol=2e-2, atol=2e-2)

    def test_the_advance_is_the_same_write(self):
        ring, x, weight, slots, runs = _pieces()
        advanced = ring.clone()
        convolve_with_ring(advanced, x, weight, slots=slots, runs=runs)
        # mix_ready's write, spelled as the handler spells it
        manual = ring.clone()
        rows = torch.tensor(slots, dtype=torch.int64)
        held = manual.index_select(0, rows)
        manual.index_copy_(0, rows, torch.cat([held[..., 1:], x.unsqueeze(-1)], dim=-1))
        torch.testing.assert_close(manual, advanced, rtol=0, atol=0)


class TheCookIsTheHostsOldSequence(CustomTestCase):
    def test_cook_early(self):
        ring, x, weight, slots, runs = _pieces()
        qk = x[:, : 2 * KEY_DIM]
        beta = torch.rand(x.shape[0], VALUE_HEADS)
        got_tilde, got_q = cook_early(
            qk,
            beta,
            _partial(ring, slots, weight)[:, : 2 * KEY_DIM],
            weight[: 2 * KEY_DIM],
            key_heads=KEY_HEADS,
            value_heads=VALUE_HEADS,
            head_k_dim=HEAD_K,
        )
        # the host's old sequence, inline: prefix convolution, split, normalise, expand, coefficient
        mixed = convolve_with_ring(
            ring.clone()[:, : 2 * KEY_DIM],
            qk,
            weight[: 2 * KEY_DIM],
            slots=slots,
            runs=runs,
            write=False,
        )
        rows = mixed.shape[0]
        q = mixed[:, :KEY_DIM].reshape(rows, KEY_HEADS, HEAD_K)
        k = mixed[:, KEY_DIM:].reshape(rows, KEY_HEADS, HEAD_K)
        q, k = normalise(q, k, scale=HEAD_K**-0.5)
        q = expand_to_value_heads(q, VALUE_HEADS)
        k = expand_to_value_heads(k, VALUE_HEADS)
        want_tilde, _ = query_coefficient(q, k, beta)
        torch.testing.assert_close(got_tilde, want_tilde, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(got_q, q, rtol=2e-2, atol=2e-2)

    def test_cook_mix_and_the_assembly(self):
        ring, x, weight, slots, runs = _pieces()
        alpha = torch.rand(x.shape[0], VALUE_HEADS)
        beta = torch.rand(x.shape[0], VALUE_HEADS)
        # the caller slices the q channels away before the cook -- the operator has one
        # query and it went early
        k, v = cook_mix(
            x[:, KEY_DIM:],
            _partial(ring, slots, weight)[:, KEY_DIM:],
            weight[KEY_DIM:],
            key_heads=KEY_HEADS,
            value_heads=VALUE_HEADS,
            head_k_dim=HEAD_K,
            head_v_dim=HEAD_V,
        )
        # the reference: the host's whole old mix on the same materials, external-query branch
        mixed = convolve_with_ring(
            ring.clone(), x, weight, slots=slots, runs=runs, write=False
        )
        query = torch.randn(x.shape[0], VALUE_HEADS, HEAD_K)
        reading = torch.randn(x.shape[0], VALUE_HEADS, HEAD_V)
        want_core, (want_k, want_v) = core_from_mixed(
            mixed,
            alpha=alpha,
            beta=beta,
            key_heads=KEY_HEADS,
            value_heads=VALUE_HEADS,
            head_k_dim=HEAD_K,
            head_v_dim=HEAD_V,
            read_state=lambda _q: reading,
            query=query,
        )
        torch.testing.assert_close(k, want_k, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(v.float(), want_v.float(), rtol=2e-2, atol=2e-2)
        # mix_ready's assembly, spelled as the handler spells it
        s = beta * (k * query).sum(-1)
        core = alpha.unsqueeze(-1) * reading + s.unsqueeze(-1) * v.float()
        torch.testing.assert_close(
            core.reshape(core.shape[0], -1), want_core, rtol=2e-2, atol=2e-2
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
