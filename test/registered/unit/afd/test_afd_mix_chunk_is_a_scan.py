"""A prefill chunk mixed in one call must equal its tokens mixed one at a time.

This is the property a chunk HAS, not a property the arrangement chose: consecutive rows of a
prefill chunk are one request's own tokens, so token n reads the state token n-1 wrote. The decode
batch is the other shape -- one token from each of several requests, no two sharing a slot -- and
there one contraction for the whole batch against the state as it stands is exactly right.

`OP_STATE_MIX` served every rider with the decode shape whatever its row count, so a chunk read the
pre-chunk state for all of its tokens and advanced the state once. Nothing raises: the first token
of a chunk is exact, each later one is further off, and the wrongness is carried in the state for
the rest of the request. Measured on Qwen3.8-27B against the same checkpoint colocated, it was
exact at --chunked-prefill-size 1 and 0.22 nats of mean top-1 logprob away at 512, with top-1
itself changing at 6 of 31 prefix lengths.

`ask_host` had already learned this -- it sends OP_STATE_SCAN for a rider with more than one row,
and `_scan` loops. `convolve_with_ring` had learned it too. The mix op was added later, when the
convolution moved to the host, and did not; this is the fourth time the same mistake has been made
in this arrangement, and the first time it has had a test.

N single-row calls are the reference, and they need no model: a one-row mix IS a decode step, and
a chunk is defined as the composition of them.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.afd.history_service import HistoryService
from sglang.srt.afd.linear_history import HistoryCache
from sglang.srt.afd.protocol import OP_STATE_MIX, Frame
from sglang.test.test_utils import CustomTestCase

KH, VH, DK, DV, TAPS = 3, 3, 4, 4, 3
WIDTH = 2 * KH * DK + VH * DV
LAYERS = 2
REQUEST = 11


def a_service(ids, conv_dtype=torch.float32):
    """A history that runs the convolution, which is what a MIX frame requires of it.

    `conv_dtype` is float32 here and bfloat16 on the deployment. The two mix paths reach the
    convolution through different kernels -- one row broadcasts a weighted sum, a chunk goes to
    `conv1d` -- so in bfloat16 they round differently and an exact comparison would be measuring
    that rather than the state arithmetic this file is about. One case below runs the deployed
    dtype so the rounding itself stays bounded.
    """
    cache = HistoryCache(
        slots=4,
        layers=LAYERS,
        value_heads=VH,
        head_k_dim=DK,
        head_v_dim=DV,
        conv_width=WIDTH,
        conv_taps=TAPS,
        conv_dtype=conv_dtype,
        device=torch.device("cpu"),
    )
    torch.manual_seed(20260828)
    # the RING's dtype: the deployment holds the weight in the dtype the ring is in, and
    # `convolve_with_ring` casts only the device. The one-row path broadcasts and would take a
    # mismatch; the chunk path reaches `conv1d`, which will not.
    weights = [
        (torch.randn(WIDTH, TAPS) * 0.3).to(cache.conv.dtype) for _ in range(LAYERS)
    ]
    return cache, HistoryService(
        cache,
        rows_of=lambda frame: list(ids),
        conv_weight=lambda layer: weights[layer],
        dims=(KH, VH, DK, DV),
    )


def mixes(service, qkv, alpha, beta, step):
    """One MIX frame, and the drain the inbound worker runs between one reply and the next."""
    out = service(Frame(1, 0, (qkv[step], alpha[step], beta[step]), OP_STATE_MIX))[0]
    service.drain()
    return out


def a_chunk(tokens: int):
    """One request's tokens, and the gates that go with them."""
    torch.manual_seed(4)
    qkv = torch.randn(tokens, WIDTH)
    # a decay in (0, 1); a saturated one would hide a state that never advanced
    alpha = torch.rand(tokens, VH) * 0.5 + 0.25
    beta = torch.rand(tokens, VH)
    return qkv, alpha, beta


def seeded(cache):
    """A state and a ring that are not zero. Against zeros, a stale read and a fresh one agree."""
    torch.manual_seed(7)
    cache.state.normal_()
    cache.conv.normal_().mul_(0.1)


class TestAChunkIsItsTokens(CustomTestCase):
    def test_four_tokens_at_once_equal_four_tokens_one_at_a_time(self):
        tokens = 4
        qkv, alpha, beta = a_chunk(tokens)

        cache_a, chunked = a_service([REQUEST] * tokens)
        seeded(cache_a)
        whole = mixes(chunked, qkv, alpha, beta, slice(None))

        cache_b, stepped = a_service([REQUEST])
        seeded(cache_b)
        one_at_a_time = [
            mixes(stepped, qkv, alpha, beta, slice(t, t + 1)) for t in range(tokens)
        ]

        torch.testing.assert_close(whole, torch.cat(one_at_a_time, dim=0))

    def test_it_holds_in_the_dtype_the_deployment_convolves_in(self):
        """bfloat16 on the ring, which is what the host runs. The two paths round differently
        there, so this bounds that rounding rather than asserting equality -- and the bound is far
        below the error the missing split produced, which moved the reply by tens of percent."""
        tokens = 4
        qkv, alpha, beta = a_chunk(tokens)

        cache_a, chunked = a_service([REQUEST] * tokens, conv_dtype=torch.bfloat16)
        seeded(cache_a)
        whole = mixes(chunked, qkv, alpha, beta, slice(None))

        cache_b, stepped = a_service([REQUEST], conv_dtype=torch.bfloat16)
        seeded(cache_b)
        one_at_a_time = torch.cat(
            [mixes(stepped, qkv, alpha, beta, slice(t, t + 1)) for t in range(tokens)]
        )

        gap = (whole - one_at_a_time).norm() / one_at_a_time.norm()
        self.assertLess(float(gap), 0.01, "a chunk is its tokens to within the ring's rounding")

    def test_the_state_it_leaves_is_the_state_the_steps_leave(self):
        """The reply is half of it. A chunk that answered right and advanced once would pass the
        check above on its first chunk and be wrong for every token after it."""
        tokens = 4
        qkv, alpha, beta = a_chunk(tokens)

        cache_a, chunked = a_service([REQUEST] * tokens)
        seeded(cache_a)
        mixes(chunked, qkv, alpha, beta, slice(None))

        cache_b, stepped = a_service([REQUEST])
        seeded(cache_b)
        for t in range(tokens):
            mixes(stepped, qkv, alpha, beta, slice(t, t + 1))

        slot_a, slot_b = cache_a.slot_of(REQUEST), cache_b.slot_of(REQUEST)
        torch.testing.assert_close(cache_a.state[0, slot_a], cache_b.state[0, slot_b])
        torch.testing.assert_close(cache_a.conv[0, slot_a], cache_b.conv[0, slot_b])

    def test_a_chunk_leaves_nothing_parked(self):
        """Parking defers one advance past a reply. A chunk has one per token and the next token
        is already waiting on it, so there is nothing to defer it past."""
        tokens = 3
        qkv, alpha, beta = a_chunk(tokens)
        cache, service = a_service([REQUEST] * tokens)
        seeded(cache)
        service(Frame(1, 0, (qkv, alpha, beta), OP_STATE_MIX))
        self.assertIsNone(service._parked)

    def test_one_row_still_parks_its_advance(self):
        """The decode path is the latency the arrangement is built around; the split must not
        cost it its deferred update."""
        qkv, alpha, beta = a_chunk(1)
        cache, service = a_service([REQUEST])
        seeded(cache)
        before = cache.state[0, cache.slot_of(REQUEST)].clone()
        service(Frame(1, 0, (qkv, alpha, beta), OP_STATE_MIX))
        self.assertIsNotNone(service._parked)
        torch.testing.assert_close(cache.state[0, cache.slot_of(REQUEST)], before)
        service.drain()
        self.assertFalse(
            torch.allclose(cache.state[0, cache.slot_of(REQUEST)], before)
        )


class TestSeveralRunsOnOneBus(CustomTestCase):
    """A bus can carry more than one run, and only rows WITHIN a run are sequentially dependent.

    `_runs` says so in as many words -- "a bus carrying both has runs of both lengths" -- so the
    chunk path walks the step index and takes every run's t'th row together rather than walking
    the batch row by row. That is a real difference in what is computed if it is got wrong: two
    requests' rows are adjacent in the batch and land in different slots, and a loop that mixed
    them up would fold one request's history into another with nothing in the reply to say so.

    The two solo services are seeded to the SAME state as the bus's two slots, so what is being
    compared is the arithmetic and not the fixture's random draw.
    """

    def _seeded_pair(self, tokens):
        torch.manual_seed(11)
        a = (torch.randn(tokens, WIDTH), torch.rand(tokens, VH) * 0.5 + 0.25, torch.rand(tokens, VH))
        b = (torch.randn(tokens, WIDTH), torch.rand(tokens, VH) * 0.5 + 0.25, torch.rand(tokens, VH))
        torch.manual_seed(3)
        state0 = torch.randn(VH, DV, DK)
        ring0 = torch.randn(WIDTH, TAPS) * 0.1
        return a, b, state0, ring0

    def _solo(self, chunk, state0, ring0, request):
        cache, service = a_service([request] * chunk[0].shape[0])
        slot = cache.slot_of(request)
        cache.state[0, slot] = state0
        cache.conv[0, slot] = ring0.to(cache.conv.dtype)
        out = mixes(service, chunk[0], chunk[1], chunk[2], slice(None))
        return out, cache.state[0, slot].clone()

    def test_two_requests_on_one_bus_equal_two_buses(self):
        tokens = 3
        (qa, aa, ba), (qb, ab, bb), state0, ring0 = self._seeded_pair(tokens)

        cache, bus = a_service([101] * tokens + [202] * tokens)
        sa, sb = cache.slot_of(101), cache.slot_of(202)
        self.assertNotEqual(sa, sb, "two requests must not share a slot")
        for slot in (sa, sb):
            cache.state[0, slot] = state0
            cache.conv[0, slot] = ring0.to(cache.conv.dtype)
        both = mixes(
            bus,
            torch.cat([qa, qb]), torch.cat([aa, ab]), torch.cat([ba, bb]),
            slice(None),
        )

        want_a, state_a = self._solo((qa, aa, ba), state0, ring0, 101)
        want_b, state_b = self._solo((qb, ab, bb), state0, ring0, 202)
        torch.testing.assert_close(both[:tokens], want_a)
        torch.testing.assert_close(both[tokens:], want_b)
        torch.testing.assert_close(cache.state[0, sa], state_a)
        torch.testing.assert_close(cache.state[0, sb], state_b)

    def test_a_decode_row_riding_beside_a_chunk_is_still_a_decode_row(self):
        """A mixed bus. The single row is not part of the chunk and must not be walked behind it."""
        (qa, aa, ba), (qb, ab, bb), state0, ring0 = self._seeded_pair(3)
        qb, ab, bb = qb[:1], ab[:1], bb[:1]

        cache, bus = a_service([101] * 3 + [202])
        sa, sb = cache.slot_of(101), cache.slot_of(202)
        for slot in (sa, sb):
            cache.state[0, slot] = state0
            cache.conv[0, slot] = ring0.to(cache.conv.dtype)
        both = mixes(
            bus,
            torch.cat([qa, qb]), torch.cat([aa, ab]), torch.cat([ba, bb]),
            slice(None),
        )
        want_a, _ = self._solo((qa, aa, ba), state0, ring0, 101)
        want_b, _ = self._solo((qb, ab, bb), state0, ring0, 202)
        torch.testing.assert_close(both[:3], want_a)
        torch.testing.assert_close(both[3:], want_b)


if __name__ == "__main__":
    unittest.main()
