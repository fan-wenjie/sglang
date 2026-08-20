"""The partition itself: which cached positions the sweep covers, and which the join does.

This is the piece with nothing behind it. A wrong query hook shows up as a wrong answer; a wrong
partition does not, because the two halves are merged and the merge is exact for ANY partition of
the keys -- including one that drops a position or counts it twice. The output stays a plausible
attention over the wrong set. So the boundary is checked here, in arithmetic, against the one
definition it has:

    sweep covers every position already in the cache -- each request's run in kv_indices, less
    its own last element, which is the slot this step just allocated
    join covers exactly that last element

Run on the CPU with a stubbed backend, because the question is index arithmetic and a GPU would
only make it slower to find out.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.test.test_utils import CustomTestCase

import unittest

import torch
from sglang.srt.afd.split_attention import _build_prefix


class Metadata:
    def __init__(self, kv_indptr, kv_indices):
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices


class Backend:
    """Only what _build_prefix reads. get_num_kv_splits records what it was asked about."""

    def __init__(self, metadata):
        self.forward_metadata = metadata
        self.asked_lens = None

    def get_num_kv_splits(self, out, seq_lens):
        self.asked_lens = seq_lens.clone()
        out.fill_(1)


class Batch:
    def __init__(self, seq_lens, seq_lens_sum):
        self.seq_lens = seq_lens
        self.seq_lens_sum = seq_lens_sum
        self.batch_size = len(seq_lens)


def _make(seq_lens, seq_lens_sum, first_slot=100):
    """Requests laid out back to back in kv_indices, with distinguishable slot numbers."""
    lens = torch.tensor(seq_lens, dtype=torch.int32)
    indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int64)
    torch.cumsum(lens, dim=0, out=indptr[1:])
    indices = torch.arange(first_slot, first_slot + int(lens.sum()), dtype=torch.int64)
    backend = Backend(Metadata(indptr, indices))
    return backend, Batch(lens, seq_lens_sum), indptr, indices


class TestPartitionIndices(CustomTestCase):
    def test_the_sweep_covers_every_cached_position_but_this_steps(self):
        seq_lens = [5, 1, 9]
        backend, batch, indptr, indices = _make(seq_lens, sum(seq_lens))
        prefix_indptr, prefix_indices, prefix_splits, cur_indptr, cur_indices, cur_splits = (
            _build_prefix(backend, batch)
        )

        expected = []
        for i, n in enumerate(seq_lens):
            run = indices[indptr[i] : indptr[i + 1]]
            expected.append(run[:-1])
        self.assertEqual(prefix_indices.tolist(), torch.cat(expected).tolist())
        self.assertEqual(prefix_indptr.tolist(), [0, 4, 4, 12])

    def test_the_join_covers_exactly_this_steps_slot(self):
        seq_lens = [5, 1, 9]
        backend, batch, indptr, indices = _make(seq_lens, sum(seq_lens))
        parts = _build_prefix(backend, batch)
        cur_indptr, cur_indices = parts[3], parts[4]
        last = [int(indices[indptr[i + 1] - 1]) for i in range(len(seq_lens))]
        self.assertEqual(cur_indices.tolist(), last)
        self.assertEqual(cur_indptr.tolist(), [0, 1, 2, 3], "one position per request")

    def test_together_the_two_halves_are_the_whole_cache(self):
        """The property the merge cannot check for itself: a partition, not a subset."""
        seq_lens = [3, 7, 2, 11]
        backend, batch, _, indices = _make(seq_lens, sum(seq_lens))
        parts = _build_prefix(backend, batch)
        covered = sorted(parts[1].tolist() + parts[4].tolist())
        self.assertEqual(covered, sorted(indices.tolist()))
        self.assertEqual(len(covered), len(set(covered)), "a position counted twice")

    def test_the_split_count_is_asked_for_the_prefix_length_not_the_sequence_length(self):
        """A kv range shorter than its split count leaves empty splits, whose partition function
        is zero and whose output is NaN. The reduction is planned for the prefix, not the run."""
        seq_lens = [5, 1, 9]
        backend, batch, _, _ = _make(seq_lens, sum(seq_lens))
        _build_prefix(backend, batch)
        self.assertEqual(backend.asked_lens.tolist(), [4, 0, 8])

    def test_an_unknown_total_gives_the_same_partition(self):
        """seq_lens_sum is None on the gpu-only path; the total is then summed rather than read."""
        seq_lens = [5, 1, 9]
        with_sum = _build_prefix(*_make(seq_lens, sum(seq_lens))[:2])
        without = _build_prefix(*_make(seq_lens, None)[:2])
        self.assertEqual(with_sum[1].tolist(), without[1].tolist())
        self.assertEqual(with_sum[4].tolist(), without[4].tolist())

    def test_nothing_to_sweep_is_refused_rather_than_partitioned(self):
        """Every request one token long: the sweep's half is empty for all of them, and an
        attention whose entire cache is this step's own token is not a partitioned attention."""
        backend, batch, _, _ = _make([1, 1], 2)
        self.assertIsNone(_build_prefix(backend, batch))


if __name__ == "__main__":
    unittest.main(verbosity=2)
