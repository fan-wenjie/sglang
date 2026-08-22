"""The pool's token-by-token scan against the model's chunked delta rule.

`benchmark/afd/gdn_split.py` verifies the split recurrence against
`fused_recurrent_gated_delta_rule_packed_decode` -- the DECODE kernel. The model's prefill runs
`chunk_gated_delta_rule`, a different algorithm for the same recurrence, and until this file
nothing compared the span to it. That gap mattered: row 0 of a first prefill reads an empty state,
so whatever makes a layer differ from the model has to come from the LATER rows of a chunk, which
is precisely where a sequential scan and a chunked one could part.

They do not. The scan agrees to bfloat16 rounding, so this file records an exclusion rather than a
guard against a bug that happened -- but it is a guard: the two implementations are free to drift
apart under any upstream change to either, and nothing else in the tree would notice.

No model is built. This compares two implementations of one recurrence at Qwen3.8-27B's own head
shapes; the tiny stack's 32-wide heads are below what the chunked kernel compiles for.
"""

import unittest

import torch
from sglang.test.test_utils import CustomTestCase

KH, VH, DK, DV = 16, 48, 128, 128
CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "the chunked kernel is CUDA-only")
class TestTheScanIsTheChunkedKernel(CustomTestCase):
    def a_chunk(self, n=8, seed=21):
        torch.manual_seed(seed)
        d = "cuda"
        return dict(
            q=torch.randn(1, n, KH, DK, device=d, dtype=torch.bfloat16),
            k=torch.randn(1, n, KH, DK, device=d, dtype=torch.bfloat16),
            v=torch.randn(1, n, VH, DV, device=d, dtype=torch.bfloat16),
            # log-space decay, as the model passes it: alpha = exp(g)
            g=-torch.rand(1, n, VH, device=d).float() * 0.5,
            beta=torch.rand(1, n, VH, device=d).float(),
            n=n,
        )

    def chunked_from(self, c, state, q=None, k=None):
        """The chunked kernel, continuing from a state IT built. Returns (output, final state)."""
        from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule

        n = c["n"]
        got = chunk_gated_delta_rule(
            q=c["q"] if q is None else q, k=c["k"] if k is None else k, v=c["v"],
            g=c["g"], beta=c["beta"], initial_state=state,
            cu_seqlens=torch.tensor([0, n], device="cuda", dtype=torch.int32),
            head_first=False, use_qk_l2norm_in_kernel=True,
            initial_state_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
        )
        # The kernel returns three things and the second is None; the final state is not among
        # them in a usable form. It advances `initial_state` IN PLACE, which is how the backend
        # chains chunks -- `ssm_states` is passed and mutated -- so the state to carry forward is
        # the tensor that went in. Feeding `got[1]` back made triton fail to compile on a null
        # pointer rather than say what was wrong.
        return got[0].reshape(n, -1).float(), state

    def scanned_from(self, c, state, q=None, k=None):
        """The token-by-token scan, continuing from a state IT built. Returns (output, state)."""
        from sglang.srt.afd.linear_history import expand_to_value_heads, normalise
        from sglang.srt.afd.split_read_kernel import read_one, update_only

        n = c["n"]
        qn, kn = normalise((c["q"] if q is None else q)[0].float(),
                           (c["k"] if k is None else k)[0].float(), scale=DK ** -0.5)
        qn, kn = expand_to_value_heads(qn, VH), expand_to_value_heads(kn, VH)
        alpha = torch.exp(c["g"][0])
        slots = torch.zeros(1, dtype=torch.long, device="cuda")
        out = []
        for t in range(n):
            qt, kt, vt = qn[t : t + 1], kn[t : t + 1], c["v"][0, t : t + 1].float()
            at, bt = alpha[t : t + 1], c["beta"][0, t : t + 1]
            h_q, h_k = read_one(state, slots, qt), read_one(state, slots, kt)
            s = (bt * (kt * qt).sum(-1)).unsqueeze(-1)
            a = at.unsqueeze(-1)
            out.append(a * h_q + s * (vt - a * h_k))
            update_only(state, slots, k=kt, v=vt, alpha=at, beta=bt)
        return torch.cat(out, dim=0).reshape(n, -1).float(), state

    def test_a_long_chunk_from_a_state_each_side_built_itself(self):
        """The deployed size, and a state that is not zero.

        The eight-token zero-state case below says the two algorithms agree to 0.0042. The
        deployment runs 122 tokens onto a state a previous chunk left, and "they agree" was never
        measured there -- so a 1 to 5 percent spread seen in the arrangement had nothing to be
        judged against.

        Each side builds its own initial state from the SAME warm-up tokens, rather than one
        tensor being handed to both. The chunked kernel keeps (heads, key dim, value dim) and
        `read_one` keeps (heads, value dim, key dim) -- transposed -- and seeding across that
        boundary by hand is exactly the kind of comparison that measures its own transpose.
        """
        warm = self.a_chunk(n=64, seed=3)
        _, chunk_state = self.chunked_from(
            warm, torch.zeros(1, VH, DK, DV, device="cuda", dtype=torch.float32))
        _, scan_state = self.scanned_from(
            warm, torch.zeros(1, VH, DV, DK, device="cuda", dtype=torch.float32))

        long = self.a_chunk(n=122, seed=9)
        a, _ = self.scanned_from(long, scan_state)
        b, _ = self.chunked_from(long, chunk_state)
        rel = self.relative(a, b)
        print(f"\n  122 tokens onto a warmed state: relative {rel:.6g}")
        self.assertLess(rel, 0.05)

    def chunked(self, c, q=None, k=None):
        from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule

        n = c["n"]
        got = chunk_gated_delta_rule(
            q=c["q"] if q is None else q, k=c["k"] if k is None else k, v=c["v"],
            g=c["g"], beta=c["beta"],
            initial_state=torch.zeros(1, VH, DK, DV, device="cuda", dtype=torch.float32),
            cu_seqlens=torch.tensor([0, n], device="cuda", dtype=torch.int32),
            head_first=False, use_qk_l2norm_in_kernel=True,
            initial_state_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
        )
        return got[0].reshape(n, -1).float()

    def scanned(self, c, q=None, k=None):
        from sglang.srt.afd.linear_history import expand_to_value_heads, normalise
        from sglang.srt.afd.split_read_kernel import read_one, update_only

        n = c["n"]
        qn, kn = normalise((c["q"] if q is None else q)[0].float(),
                           (c["k"] if k is None else k)[0].float(), scale=DK ** -0.5)
        qn, kn = expand_to_value_heads(qn, VH), expand_to_value_heads(kn, VH)
        alpha = torch.exp(c["g"][0])
        state = torch.zeros(1, VH, DV, DK, device="cuda", dtype=torch.float32)
        slots = torch.zeros(1, dtype=torch.long, device="cuda")
        out = []
        for t in range(n):
            qt, kt, vt = qn[t : t + 1], kn[t : t + 1], c["v"][0, t : t + 1].float()
            at, bt = alpha[t : t + 1], c["beta"][0, t : t + 1]
            h_q, h_k = read_one(state, slots, qt), read_one(state, slots, kt)
            s = (bt * (kt * qt).sum(-1)).unsqueeze(-1)
            a = at.unsqueeze(-1)
            out.append(a * h_q + s * (vt - a * h_k))
            update_only(state, slots, k=kt, v=vt, alpha=at, beta=bt)
        return torch.cat(out, dim=0).reshape(n, -1).float()

    def relative(self, a, b):
        return float((a - b).norm() / (b.norm() + 1e-9))

    def test_a_chunk_scanned_equals_a_chunk_chunked(self):
        c = self.a_chunk()
        self.assertLess(self.relative(self.scanned(c), self.chunked(c)), 0.02)

    def test_the_control_permutes_one_side_only(self):
        """Two rows swapped on ONE side. Permuting both changes the same thing in each and they
        agree again -- a control that cannot fail, which this line of work has written twice."""
        c = self.a_chunk()
        wrong = c["q"].clone()
        wrong[0, [1, 2]] = wrong[0, [2, 1]]
        self.assertGreater(self.relative(self.scanned(c, q=wrong), self.chunked(c)), 0.2)


if __name__ == "__main__":
    unittest.main()
