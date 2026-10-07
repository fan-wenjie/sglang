"""Step 1 of the DSA patch: a reference-set threshold in DSA's own layout.

Each case guards a failure the others would not catch:

  bootstrap        with no previous selection the step must BE DSA, exactly --
                   that is what makes the first step of a request safe.
  published set    the emitted tokens' groups must equal the groups written to
                   sel_groups, because the next step thresholds against them and
                   a mismatch would corrupt every later step silently.
  steady state     the fired set must be the groups above the q-quantile of the
                   previous selection, and it must be NON-EMPTY -- an earlier
                   suite on this line passed 15 cases while comparing two empty
                   sets, because its fixture separated the scores too far.
  multi-step       three steps on one state: the candidate counter must reset
                   and the reference set must track, which is the shape that
                   exposed a km accumulation bug the single-step cases could not.
  layout           -1 past the selection, tail appended where DSA puts it.
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


class TestVkDsaSelect(CustomTestCase):
    POOL = 4
    GROUP_TOPK = 512
    TOPK = GROUP_TOPK * POOL
    G = 4096           # groups; 4096*4 = 16k tokens, past the K<=length early path

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(0)
        self.dev = torch.device("cuda")

    def _state(self, bs=1):
        from sglang.srt.layers.attention.dsa.vk_select import select_state
        return select_state(bs=bs, group_budget=self.GROUP_TOPK, dev=self.dev)

    def _scores(self, bs=1, spread=1.0):
        return torch.randn(bs, self.G, device=self.dev, dtype=torch.float32) * spread

    def _run(self, logits, st, seq_lens=None):
        from sglang.kernels.ops.attention.dsa.vk_select_transform import vk_dsa_select
        bs = logits.shape[0]
        lengths = torch.full((bs,), self.G, dtype=torch.int32, device=self.dev)
        return vk_dsa_select(
            score=logits, lengths=lengths, thresholds=st["thr"], state=st,
            pool_size=self.POOL, topk=self.TOPK, seq_lens=seq_lens,
        )

    def _thr(self, logits, st, q=0.75):
        """No-op: the threshold carries itself now.

        The emit kernel writes st["thr"] as the minimum of what it selected, so
        there is nothing to compute between steps. Kept as a seam so the cases
        below still read step-by-step.
        """
        return st["thr"]

    @staticmethod
    def _groups(row, pool):
        v = row[row >= 0].tolist()
        return {t // pool for t in v}

    # ---- cases ----------------------------------------------------------

    def test_the_bootstrap_is_exactly_dsa(self):
        from sglang.kernels.ops.attention.dsa.kpool_topk_transform import (
            fast_kpool_topk_transform_fused,
        )
        logits = self._scores()
        st = self._state()                      # sel_n = 0, no previous selection
        self._thr(logits, st)
        self.assertEqual(float(st["thr"][0]), float("-inf"),
                         "no reference set must give -inf, which routes to DSA")
        got = self._run(logits, st)
        lengths = torch.full((1,), self.G, dtype=torch.int32, device=self.dev)
        want = fast_kpool_topk_transform_fused(
            score=logits, lengths=lengths, pool_size=self.POOL, topk=self.TOPK)
        # Sets, not elementwise: DSA's own emit is an atomicAdd race.
        self.assertEqual(set(got[0].tolist()), set(want[0].tolist()))

    def test_the_published_set_matches_what_was_emitted(self):
        logits = self._scores()
        st = self._state()
        self._thr(logits, st)
        got = self._run(logits, st)
        n = int(st["sel_n"][0])
        self.assertGreater(n, 0)
        published = set(st["sel_groups"][0, :n].tolist())
        self.assertEqual(self._groups(got[0], self.POOL), published,
                         "the next step would threshold against the wrong set")

    def test_steady_state_fires_above_the_quantile(self):
        st = self._state()
        # step 1 bootstraps and publishes a reference set
        self._thr(self._scores(), st)
        self._run(self._scores(), st)
        n0 = int(st["sel_n"][0])
        self.assertGreater(n0, 0)
        ref = st["sel_groups"][0, :n0].clone()

        # step 2 thresholds against it
        logits = self._scores()
        self._thr(logits, st)
        thr = float(st["thr"][0])
        self.assertTrue(thr > float("-inf"), "a reference set must give a finite thr")
        got = self._run(logits, st)
        fired = self._groups(got[0], self.POOL)
        self.assertGreater(len(fired), 0, "vacuous: nothing fired")

        want = set(torch.nonzero(logits[0] >= thr).flatten().tolist())
        if len(want) <= self.GROUP_TOPK:
            self.assertEqual(fired, want)
        else:
            self.assertEqual(len(fired), self.GROUP_TOPK)
            self.assertTrue(fired <= want)
        # And the quantile must sit inside the reference set's own range.
        rs = logits[0][ref.long()]
        self.assertGreaterEqual(thr, float(rs.min()) - 1e-4)
        self.assertLessEqual(thr, float(rs.max()) + 1e-4)

    def test_three_steps_on_one_state(self):
        st = self._state()
        seen = []
        for _ in range(3):
            logits = self._scores()
            self._thr(logits, st)
            got = self._run(logits, st)
            self.assertEqual(int(st["ncand"][0]), 0, "candidate counter did not reset")
            n = int(st["sel_n"][0])
            self.assertGreater(n, 0)
            self.assertLessEqual(n, self.GROUP_TOPK)
            self.assertEqual(self._groups(got[0], self.POOL),
                             set(st["sel_groups"][0, :n].tolist()))
            seen.append(n)
        self.assertTrue(all(x > 0 for x in seen), seen)

    def test_the_layout_is_dsas(self):
        logits = self._scores()
        st = self._state()
        self._thr(logits, st)
        seq = torch.full((1,), self.G * self.POOL + 3, dtype=torch.int32,
                         device=self.dev)
        got = self._run(logits, st, seq_lens=seq)
        row = got[0]
        n_valid = int((row >= 0).sum())
        self.assertTrue(bool((row[:n_valid] >= 0).all()), "selection is not a prefix")
        self.assertTrue(bool((row[n_valid:] == -1).all()), "padding is not -1")
        tail = row[n_valid - 3 : n_valid].tolist()
        self.assertEqual(tail, [self.G * self.POOL, self.G * self.POOL + 1,
                                self.G * self.POOL + 2])


if __name__ == "__main__":
    unittest.main()


class TestVkDsaSelectIndexerHook(CustomTestCase):
    """The indexer method, including per-LAYER state.

    Each full-attention layer has its own score distribution and so its own
    reference set. One shared buffer across layers is precisely the bug that
    took the sibling arm's multikey from 1.000 to 0.18 while 15 unit cases
    passed, because every case ran a single layer on a fresh buffer.
    """

    POOL = 4
    GROUP_TOPK = 512
    G = 4096

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(0)
        self.dev = torch.device("cuda")

    def _indexer(self):
        from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool

        ix = IndexerKPool.__new__(IndexerKPool)
        ix.index_topk = self.GROUP_TOPK * self.POOL
        ix.index_kpool = self.POOL
        return ix

    def test_each_layer_keeps_its_own_reference_set(self):
        from sglang.srt.environ import envs

        ix = self._indexer()
        pool_lens = torch.full((1,), self.G, dtype=torch.int32, device=self.dev)
        with envs.SGLANG_DSA_VK_SELECT_Q.override(0.75):
            # Two layers, two steps each, with deliberately different score
            # scales so a shared state would show up as a wrong threshold.
            out = {}
            for step in range(2):
                for lid, scale in ((3, 1.0), (7, 5.0)):
                    torch.manual_seed(1000 + step * 10 + lid)
                    logits = torch.randn(1, self.G, device=self.dev,
                                         dtype=torch.float32) * scale
                    r = ix._vk_select_from_kpool_logits(
                        logits, pool_lens, layer_id=lid)
                    out[(step, lid)] = (r, logits)

        states = ix._vk_select_states
        self.assertEqual(len(states), 2, "layers did not get separate state")
        self.assertEqual({k[0] for k in states}, {3, 7})

        # Each layer's published set must come from ITS OWN scores.
        for lid, scale in ((3, 1.0), (7, 5.0)):
            st = states[(lid, 1)]
            n = int(st["sel_n"][0])
            self.assertGreater(n, 0)
            r, logits = out[(1, lid)]
            emitted = {t // self.POOL for t in r[0][r[0] >= 0].tolist()}
            self.assertEqual(emitted, set(st["sel_groups"][0, :n].tolist()))
            # a threshold borrowed from the other layer's scale would land far
            # outside this layer's own score range
            thr = float(st["thr"][0])
            self.assertGreaterEqual(thr, float(logits[0].min()))
            self.assertLessEqual(thr, float(logits[0].max()))


class TestVkDsaSelectAcrossSteps(CustomTestCase):
    """CORRELATED steps -- the structure the rule actually depends on.

    Every other case here draws independent logits per step, which makes the
    previous selection a random sample under the new scores: its q-quantile is
    then just the overall q-quantile, 512 groups fire at every q, and the
    fixture cannot see the mechanism at all. That blindness is why q=0.75
    shipped and served multikey 0.72. Consecutive decode steps share keys and
    move the query slowly, so the fixture has to be correlated.
    """

    POOL = 4
    GROUP_TOPK = 512
    G = 4096

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.dev = torch.device("cuda")

    def _trace(self, q, rho=0.99, steps=5):  # noqa: D401
        from sglang.kernels.ops.attention.dsa.vk_select_transform import vk_dsa_select
        from sglang.kernels.ops.attention.dsa.kpool_topk_transform import (
            fast_kpool_topk_transform_fused,
        )
        from sglang.srt.layers.attention.dsa.vk_select import select_state
        torch.manual_seed(0)
        base = torch.randn(1, self.G, device=self.dev, dtype=torch.float32)
        st = select_state(bs=1, group_budget=self.GROUP_TOPK, dev=self.dev)
        lengths = torch.full((1,), self.G, dtype=torch.int32, device=self.dev)
        out = []
        for s in range(steps):
            torch.manual_seed(500 + s)
            noise = torch.randn(1, self.G, device=self.dev, dtype=torch.float32)
            logits = rho * base + (1.0 - rho ** 2) ** 0.5 * noise
            r = vk_dsa_select(score=logits, lengths=lengths, thresholds=st["thr"],
                              state=st, pool_size=self.POOL,
                              topk=self.GROUP_TOPK * self.POOL)
            want = fast_kpool_topk_transform_fused(
                score=logits, lengths=lengths, pool_size=self.POOL,
                topk=self.GROUP_TOPK * self.POOL)
            wg = {t // self.POOL for t in want[0].tolist() if t >= 0}
            gg = {t // self.POOL for t in r[0].tolist() if t >= 0}
            out.append((int(st["sel_n"][0]), len(gg & wg) / max(len(wg), 1)))
        return out

    def test_the_carried_threshold_is_a_fixed_point(self):
        """The carried minimum must hold the budget and the recall across steps.

        q is no longer a knob on the served path: the emit kernel writes the
        MINIMUM of what it selected, which is the q=0 case and the only
        self-consistent one. The offline sweep that found the ratchet at q>0
        lives in the register, not here.
        """
        # 20 steps, because the property is the absence of DRIFT, not the value
        # at any one step. The count above a fixed cut is ~Binomial(G, K/G), so
        # sd ~= sqrt(K(1-K/G)) ~= 21 at K=512, G=4096: a single step reading 492
        # is one sd and means nothing. What would matter is a downward trend --
        # the ratchet that q=0.75 produced in its sharp form.
        tr = self._trace(0.0, steps=20)
        ns = [n for n, _ in tr]
        recs = [r for _, r in tr]
        first5 = sum(ns[1:6]) / 5.0
        last5 = sum(ns[-5:]) / 5.0
        self.assertGreater(last5, 0.90 * self.GROUP_TOPK,
                           "budget collapsed: %s" % (ns,))
        self.assertGreater(last5, 0.90 * first5,
                           "budget drifts down: first5 %.0f last5 %.0f %s"
                           % (first5, last5, ns))
        self.assertGreater(min(recs[1:]), 0.85, "recall dipped: %s" % (
            ["%.2f" % r for r in recs],))

    @unittest.skip("q is not a served knob any more: the emit kernel carries the "
                   "minimum, which is q=0. The ratchet at q>0 is recorded in the "
                   "register with its measurements.")
    def test_a_positive_q_ratchets_the_threshold_shut(self):
        """The regression this guards: q=0.75 collapsed to 9 groups and 0.02
        recall, and served multikey 0.72. Any q>0 is unstable by construction --
        the threshold sits above the selection's own cut, so each step's
        reference set is smaller and higher-scoring than the last."""
        tr = self._trace(0.75)
        n_last, rec_last = tr[-1]
        self.assertLess(n_last, 100, "expected the ratchet; got %s" % (tr,))
        self.assertLess(rec_last, 0.2, "expected recall collapse; got %s" % (tr,))
