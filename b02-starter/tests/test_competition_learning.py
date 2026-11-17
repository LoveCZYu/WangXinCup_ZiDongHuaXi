from collections import Counter
import json
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.competition_learning.data import label_call
from experiments.competition_learning.model import Model, objective
from experiments.competition_learning.runtime import LearnedScorer
from experiments.competition_learning.reject_calibration import fit_offset
from experiments.competition_learning.pilot import OffsetScorer
from experiments.id_recovery.affinity import Affinity


def packet():
    return (torch.zeros(2, 96), torch.zeros(2), torch.zeros(1, 101), torch.zeros(1, 1, 96),
        torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, 96), torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool))


def output():
    return {k: [] for k in ('pair', 'row', 'edge_known', 'edge_positive', 'edge_row', 'edge_col',
        'row_allowed', 'col_target', 'col_allowed', 'col_eval')}


def call(batch):
    return (batch, (0,), np.zeros(101, np.float32), np.zeros((1, 96), np.float32), np.asarray([.9]))


class CompetitionLearningTest(unittest.TestCase):
    def test_training_offset_preserves_float32_ties_without_validation_labels(self):
        state = {'margin': np.asarray([1., 1., -.1], dtype=np.float32),
                 'positive': np.asarray([True, True, True]), 'correct': np.asarray([True, True, True])}
        offset = fit_offset(state, state)
        self.assertEqual(int((state['margin'] > offset).sum()), 2)

    def test_runtime_can_reject_a_single_candidate_without_ignoring_parent_scale(self):
        class Prior:
            model = None
            def __call__(self, events):return [(np.zeros(len(e['indices'])), 0.) for e in events]
        class Learned:
            def score(self, x):return np.full(len(x), -2., dtype=np.float32)
            def rejection(self, x):return torch.full((len(x),), 2.)
        tracker = SimpleNamespace(emissions=Counter({1: 3}), last_emission={1: 0}, memories={}, state={'scan': 1})
        edge = np.zeros((1, 85), np.float32)
        event = {'batch': 1, 'indices': [0], 'track_features': np.zeros(98), 'edge_features': edge,
                 'base_costs': [.1], 'dummy_cost': 1.15}
        delta, null = LearnedScorer(Prior(), Learned(), tracker, 'learned_joint')([event])[0]
        self.assertAlmostEqual(.1+.25*delta[0], .5)
        self.assertAlmostEqual(1.15+.25*null, -.5)
        delta, null = OffsetScorer(Prior(), Learned(), tracker, .2)([event])[0]
        self.assertAlmostEqual(.1+.25*delta[0], .5)
        self.assertAlmostEqual(1.15+.25*null, -.55)
        tracker.emissions[1] = 2
        delta, null = LearnedScorer(Prior(), Learned(), tracker, 'learned_joint')([event])[0]
        self.assertEqual(delta[0], 0.)
        self.assertEqual(null, 0.)

    def test_warm_start_keeps_pretrained_function_despite_new_metadata(self):
        torch.manual_seed(17)
        old = Affinity().eval()
        with torch.no_grad():
            old.mean.copy_(torch.randn(93))
            old.std.copy_(torch.rand(93)+.1)
        model = Model().eval()
        x = torch.randn(20, 96)
        model.initialize(old, x, torch.randn(20, 101))
        torch.testing.assert_close(model.edges(x), old(x[:, :93]), atol=1e-6, rtol=1e-5)

    def test_joint_loss_can_learn_to_reject_a_single_wrong_candidate(self):
        model = Model()
        for p in model.pair.parameters():p.data.zero_()
        loss = objective(model, packet(), True)
        loss.backward()
        self.assertGreater(float(model.null[-1].bias.grad.abs().sum()), 0.)
        optimizer = torch.optim.SGD(model.parameters(), lr=.2)
        optimizer.step()
        self.assertGreater(float(model.rejection(torch.zeros(1, 101)).detach()),
                           float(model.edges(torch.zeros(1, 96)).detach()))

    def test_binary_control_does_not_accidentally_train_the_null_head(self):
        model = Model()
        objective(model, packet(), False).backward()
        self.assertTrue(all(p.grad is None for p in model.null.parameters()))

    def test_unknown_candidate_is_not_a_false_null_training_target(self):
        out = output()
        label_call([call(1)], [{'PntID': 10}], {10: -1}, {1: Counter({7: 4})}, {}, out, Counter())
        self.assertEqual(out['row_allowed'], [False])
        self.assertEqual(out['col_allowed'], [False])
        self.assertEqual(out['edge_known'], [False])

    def test_canonical_column_preference_is_distinct_from_same_identity(self):
        out = output()
        counts = Counter()
        label_call([call(1), call(2)], [{'PntID': 10}], {10: 7},
            {1: Counter({7: 4}), 2: Counter({7: 4})}, {7: Counter({1: 10, 2: 4})}, out, counts)
        self.assertEqual(out['edge_positive'], [True, True])
        self.assertEqual(out['col_target'], [True, False])
        self.assertEqual(out['row_allowed'], [True, True])
        json.dumps(dict(counts))

    def test_current_label_cannot_create_a_history_anchor(self):
        out = output()
        label_call([call(1)], [{'PntID': 10}], {10: 7}, {}, {}, out, Counter())
        self.assertEqual(out['pair'], [])


if __name__ == '__main__':unittest.main()
