from collections import Counter
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.identity_competition.scoring import ColumnScorer


class Base:
    model = None
    def __call__(self, events):
        return [(np.zeros(len(event['indices']), dtype=np.float32), 0.) for event in events]


class Affinity:
    def score(self, features):
        return np.asarray([3., -1.])[:len(features)]


def setup():
    events = []
    for batch in (1, 2):
        edge = np.zeros((1, 85), dtype=np.float32)
        edge[0, 11] = .9
        events.append({'batch': batch, 'indices': [0], 'track_features': np.zeros(98), 'edge_features': edge})
    tracker = SimpleNamespace(emissions=Counter({1: 10, 2: 10}), gate=SimpleNamespace(policy=SimpleNamespace(enter=.525)))
    return events, tracker


class ColumnScoringTest(unittest.TestCase):
    def test_wider_stale_gate_does_not_receive_a_free_competition_discount(self):
        events, tracker = setup()
        tracker.state = {'scan': 10, 'tracks': {1: SimpleNamespace(last_scan=9), 2: SimpleNamespace(last_scan=8)}}
        delta = ColumnScorer(Base(), Affinity(), tracker, 'volume')(events)
        self.assertEqual(delta[0][0][0], 0.)
        self.assertGreater(.09 + .25*delta[1][0][0], .10)

    def test_single_candidate_per_row_does_not_disable_cross_id_evidence(self):
        events, tracker = setup()
        delta = ColumnScorer(Base(), Affinity(), tracker, 'identity')(events)
        def owner(values):
            costs = np.array([[.10 + .25*values[0][0][0], 1.15, 1e6],
                              [.09 + .25*values[1][0][0], 1e6, 1.15]])
            rows, cols = linear_sum_assignment(costs)
            return next(row+1 for row, col in zip(rows, cols) if col == 0)
        self.assertEqual(owner(Base()(events)), 2)
        self.assertEqual(owner(delta), 1)
        self.assertEqual(delta[0][0][0], 0)
        self.assertEqual(delta[1][0][0], 4)
        self.assertTrue(all(null == 0 for _, null in delta))

    def test_unpublished_or_weak_point_does_not_trigger_column_penalty(self):
        for weak in (False, True):
            events, tracker = setup()
            if weak:
                for event in events:event['edge_features'][0, 11] = .1
            else:tracker.emissions[2] = 2
            delta = ColumnScorer(Base(), Affinity(), tracker, 'identity')(events)
            self.assertTrue(all(np.array_equal(edge, [0.]) for edge, _ in delta))


if __name__ == '__main__':
    unittest.main()
