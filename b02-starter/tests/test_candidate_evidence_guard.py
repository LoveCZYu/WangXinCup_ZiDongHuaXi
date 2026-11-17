from collections import Counter
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.candidate_onset.evidence_guard import EvidenceGuard


class EvidenceGuardTest(unittest.TestCase):
    def test_weak_contested_point_keeps_prior_costs_but_supported_point_keeps_penalty(self):
        tracker = SimpleNamespace(
            emissions={1: 3, 2: 3}, gate=SimpleNamespace(policy=SimpleNamespace(enter=.525)),
            affinity=SimpleNamespace(score=lambda _: np.array([-.51, -1.81, .1, 1.1])),
            motion_counts=Counter(),
        )
        prior = lambda _: [(np.array([.08, .09]), .3), (np.array([.05, .04]), .2)]
        prior.model = None
        features = np.zeros((2, 85), np.float32)
        features[:, 11] = .99
        events = [dict(batch=batch, indices=[0, 1], track_features=np.zeros(98, np.float32),
                       edge_features=features) for batch in (1, 2)]
        result = EvidenceGuard(prior, tracker)(events)
        np.testing.assert_allclose(result[0][0], [.08, 1.09])
        np.testing.assert_allclose(result[1][0], [.05, .04])
        self.assertEqual([null for _, null in result], [.3, .2])
        self.assertEqual(tracker.motion_counts["weak_identity_columns_preserved"], 1)
