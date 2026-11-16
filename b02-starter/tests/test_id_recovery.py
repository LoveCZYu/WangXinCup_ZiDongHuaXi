import unittest

import numpy as np

from experiments.id_recovery.tracker import CONFIGS, RecoveryTracker
from experiments.v2_association.model import AssociationModel, Scorer
from experiments.v3_tracking.tracker import Tracker
from test_v2_tracker import sequence


class StrongAffinity:
    def score(self, examples):
        return np.full(len(examples), 8., dtype=np.float32)


def point(scan):
    return dict(PntID=scan, CircleID=scan, Azimuth=.20, Elevation=.20, Distance=.32, Amp=.8, SNR=.8, Time=scan*5000)


class RecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = Scorer(AssociationModel(98, 85), strategy='rank_only')

    def test_disabled_recovers_original_outputs_and_state(self):
        prior = Tracker(self.scorer, memory=True)
        tracked = RecoveryTracker(self.scorer, None, CONFIGS['baseline'])
        for points, scores in sequence(frames=16):
            self.assertEqual(tracked(points, scores), prior(points, scores))
            self.assertEqual(tracked.state, prior.state)

    def test_retired_id_reconnects_after_deletion_using_only_current_point(self):
        recovered = RecoveryTracker(self.scorer, StrongAffinity(), CONFIGS['retired'])
        for scan in range(12):
            outputs = recovered([point(scan)], [.99])
        old = outputs[0]['BatchID']
        for _ in range(4):
            self.assertEqual(recovered([], []), [])
        self.assertNotIn(old, recovered.state['tracks'])
        self.assertIn(old, recovered.retired)
        outputs = recovered([point(16)], [.99])
        self.assertEqual([(p['BatchID'], p['PntID']) for p in outputs], [(old, 16)])
        self.assertNotIn(old, recovered.retired)

    def test_retired_id_expires(self):
        recovered = RecoveryTracker(self.scorer, StrongAffinity(), CONFIGS['retired_strict'])
        for scan in range(12):
            outputs = recovered([point(scan)], [.99])
        old = outputs[0]['BatchID']
        for _ in range(9):
            recovered([], [])
        self.assertNotIn(old, recovered.retired)
        self.assertEqual(recovered([point(21)], [.99]), [])

    def test_active_recovery_transfers_existing_strong_match_once(self):
        recovered = RecoveryTracker(self.scorer, StrongAffinity(), CONFIGS['active'])
        recovered.emissions[7] = 20
        recovered.last_emission[7] = 9
        matched = {8: 0}
        events = [dict(batch=batch, indices=[0], track_features=np.zeros(98), edge_features=np.zeros((1, 85)))
                  for batch in (7, 8)]
        recovered.reconnect_active(matched, events, 10, [.99])
        self.assertEqual(matched, {7: 0})
        self.assertEqual(recovered.decisions, [('active', 7, 0, 8)])


if __name__ == '__main__':
    unittest.main()
