import copy
import unittest

import numpy as np
import torch

from experiments.id_recovery.tracker import CONFIGS, RecoveryTracker
from experiments.prebirth.tracker import CONFIGS as PREBIRTH, PrebirthTracker
from test_id_recovery import StrongAffinity, point
from test_v2_tracker import sequence


class RejectMature:
    def __init__(self):
        self.model = type('Shape', (), {'track_mean': torch.zeros(98)})()
        self.reject = False

    def __call__(self, events):
        return [(np.full(len(e['indices']), 8. if self.reject and e['track_features'][36] >= .3 else 0.),
                 -8. if self.reject and e['track_features'][36] >= .3 else 0.) for e in events]


class PrebirthTest(unittest.TestCase):
    def test_control_is_exact_combined_state_and_output(self):
        scorer = RejectMature()
        control = RecoveryTracker(scorer, StrongAffinity(), CONFIGS['combined'])
        new = PrebirthTracker(scorer, StrongAffinity(), PREBIRTH['control'])
        for points, scores in sequence(frames=30):
            self.assertEqual(control(points, scores), new(points, scores))
            self.assertEqual(control.state, new.state)

    def test_first_output_reuses_live_id_without_delaying_or_changing_point(self):
        scorer = RejectMature()
        tracker = PrebirthTracker(scorer, StrongAffinity(), PREBIRTH['live'])
        for scan in range(12):
            out = tracker([point(scan)], [.99])
        old = out[0]['BatchID']
        scorer.reject = True
        for scan in range(12, 14):
            self.assertEqual(tracker([point(scan)], [.99]), [])
        source = point(14)
        original = copy.deepcopy(source)
        out = tracker([source], [.99])
        self.assertEqual([(p['BatchID'], p['PntID']) for p in out], [(old, 14)])
        self.assertEqual(source, original)
        self.assertEqual(tracker.decisions[0][0], 'prebirth_live')
        self.assertEqual(len({p['BatchID'] for p in out}), len(out))

    def test_distant_genuine_birth_keeps_original_confirmation_time(self):
        scorer = RejectMature()
        tracker = PrebirthTracker(scorer, StrongAffinity(), PREBIRTH['both'])
        for scan in range(12):
            out = tracker([point(scan)], [.99])
        old = out[0]['BatchID']
        for scan in range(12, 15):
            fresh = point(scan)
            fresh['Azimuth'] = .8
            out = tracker([fresh], [.99])
            if scan < 14:
                self.assertEqual(out, [])
        self.assertNotEqual(out[0]['BatchID'], old)
        self.assertEqual(tracker.decisions, [])

    def test_one_output_old_id_is_not_discarded_before_new_confirmation(self):
        scorer = RejectMature()
        tracker = PrebirthTracker(scorer, StrongAffinity(), PREBIRTH['short_live'])
        for scan in range(3):
            out = tracker([point(scan)], [.99])
        old = out[0]['BatchID']
        self.assertEqual(tracker.emissions[old], 1)
        scorer.reject = True
        for scan in range(3, 6):
            out = tracker([point(scan)], [.99])
        self.assertEqual(out[0]['BatchID'], old)
        self.assertEqual(tracker.decisions[0][0], 'prebirth_short_live')

    def test_short_retired_bank_restores_old_id(self):
        tracker = PrebirthTracker(RejectMature(), StrongAffinity(), PREBIRTH['short_both'])
        for scan in range(3):
            out = tracker([point(scan)], [.99])
        old = out[0]['BatchID']
        for _ in range(5):
            tracker([], [])
        self.assertIn(old, tracker.short_retired)
        for scan in range(8, 11):
            out = tracker([point(scan)], [.99])
        self.assertEqual(out[0]['BatchID'], old)
        self.assertNotIn(old, tracker.short_retired)

    def test_long_absence_expires_short_identity(self):
        tracker = PrebirthTracker(RejectMature(), StrongAffinity(), PREBIRTH['short_both'])
        for scan in range(3):
            out = tracker([point(scan)], [.99])
        old = out[0]['BatchID']
        for _ in range(15):
            tracker([], [])
        self.assertNotIn(old, tracker.short_retired)
        for scan in range(18, 21):
            out = tracker([point(scan)], [.99])
        self.assertNotEqual(out[0]['BatchID'], old)


if __name__ == '__main__':
    unittest.main()
