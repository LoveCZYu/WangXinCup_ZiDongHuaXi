from types import SimpleNamespace
import unittest

from experiments.candidate_onset.propagation_policy import Quarantine


class Tracker:
    def __init__(self):
        self.state = {"tracks": {
            7: SimpleNamespace(position=(1., 2., 3.), velocity=(.5, 0., 0.), last_scan=10, hits=8)
        }}
        self.history = {7:[1, 2, 3, 4]}
        self.context_history = {7:[1, 2, 3, 4]}
        self.quality_hits = {7:4}
        self.strong_hits = {7:4}
        self.emissions = {7:4}
        self.memories = {}

    def __call__(self, points, scores):
        track = self.state["tracks"][7]
        track.position, track.velocity, track.last_scan = (5., 8., 9.), (3., 5., 6.), 12
        track.hits += 1
        self.history[7].append(5)
        self.emissions[7] += 1
        return [{"BatchID":7,"PntID":3}]


class PropagationPolicyTest(unittest.TestCase):
    def test_full_rollback_preserves_output_while_restoring_association_history(self):
        tracker = Tracker()
        screen = SimpleNamespace(flagged=lambda tracker,scores,eligible:[(7,0)] if 7 in eligible else [])
        result = Quarantine(tracker,screen,full=True)([{"PntID":3}],[.99])
        self.assertEqual(result,[{"BatchID":7,"PntID":3}])
        self.assertEqual(tracker.emissions[7],5)
        self.assertEqual(tracker.history[7],[1,2,3,4])
        self.assertEqual(tracker.state["tracks"][7].position,(1.,2.,3.))
        self.assertEqual(tracker.state["tracks"][7].last_scan,10)

    def test_motion_rollback_advances_prior_state_without_restoring_time_or_history(self):
        tracker = Tracker()
        screen = SimpleNamespace(flagged=lambda tracker,scores,eligible:[(7,0)] if 7 in eligible else [])
        Quarantine(tracker,screen,full=False)([{"PntID":3}],[.99])
        track = tracker.state["tracks"][7]
        self.assertEqual(track.position,(2.,2.,3.))
        self.assertEqual(track.velocity,(.5,0.,0.))
        self.assertEqual(track.last_scan,12)
        self.assertEqual(tracker.history[7],[1,2,3,4,5])
