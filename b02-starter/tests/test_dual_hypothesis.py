import unittest

import numpy as np

from experiments.candidate_onset.dual_hypothesis import TrackState, advance, alternative, trajectory
from experiments.candidate_onset.dual_hypothesis_report import threshold


def observation(positions):
    return np.asarray([[x, 0, 0] for x in positions]), np.zeros((len(positions), 2)), np.ones(len(positions))


def state(position, velocity):
    return TrackState(np.array([position, 0, 0]), np.array([velocity, 0, 0]), np.zeros(2), 0)


class DualHypothesisTest(unittest.TestCase):
    def test_joint_match_never_assigns_one_point_twice(self):
        _, _, matched = advance((state(-.25, 0), state(.25, 0)), 1, observation([0]), [12])
        self.assertCountEqual(matched, [12, None])

    def test_future_observations_can_favor_transferring_the_shared_point(self):
        states = (state(-1, 1), state(1, -1))
        future = [(observation([1.6, -1]), [1, 2]), (observation([2.8, -2]), [1, 2])]
        original = trajectory(states, 1, observation([-.5]), (0, None), future)
        challenger = trajectory(states, 1, observation([-.5]), (None, 0), future)
        self.assertLess(sum(step[0] for step in challenger), sum(step[0] for step in original))

    def test_alternative_swaps_only_when_reciprocal_edge_exists(self):
        events = {11: dict(indices=[4, 7], effective_costs=[.1, .2], effective_dummy=1),
                  12: dict(indices=[4, 7], effective_costs=[.3, .25], effective_dummy=1)}
        self.assertEqual(alternative(events, {11: 4, 12: 7}, {11, 12}, 11, 4)[1:], (12, 7, True))
        events[11]['indices'].pop()
        self.assertEqual(alternative(events, {11: 4, 12: 7}, {11, 12}, 11, 4)[1:], (12, 7, False))

    def test_false_budget_uses_whole_stream_denominator_and_strict_threshold(self):
        scene = dict(counts={'eligible_1': 4000}, rows=[dict(kind=1, single_score=x) for x in [1, 3, 3, 4]])
        self.assertEqual(threshold([scene], 'single_score', .00025), 3)


if __name__ == '__main__':
    unittest.main()
