import unittest
from unittest.mock import patch

from experiments.candidate_onset.joint_context_hold import ContextHoldTracker
from experiments.candidate_onset.joint_cost_replay import JointCostTracker, Proposal
from experiments.candidate_onset.joint_event import cut_selected


class JointEventTest(unittest.TestCase):
    def test_single_event_veto_does_not_remove_other_accepted_actions(self):
        focal = Proposal(.2, 1, 20, 2, None, False, False)
        other = Proposal(.3, 3, 40, 4, 60, True, True)
        matches = {1: 20, 3: 40, 4: 60}
        accepted = cut_selected(matches, [focal, other], 0, (1, 2), 20)
        self.assertEqual(accepted, [other])
        self.assertEqual(matches, {1: 20, 3: 60, 4: 40})

    def test_hold_only_unmatched_receivers_context(self):
        tracker = ContextHoldTracker.__new__(ContextHoldTracker)
        tracker.assignment_stage = 0
        tracker.context_history = {2: ['old'], 4: ['old']}
        tracker.history = {2: ['old'], 4: ['old']}
        transfer = Proposal(.2, 1, 0, 2, None, False, False)
        swap = Proposal(.2, 3, 1, 4, 2, True, True)

        def assign(self, *_):
            self.joint_applied = [transfer, swap]
            return {2: 0, 3: 2, 4: 1}

        def call(self, points, scores):
            matches = self.assign([], 3, [], {}, set(), points, scores)
            for batch in (2, 4):
                self.context_history[batch].append('new')
                self.history[batch].append('new')
            return matches

        with patch.object(JointCostTracker, 'assign', assign), patch.object(JointCostTracker, '__call__', call):
            self.assertEqual(tracker([], []), {2: 0, 3: 2, 4: 1})
        self.assertEqual(tracker.context_history, {2: ['old'], 4: ['old', 'new']})
        self.assertEqual(tracker.history, {2: ['old', 'new'], 4: ['old', 'new']})
        self.assertEqual(tracker.total_context_holds, 1)


if __name__ == '__main__':
    unittest.main()
