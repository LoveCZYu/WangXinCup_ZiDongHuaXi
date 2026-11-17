import unittest

from experiments.candidate_onset.joint_cost_replay import Proposal, apply_proposals, audit
from experiments.candidate_onset.joint_cost_report import scene_cutoff


class JointCostTest(unittest.TestCase):
    def test_disjoint_edits_preserve_other_matches_and_do_not_reuse_a_point(self):
        best = Proposal(.9, 1, 10, 2, 20, True, True)
        overlapping = Proposal(.8, 2, 20, 3, None, False, False)
        secondary = Proposal(.5, 4, 40, 5, 50, False, True)
        matches = {1: 10, 2: 20, 4: 40, 5: 50, 6: 60}
        self.assertEqual(apply_proposals(matches, [overlapping, secondary, best], .4), [best, secondary])
        self.assertEqual(matches, {1: 20, 2: 10, 5: 40, 6: 60})

    def test_truth_audit_counts_the_displaced_secondary_publication(self):
        action = Proposal(.2, 1, 0, 2, 1, False, True)
        points = [{'PntID': 5}, {'PntID': 8}]
        histories = {1: [3] * 4, 2: [7] * 4}
        self.assertEqual(audit(action, points, {5: 7, 8: 7}, histories), (1, 2, True))

    def test_calibrated_tie_is_excluded_by_strict_threshold(self):
        cutoff = scene_cutoff([.2, .7, .7, .9], 2)
        self.assertEqual(cutoff, .7)
        matches = {1: 10}
        self.assertEqual(apply_proposals(matches, [Proposal(.7, 1, 10, 2, None, False, False)], cutoff), [])
        self.assertEqual(matches, {1: 10})


if __name__ == '__main__':
    unittest.main()
