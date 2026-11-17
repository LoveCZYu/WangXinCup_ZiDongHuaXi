import copy
from collections import Counter
from types import SimpleNamespace
import unittest

from experiments.identity_competition.preflight import CompetitionAudit, prefix_anchor


def tracker():
    return SimpleNamespace(events=[{'batch': 1, 'indices': [0], 'gated_indices': [0]}],
        retired_edges=[], proposals=[], last_matches={1: 0}, trusted_updates={},
        gate=SimpleNamespace(policy=SimpleNamespace(enter=.525, maintain=.05)))


class IdentityPreflightTest(unittest.TestCase):
    def test_anchor_requires_repeated_pure_past_known_labels(self):
        self.assertIsNone(prefix_anchor(Counter({7: 2})))
        self.assertEqual(prefix_anchor(Counter({7: 4, 0: 1})), 7)
        self.assertIsNone(prefix_anchor(Counter({7: 3, 0: 1})))

    def test_current_observation_cannot_establish_its_own_anchor(self):
        audit = CompetitionAudit()
        t = tracker()
        before = copy.deepcopy(t)
        audit.observe([{'PntID': 10}], {10: 7}, [{'PntID': 10, 'BatchID': 1}], t, {1}, [.9])
        self.assertEqual(audit.counts['anchored_recorded_rows'], 0)
        self.assertEqual(audit.counts['prefix_canonical_points'], 0)
        self.assertEqual(audit.by_truth[7][1], 1)
        self.assertEqual(t, before)

    def test_lost_candidate_and_wrong_internal_identity_are_distinct(self):
        audit = CompetitionAudit()
        audit.by_id[1].update({7: 4})
        audit.by_truth[7].update({1: 4})
        t = tracker()
        t.events[0]['indices'] = [0, 1]
        t.events[0]['gated_indices'] = [0, 1]
        t.last_matches = {1: 1}
        audit.observe([{'PntID': 10}, {'PntID': 11}], {10: 7, 11: 8},
            [{'PntID': 10, 'BatchID': 2}, {'PntID': 11, 'BatchID': 1}], t, {1}, [.9, .9])
        self.assertEqual(audit.counts['prefix_miss_recorded_edge'], 1)
        self.assertEqual(audit.counts['with_correct_candidate_other_target'], 1)
        self.assertEqual(audit.counts['prefix_miss_canonical_outputs_other_target'], 1)

    def test_clutter_assignment_without_correct_candidate_needs_rejection(self):
        audit = CompetitionAudit()
        audit.by_id[1].update({7: 4})
        t = tracker()
        t.trusted_updates = {1: 0}
        audit.observe([{'PntID': 10}], {10: 0}, [], t, {1}, [.01])
        self.assertEqual(audit.counts['without_correct_candidate_clutter'], 1)
        self.assertEqual(audit.counts['row_selected_clutter_below_maintain'], 1)
        self.assertEqual(audit.counts['trusted_update_clutter'], 1)


if __name__ == '__main__':
    unittest.main()
