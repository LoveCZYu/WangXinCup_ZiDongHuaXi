import unittest

from experiments.candidate_onset.mix_attribution import report_episodes, split_episodes
from experiments.candidate_onset.mix_attribution_presence import absence_kind


def record(frame, actual, expected=7, category='missing_candidate', **flags):
    return dict(frame=frame, batch=1, truth=actual, expected=expected, reciprocal=False,
                **(dict(tied=False, seen_expected=True, category=category) | flags))


class MixAttributionTest(unittest.TestCase):
    def test_gap_and_correct_update_separate_distinct_episodes(self):
        rows = [record(1, 8), record(3, 8), record(4, 7), record(5, 8), record(6, 9)]
        self.assertEqual([r['count'] for r in split_episodes(rows, 0)], [1, 1, 1, 1])
        self.assertEqual([r['count'] for r in split_episodes(rows, 3)], [2, 1, 1])

    def test_persistence_and_onset_attribution_count_the_same_points_once(self):
        rows = [record(1, 8, seen_expected=False), record(2, 8), record(3, 7),
                record(4, 9, category='available_but_not_selected'), record(5, 9)]
        result = report_episodes(split_episodes(rows, 3), 2)
        self.assertEqual(result['decomposition'], dict(unestablished_identity=1,
                         continued=2, available_but_not_selected=1))
        self.assertEqual(result['weighted_from_first_error'], dict(unestablished_identity=2,
                         available_but_not_selected=2))

    def test_short_missing_observations_are_not_treated_as_terminal(self):
        visible = {7: [10, 15, 21]}
        self.assertEqual(absence_kind(dict(expected=7, circle=11), visible), 'returns_in_next_four')
        self.assertEqual(absence_kind(dict(expected=7, circle=16), visible), 'returns_after_four')
        self.assertEqual(absence_kind(dict(expected=7, circle=22), visible), 'never_observed_again')


if __name__ == '__main__':
    unittest.main()
