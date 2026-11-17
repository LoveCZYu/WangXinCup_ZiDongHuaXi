import unittest

from experiments.candidate_onset.context_prevalence import pick_cases
from experiments.candidate_onset.context_prevalence_report import counts, describe


class ContextPrevalenceTest(unittest.TestCase):
    def test_pick_only_a_correct_preceding_write_and_match_a_stable_control(self):
        records = {
            1: [(4, 7, 7, (8., 2.)), (5, 7, 7, (.3, 2.)), (6, 7, 7, (.6, 2.)),
                (7, 7, 7, (.4, 2.)), (8, 9, 7, (9., 2.)), (9, 9, 7, (10., 2.))],
            2: [(10, 3, 3, (.62, 2.)), (11, 3, 3, None), (12, 3, 3, None), (13, 3, 3, None)],
        }
        chosen, cases, _ = pick_cases(records, 30, dict(horizon=6, cases_per_scene=8, precursor_scans=3))
        self.assertEqual(cases, 1)
        self.assertEqual((chosen[0]['start'], chosen[0]['onset'], chosen[0]['kind']), (6, 8, 'precursor'))
        self.assertEqual((chosen[1]['batch'], chosen[1]['kind']), (2, 'control'))

    def test_fewer_mix_points_with_lost_correct_outputs_is_not_a_benefit(self):
        row = dict(wrong_reduction=2, correct_gain=-1, onset_prevented=True,
                   onset=8, start=7, held=['silent_or_other'])
        result = counts([row])
        self.assertEqual((result['improved'], result['harmed'], result['onset_became_correct']), (0, 1, 0))

    def test_scene_aggregation_preserves_negative_and_zero_changes(self):
        row = dict(kind='precursor', onset_prevented=False, correct_gain=0)
        scenes = [dict(scene=str(i), rows=[dict(row, wrong_reduction=value)],
                       writes=1, precursor_candidates=1, normal_candidates=0)
                  for i, value in enumerate((-3, 2))]
        result = describe(scenes)['precursor']
        self.assertEqual((result['wrong_reduction'], result['correct_gain']), (-1, 0))


if __name__ == '__main__':
    unittest.main()
