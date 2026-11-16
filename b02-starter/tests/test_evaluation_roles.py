import json
from pathlib import Path
import unittest

from b02.__main__ import load_split


class EvaluationRoleTest(unittest.TestCase):
    def test_current_independent_holdout_excludes_labelled_audit_scenes(self):
        root=Path(__file__).resolve().parents[1]
        policy=json.loads((root/'splits/a_board.json').read_text(encoding='utf-8'))
        current=load_split('holdout_unseen')
        self.assertEqual(len(current),29)
        self.assertFalse(set(current) & set(policy['labelled_training_csvs_used_in_score_audit']))
        self.assertEqual(set(load_split('holdout'))-set(current),{'train_003'})

    def test_same_packet_names_do_not_claim_verified_csv_overlap(self):
        root=Path(__file__).resolve().parents[1]
        policy=json.loads((root/'splits/a_board.json').read_text(encoding='utf-8'))
        self.assertFalse(policy['actual_training_or_holdout_overlap_verified'])
        self.assertIsNone(policy['historical_holdout_overlap'])
        self.assertEqual(len(policy['same_name_csv_input_comparison']),5)
        for row in policy['same_name_csv_input_comparison']:
            self.assertNotEqual(row['training_csv_points'],row['official_dat_tcp_points'])

    def test_historical_training_and_development_splits_stay_unchanged(self):
        root=Path(__file__).resolve().parents[1]
        original=json.loads((root/'splits/v1.json').read_text(encoding='utf-8'))
        for name in ('train','dev','holdout'):
            self.assertEqual(load_split(name),original[name])


if __name__=='__main__':unittest.main()
