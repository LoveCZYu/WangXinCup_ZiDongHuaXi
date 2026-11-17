import unittest
import numpy as np
from experiments.graph_association_diagnose import audit_targets


class GraphTargetAuditTest(unittest.TestCase):
    def data(self,positive):
        return dict(row=np.zeros((2,101)),row_index=np.zeros((2,12),int),edge_row=np.array([0,1]),
            edge_col=np.array([0,0]),col_allowed=np.array([True]),row_allowed=np.array([True,True]),
            edge_known=np.array([True,True]),edge_positive=np.array(positive),col_target=np.array([True,False]))

    def test_two_positive_rows_cannot_both_take_one_point(self):
        audit = audit_targets(self.data([True,True]))
        self.assertEqual(audit['supervised_positive_rows'],2)
        self.assertEqual(audit['maximum_simultaneous_correct_rows'],1)
        self.assertEqual(audit['minimum_positive_rows_unsatisfied_by_one_to_one'],1)
        self.assertEqual(audit['positive_rows_without_any_preferred_column_edge'],1)

    def test_competition_by_a_different_identity_is_not_a_target_conflict(self):
        audit = audit_targets(self.data([True,False]))
        self.assertEqual(audit['minimum_positive_rows_unsatisfied_by_one_to_one'],0)
        self.assertEqual(audit['columns_with_multiple_semantic_positive_rows'],0)


if __name__ == '__main__':unittest.main()
