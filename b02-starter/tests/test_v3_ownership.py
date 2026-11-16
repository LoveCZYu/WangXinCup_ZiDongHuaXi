from collections import defaultdict
import copy
import unittest
import numpy as np
from experiments.v3_tracking.ownership import ownership_costs,OwnershipTracker


class OwnershipTest(unittest.TestCase):
    def test_single_candidate_rows_can_resolve_shared_point_and_keep_column_floor(self):
        refs={3:[(0,0),(1,0)]};prior=[[.1],[.2]];scores=[0.,0.,0.,.9]
        changed,c=ownership_costs(prior,[[4.],[0.]],refs,scores)
        self.assertLess(changed[1][0],changed[0][0]);self.assertAlmostEqual(min(x[0] for x in changed),.1)
        same,c=ownership_costs(prior,[[0.],[0.]],refs,scores)
        self.assertEqual([x.tolist() for x in same],prior)
        self.assertTrue(all((x==0).all() for x in c))

    def test_weak_points_and_sole_owners_keep_their_costs(self):
        p=[[.1,.2],[.3]];refs={0:[(0,0),(1,0)],1:[(0,1)]}
        result,_=ownership_costs(p,[[4.,-4.],[0.]],refs,[.1,.9])
        self.assertEqual([x.tolist() for x in result],p)

    def test_zero_model_reproduces_prior_and_does_not_use_current_emission_count(self):
        import torch
        from experiments.v2_association.model import AssociationModel,Scorer
        from experiments.v3_tracking.tracker import Tracker
        from test_v2_tracker import sequence
        torch.manual_seed(19);prior=AssociationModel(60,50)
        with torch.no_grad():prior.edge[-1].weight.normal_(0,.1)
        a=Tracker(Scorer(copy.deepcopy(prior),strategy='rank_only'),memory=True)
        b=OwnershipTracker(Scorer(prior,strategy='rank_only'),AssociationModel(101,88));b.graph_capture=True
        for points,scores in sequence(frames=16):
            past=dict(b.emitted);outputs=b(points,scores)
            self.assertEqual(a(points,scores),outputs);self.assertEqual(a.state,b.state)
            for graph in b.graphs:
                for row,batch in enumerate(graph['batches']):self.assertAlmostEqual(float(graph['track'][row,98]),min(4,past.get(batch,0)/50),places=6)


if __name__=='__main__':unittest.main()
