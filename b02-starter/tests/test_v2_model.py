import copy
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch needed for association model')
class AssociationModelTest(unittest.TestCase):
    def test_rank_strategy_preserves_null_and_single_candidate(self):
        import numpy as np
        import torch
        from experiments.v2_association.model import AssociationModel,Scorer
        from experiments.v2_association.features import TRACK_DIM,EDGE_DIM
        model=AssociationModel()
        with torch.no_grad():model.edge[-1].bias.fill_(8);model.null[-1].bias.fill_(-9)
        event={'indices':[1],'track_features':np.zeros(TRACK_DIM,dtype=np.float32),
               'edge_features':np.zeros((1,EDGE_DIM),dtype=np.float32)}
        for strategy in ('rank_only','trusted_rank'):
            real,null=Scorer(model,strategy=strategy)([event])[0]
            self.assertEqual(real.tolist(),[0.]);self.assertEqual(null,0.)

    def test_zero_initialized_cost_residual_reproduces_v1(self):
        import torch
        from experiments.v2_association.model import AssociationModel,Scorer
        from experiments.v2_association.tracker import Tracker
        from test_v2_tracker import sequence
        torch.manual_seed(19)
        a=Tracker('v1');b=Tracker('v1',scorer=Scorer(AssociationModel()))
        for points,scores in sequence():
            self.assertEqual(a(points,scores),b(points,scores))
            self.assertEqual(a.state,b.state)

    def test_pair_network_can_learn_identity_interactions_and_ignore_masked_edges(self):
        import torch
        from experiments.v2_association.model import AssociationModel,association_loss
        from experiments.v2_association.features import TRACK_DIM,EDGE_DIM,MAX_EDGES
        before=torch.get_num_threads();torch.set_num_threads(1)
        try:
            torch.manual_seed(19);model=AssociationModel()
            track=torch.zeros((4,TRACK_DIM));edges=torch.zeros((4,MAX_EDGES,EDGE_DIM))
            track[:,0]=torch.tensor([1.,-1.,1.,-1.])
            edges[:,:2,0]=torch.tensor([[1.,-1.]]*4)
            base=torch.ones((4,MAX_EDGES+1));valid=torch.zeros_like(base,dtype=torch.bool)
            valid[:,:2]=True;valid[:,-1]=True;target=torch.zeros_like(valid)
            target[torch.arange(4),torch.tensor([0,1,0,1])]=True
            optimizer=torch.optim.Adam(model.parameters(),lr=.03)
            for _ in range(70):
                optimizer.zero_grad();loss=association_loss(model,track,edges,base,valid,target)
                loss.backward();optimizer.step()
            delta=model(track,edges)
            logits=(-base/.25-delta).masked_fill(~valid,-1e9)
            self.assertEqual(logits.argmax(1).tolist(),[0,1,0,1])
            self.assertLess(float(loss),.1)
            changed=edges.clone();changed[:,3:]=999
            first=association_loss(model,track,edges,base,valid,target)
            self.assertEqual(float(first),float(association_loss(model,track,changed,base,valid,target)))
        finally:torch.set_num_threads(before)


if __name__=='__main__':unittest.main()
