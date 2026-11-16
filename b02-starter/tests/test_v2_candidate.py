from types import SimpleNamespace
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch needed')
class CandidateStateTest(unittest.TestCase):
    def test_packet_state_is_independent_when_interleaved(self):
        import numpy as np
        import torch
        from experiments.v2_association.candidate import CandidateHandler
        from experiments.v2_association.model import AssociationModel,Scorer
        from test_v2_tracker import sequence
        class Constant(torch.nn.Module):
            def forward(self,x):return torch.full((x.shape[0],),2.)
        base=SimpleNamespace(model=Constant(),device='cpu',history=6,
            features=lambda points,history:np.zeros((len(points),1),dtype=np.float32))
        torch.manual_seed(23);model=AssociationModel(60,50)
        with torch.no_grad():model.edge[-1].weight.normal_(0,.1)
        handler=CandidateHandler(base,Scorer(model,strategy='rank_only'))
        a={};b={};reference={};seq=sequence(frames=12);other=sequence(seed=99,frames=12)
        outputs=[];per_frame=[]
        for (points,_),(other_points,_) in zip(seq,other):
            actual=handler(points,a);outputs.extend(actual);per_frame.append(actual)
            self.assertEqual(actual,handler(points,reference));handler(other_points,b)
        self.assertTrue(outputs)
        self.assertEqual(a['v2_tracker'].state,reference['v2_tracker'].state)
        fresh={}
        for i,(points,_) in enumerate(seq[:5]):self.assertEqual(handler(points,fresh),per_frame[i])
        self.assertIsNot(a['v2_point_history'],b['v2_point_history'])


if __name__=='__main__':unittest.main()
