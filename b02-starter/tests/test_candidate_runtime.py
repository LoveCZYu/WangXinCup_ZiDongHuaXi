import unittest
from types import SimpleNamespace
import numpy as np
import torch
from experiments.candidate_onset_runtime import Handler


class CandidateRuntimeTest(unittest.TestCase):
    def test_packet_state_isolation_includes_point_history_and_track_identity(self):
        seen=[]
        def features(points,history):
            seen.append(len(history));return np.zeros((len(points),1),np.float32)
        point=SimpleNamespace(history=4,features=features,device='cpu',model=lambda x:torch.zeros(len(x)))
        class Tracker:
            def __init__(self):self.calls=0
            def __call__(self,points,scores):
                self.calls+=1;return [dict(PntID=points[0]['PntID'],BatchID=self.calls,score=scores[0])]
        handler=Handler(point,Tracker);a={};b={}
        self.assertEqual(handler([dict(PntID=1)],a)[0]['BatchID'],1)
        self.assertEqual(handler([dict(PntID=2)],a)[0]['BatchID'],2)
        self.assertEqual(handler([dict(PntID=3)],b)[0]['BatchID'],1)
        self.assertEqual(seen,[0,1,0])
        self.assertEqual(len(a['candidate_point_history']),2)
        self.assertEqual(len(b['candidate_point_history']),1)


if __name__=='__main__':unittest.main()
