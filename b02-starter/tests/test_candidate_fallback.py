from collections import Counter,defaultdict
from math import floor
from types import SimpleNamespace
import unittest
import numpy as np

from b02.algorithm import DEFAULT_CONFIG,Track
from experiments.candidate_onset.fallback import BackupMixin,BackupScorer


def event(indices):
    return dict(indices=indices,track_features=np.zeros(98,np.float32),edge_features=np.zeros((len(indices),85),np.float32),
        base_costs=[.1]*len(indices),dummy_cost=1.15)


class CandidateFallbackTest(unittest.TestCase):
    def test_guard_accepts_strong_unique_identity_and_uses_correct_cost_scale(self):
        prior=SimpleNamespace(model=None)
        affinity=SimpleNamespace(score=lambda x:np.asarray([5.,.5],np.float32))
        e=event([0,1]);delta,null=BackupScorer(prior,affinity)([e])[0]
        self.assertLess(.1+.25*delta[0],0.)
        self.assertGreater(.1+.25*delta[1],0.)
        self.assertAlmostEqual(1.15+.25*null,0.)

    def test_guard_refuses_cross_id_ambiguity(self):
        prior=SimpleNamespace(model=None)
        affinity=SimpleNamespace(score=lambda x:np.asarray([5.,4.9],np.float32))
        result=BackupScorer(prior,affinity)([event([0]),event([0])])
        self.assertTrue(all(.1+.25*row[0][0]>0 for row in result))

    def harness(self):
        class Parent:
            def assign(self,track_ids,scan,positions,grid,occupied,points,scores):
                occupied.add(0)
                self.events=[dict(batch=1,indices=[]),dict(batch=2,indices=[0])]
                return {2:0}
        class Harness(BackupMixin,Parent):pass
        t=Harness();t.assignment_stage=0;t.backup_mode='backup_motion';t.motion_counts=Counter()
        t.state={'tracks':{1:Track((.5,1.,0.),5,velocity=(.1,.3,0.),hits=8),2:Track((0.,0.,0.),5,hits=8)}}
        t.memories={1:SimpleNamespace(entries=[(np.array([s*.1,0.,0.]),None,s,.99,.5) for s in range(1,5)])}
        t.emissions=Counter({1:6,2:6});t.config=DEFAULT_CONFIG;t.strong_hits=Counter({1:6})
        t.gate=SimpleNamespace(policy=SimpleNamespace(enter=.525));t.model_scorer=SimpleNamespace(model=None)
        t.scorer=None;t.feature_capture=False;t.capture=True
        t.variant='v1'
        return t

    def test_actual_matching_keeps_original_match_and_adds_only_free_point(self):
        t=self.harness();positions=[(.6,.3,0.),(.6,.3,0.)];grid=defaultdict(list)
        for i,pos in enumerate(positions):grid[tuple(map(floor,pos))].append(i)
        occupied=set();result=t.assign([1,2],6,positions,grid,occupied,[{},{}],[.99,.99])
        self.assertEqual(result,{2:0,1:1})
        self.assertEqual(occupied,{0,1})
        self.assertEqual(t.state['tracks'][1].last_scan,5)

    def test_no_accepted_backup_restores_original_physical_position(self):
        t=self.harness();positions=[(.6,.3,0.)];grid={(0,0,0):[0]};occupied=set()
        result=t.assign([1,2],6,positions,grid,occupied,[{}],[.99])
        self.assertEqual(result,{2:0})
        self.assertEqual(t.state['tracks'][1].position,(.5,1.,0.))
        self.assertEqual(occupied,{0})


if __name__=='__main__':unittest.main()
