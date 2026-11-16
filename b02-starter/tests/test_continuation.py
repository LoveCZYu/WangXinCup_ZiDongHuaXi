import unittest

import numpy as np

from experiments.continuation.tracker import ContinuationTracker, DIMENSION
from experiments.prebirth.tracker import CONFIGS, PrebirthTracker
from test_id_recovery import StrongAffinity, point
from test_prebirth import RejectMature
from test_v2_tracker import sequence


class ContinuationTest(unittest.TestCase):
    def test_unknown_history_is_masked_and_impure_identity_is_negative(self):
        from experiments.continuation.learn import historical_target
        self.assertIsNone(historical_target(7,{-1:5}))
        self.assertIsNone(historical_target(-1,{7:5}))
        self.assertEqual(historical_target(7,{7:3,8:2})[0],0.)
        self.assertEqual(historical_target(7,{7:9,8:1,-1:4})[0],1.)
        self.assertEqual(historical_target(0,{7:9})[0],0.)

    def test_group_objective_ignores_unknown_and_padding_edges(self):
        import torch
        from experiments.continuation.learn import ContinuationModel,objective
        torch.manual_seed(17)
        model=ContinuationModel().eval()
        data={'features':torch.randn(3,12,DIMENSION),'valid':torch.zeros(3,12,dtype=torch.bool),
              'labels':torch.zeros(3,12),'priority':torch.ones(3,12)}
        data['valid'][:,0]=True;data['labels'][0,0]=1.
        original=objective(model,data)
        changed={k:v.clone() for k,v in data.items()}
        changed['features'][:,1:]=999.
        self.assertEqual(float(original.detach()),float(objective(model,changed).detach()))

    def test_wide_shadow_preserves_frozen_state_and_output(self):
        scorer=RejectMature()
        old=PrebirthTracker(scorer, StrongAffinity(), CONFIGS['short_wide'])
        new=ContinuationTracker(scorer, StrongAffinity())
        for points,scores in sequence(frames=32):
            self.assertEqual(old(points,scores),new(points,scores))
            self.assertEqual(old.state,new.state)

    def test_narrow_shadow_preserves_narrow_control(self):
        scorer=RejectMature()
        old=PrebirthTracker(scorer, StrongAffinity(), CONFIGS['short_relaxed'])
        new=ContinuationTracker(scorer, StrongAffinity(),'narrow_shadow')
        for points,scores in sequence(frames=28):
            self.assertEqual(old(points,scores),new(points,scores))

    def test_prefix_examples_are_finite_and_do_not_include_numeric_point_ids(self):
        a,b=ContinuationTracker(RejectMature(),StrongAffinity()),ContinuationTracker(RejectMature(),StrongAffinity())
        for scan in range(3):
            a([point(scan)],[.99]);b([{**point(scan),'PntID':scan+9999}],[.99])
        a.model_scorer.reject=b.model_scorer.reject=True
        found=False
        for scan in range(3,6):
            a([point(scan)],[.99]);b([{**point(scan),'PntID':scan+9999}],[.99])
            if a.samples:
                found=True
                self.assertEqual(a.samples[0]['features'].shape,(DIMENSION,))
                np.testing.assert_array_equal(a.samples[0]['features'],b.samples[0]['features'])
        self.assertTrue(found)

    def test_rejecting_model_keeps_new_target_confirmation_time(self):
        class Reject:
            def score(self,rows):return np.full(len(rows),-20.)
        scorer=RejectMature()
        tracker=ContinuationTracker(scorer,StrongAffinity(),'learned',Reject(),.8)
        for scan in range(3):out=tracker([point(scan)],[.99])
        old=out[0]['BatchID']
        scorer.reject=True
        for scan in range(3,6):out=tracker([point(scan)],[.99])
        self.assertEqual(out[0]['PntID'],5)
        self.assertNotEqual(out[0]['BatchID'],old)

    def test_memory_guard_without_rejections_matches_control(self):
        from experiments.continuation.memory_guard import MemoryGuardTracker
        scorer=RejectMature()
        a=ContinuationTracker(scorer,StrongAffinity())
        b=MemoryGuardTracker(scorer,StrongAffinity())
        for scan in range(14):
            self.assertEqual(a([point(scan)],[.99]),b([point(scan)],[.99]))
            self.assertEqual(a.state,b.state)
            for batch,memory in a.memories.items():
                self.assertEqual(len(memory.entries),len(b.memories[batch].entries))

    def test_memory_guard_quarantines_update_without_dropping_current_point(self):
        from experiments.continuation.memory_guard import MemoryGuardTracker
        class Weak:
            def score(self,rows):return np.full(len(rows),-8.)
        scorer=RejectMature()
        a=ContinuationTracker(scorer,Weak())
        b=MemoryGuardTracker(scorer,Weak())
        for scan in range(14):
            self.assertEqual(a([point(scan)],[.99]),b([point(scan)],[.99]))
        self.assertGreater(b.counters['memory_updates_blocked'],0)
        batch=next(iter(a.memories))
        self.assertLess(len(b.memories[batch].entries),len(a.memories[batch].entries))

    def test_zero_score_correction_and_combined_guard_preserve_frozen_control(self):
        from experiments.continuation.fusion import FusionTracker,MemoryFusionTracker
        class Head:
            def score(self,rows):return np.full(len(rows),-20.)
        scorer=RejectMature()
        a=ContinuationTracker(scorer,StrongAffinity())
        b=FusionTracker(scorer,StrongAffinity(),Head(),{'amount':0.,'threshold':.8})
        c=MemoryFusionTracker(scorer,StrongAffinity(),Head(),{'amount':0.,'threshold':.8},guard=.9)
        for scan in range(16):
            if scan==8:scorer.reject=True
            expected=a([point(scan)],[.99])
            self.assertEqual(expected,b([point(scan)],[.99]))
            self.assertEqual(expected,c([point(scan)],[.99]))
            self.assertEqual(a.state,b.state)
            self.assertEqual(a.state,c.state)


if __name__=='__main__':unittest.main()
