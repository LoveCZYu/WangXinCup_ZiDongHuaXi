import copy
import unittest
from experiments.v2_association.context import descriptors
from experiments.v3_tracking.identity import IdentityManager
from test_v2_tracker import sequence


class IdentityManagerTest(unittest.TestCase):
    def test_original_point_set_prefix_and_state_reset(self):
        a=IdentityManager();b=IdentityManager();saved=[];held=None
        seq=sequence(frames=16)
        for i,(points,scores) in enumerate(seq):
            # Arbitrary upstream ID changes must not create logical ID changes.
            chosen=[{k:v for k,v in p.items() if k in ('Time','Azimuth','Elevation','Distance','CircleID','PntID')}|{'BatchID':i*10+j+1} for j,p in enumerate(points[:3])]
            original=copy.deepcopy(chosen);ctx=descriptors(points,scores)
            output=a(points,scores,chosen,ctx);saved.append(output)
            self.assertEqual(chosen,original)
            self.assertEqual([{k:v for k,v in p.items() if k!='BatchID'} for p in output],[{k:v for k,v in p.items() if k!='BatchID'} for p in chosen])
            self.assertEqual(len({p['BatchID'] for p in output}),len(output))
            if held is None:held=(output,copy.deepcopy(output))
            if i<8:self.assertEqual(output,b(points,scores,chosen,ctx))
        self.assertEqual(*held)
        fresh=IdentityManager()
        for i,(points,scores) in enumerate(seq[:8]):
            chosen=[{k:v for k,v in p.items() if k in ('Time','Azimuth','Elevation','Distance','CircleID','PntID')}|{'BatchID':999+j} for j,p in enumerate(points[:3])]
            self.assertEqual(fresh(points,scores,chosen,descriptors(points,scores)),saved[i])

    def test_empty_output_does_not_invent_or_backfill_points(self):
        manager=IdentityManager(max_gap=8)
        for points,scores in sequence(frames=3):self.assertEqual(manager(points,scores,[],descriptors(points,scores)),[])
        self.assertEqual(manager.engine.state['tracks'],{})


if __name__=='__main__':unittest.main()
