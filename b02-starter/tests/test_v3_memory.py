import copy
import unittest
import numpy as np
from experiments.v3_tracking.memory import Memory,TRACK_DIM,EDGE_DIM,reliable


class MemoryFeatureTest(unittest.TestCase):
    def test_empty_and_bounded_trusted_history(self):
        m=Memory();track=np.zeros(60,dtype=np.float32);track[38]=.25
        edges=np.zeros((2,50),dtype=np.float32);context=np.zeros((2,10),dtype=np.float32)
        tx,ex=m.augment(track,edges,context,1)
        self.assertEqual(tx.shape,(TRACK_DIM,));self.assertEqual(ex.shape,(2,EDGE_DIM));self.assertEqual(tx[-1],0)
        for scan in range(30):
            point=dict(Azimuth=scan*.001,Elevation=.2,Distance=.3)
            m.add(point,np.full(10,scan/100,dtype=np.float32),scan,.9,.5)
        self.assertEqual(len(m.entries),16);self.assertEqual(m.anchor[2],0)
        a,b=m.augment(track,edges,context,31);self.assertEqual(a[-1],1)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())

    def test_high_point_score_alone_does_not_make_association_reliable(self):
        event={'indices':[1,2],'base_costs':[.2,.21],'dummy_cost':1.15,
               'edge_features':np.zeros((2,50),dtype=np.float32)}
        self.assertFalse(reliable(event,1,.99,4)[0])
        event['base_costs']=[.2,.7]
        self.assertTrue(reliable(event,1,.99,4)[0])
        self.assertFalse(reliable(event,1,.1,4)[0])

    def test_memory_capture_preserves_frozen_v2_prefix_and_earlier_features(self):
        import torch
        from experiments.v2_association.model import AssociationModel,Scorer
        from experiments.v2_association.tracker import Tracker as V2
        from experiments.v3_tracking.tracker import Tracker
        from test_v2_tracker import sequence
        torch.manual_seed(13);model=AssociationModel(60,50)
        with torch.no_grad():model.edge[-1].weight.normal_(0,.1)
        a=V2('v1',contextual=True,scorer=Scorer(copy.deepcopy(model),strategy='rank_only'))
        b=Tracker(Scorer(model,strategy='rank_only'),memory=True,feature_capture=True)
        earlier=None
        for points,scores in sequence(frames=16):
            self.assertEqual(a(points,scores),b(points,scores));self.assertEqual(a.state,b.state)
            for event in b.events:
                if event['indices']:self.assertEqual(event['track_features'].shape,(98,))
            if earlier is None and b.events:
                earlier=copy.deepcopy(b.events)
                saved=b.events
        if earlier is not None:
            for old,held in zip(earlier,saved):
                for key in ('track_features','edge_features'):
                    np.testing.assert_array_equal(old[key],held[key])


if __name__=='__main__':unittest.main()
