import copy
import random
import unittest

from experiments.association.interventions import Tracker as FrozenTracker
from experiments.v2_association.tracker import Tracker, VARIANTS
from experiments.v2_association.features import TRACK_DIM, EDGE_DIM
from experiments.point_scorer_expanded.policy import Policy


def sequence(seed=17,frames=55):
    rng=random.Random(seed);seq=[]
    for scan in range(frames):
        points=[];scores=[]
        for i in range(24):
            if scan%11==7 and i<3:continue
            x=.35+.0009*scan+.012*i if i<4 else rng.random()
            points.append(dict(Time=scan*5000+i,Azimuth=x,Elevation=.3 if i<4 else rng.random(),
                               Distance=.4 if i<4 else rng.random(),Amp=rng.random(),SNR=rng.random(),
                               CircleID=scan,PntID=scan*100+i))
            scores.append((1.,.51,.04,.6)[i%4])
        if scan in (18,19,20,21):points=[];scores=[]
        seq.append((points,scores))
    return seq


class V2TrackerTest(unittest.TestCase):
    def test_v1_backend_is_exactly_frozen_tracker(self):
        for seed in (17,51):
            frozen=FrozenTracker('combined',Policy('frozen_v1',.525,.05,3))
            new=Tracker('v1',capture=True,feature_capture=True)
            for points,scores in sequence(seed):
                self.assertEqual(new(points,scores),frozen(points,scores))
                self.assertEqual(new.state,frozen.state)
                self.assertEqual(new.strong_hits,frozen.strong_hits)
                self.assertEqual(new.gate.last_strong,frozen.gate.last_strong)
                for event in new.events:
                    self.assertEqual(event['track_features'].shape,(TRACK_DIM,))
                    self.assertEqual(event['edge_features'].shape,(len(event['indices']),EDGE_DIM))

    def test_variants_are_causal_resettable_and_preserve_original_points(self):
        seq=sequence()
        for variant in VARIANTS:
            a=Tracker(variant);b=Tracker(variant)
            prefix=[]
            for points,scores in seq[:24]:
                original=copy.deepcopy(points);result=a(points,scores);prefix.append(result)
                self.assertEqual(points,original)
                self.assertEqual(len({p['BatchID'] for p in result}),len(result))
                for output in result:
                    source=next(p for p in points if p['PntID']==output['PntID'])
                    self.assertEqual({k:v for k,v in output.items() if k!='BatchID'},
                                     {k:source[k] for k in output if k!='BatchID'})
            for i,(points,scores) in enumerate(seq):
                result=b(points,scores)
                if i<24:self.assertEqual(result,prefix[i])

    def test_position_trust_changes_internal_position_without_editing_input(self):
        a=Tracker('v1');b=Tracker('position_trust')
        for points,scores in sequence(frames=4):
            a(points,scores);b(points,scores)
        common=set(a.state['tracks']) & set(b.state['tracks'])
        self.assertTrue(any(a.state['tracks'][k].position!=b.state['tracks'][k].position for k in common))

    def test_features_ignore_numeric_ids_and_absolute_time(self):
        import numpy as np
        a=Tracker('v1',capture=True,feature_capture=True)
        b=Tracker('v1',capture=True,feature_capture=True)
        for points,scores in sequence(frames=12):
            changed=copy.deepcopy(points)
            for p in changed:p.update(PntID=p['PntID']+100000,Time=p['Time']+1234567,CircleID=p['CircleID']+1000)
            a(points,scores);b(changed,scores)
            self.assertEqual(len(a.events),len(b.events))
            for x,y in zip(a.events,b.events,strict=True):
                np.testing.assert_array_equal(x['track_features'],y['track_features'])
                np.testing.assert_array_equal(x['edge_features'],y['edge_features'])


if __name__=='__main__':unittest.main()
