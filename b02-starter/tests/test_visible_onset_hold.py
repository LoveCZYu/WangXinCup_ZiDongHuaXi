import unittest
from types import SimpleNamespace

import numpy as np

from experiments.candidate_onset.visible_onset_hold import eligible_ids, label_update, proposals


class VisibleOnsetHoldTest(unittest.TestCase):
    def test_an_observation_gap_also_prevents_a_consecutive_hold(self):
        tracker = SimpleNamespace(state={'scan':11,'tracks':{1:SimpleNamespace(last_scan=11), 2:SimpleNamespace(last_scan=10)}},
                                  emissions={1:5,2:5}, history={1:[0]*4,2:[0]*4})
        self.assertEqual(eligible_ids(tracker), {1})

    def test_requires_a_second_strong_admitted_observation(self):
        points = [{'PntID':1},{'PntID':2}]
        output = [{'BatchID':7,'PntID':1}]
        features = np.zeros((2,85))
        features[1,9] = .3
        row = dict(batch=7, indices=[0,1], track_features=np.zeros(98), edge_features=features,
                   effective_costs=[.1,.2])
        tracker = SimpleNamespace(last_matches={7:0},events=[row],
                                  affinity=SimpleNamespace(score=lambda packed:np.ones(len(packed))))
        self.assertEqual(proposals(tracker,points,[.9,.5],output,{7}), [])
        found, = proposals(tracker,points,[.9,.9],output,{7})
        self.assertEqual(found[:4], (7,0,1,(1,)))
        features[1,9] = .8
        self.assertEqual(proposals(tracker,points,[.9,.9],output,{7}), [])

    def test_absence_and_a_true_admitted_candidate_are_distinct_audit_labels(self):
        points = [{'PntID':1},{'PntID':2}]
        candidate = (7,0,1,(1,),None)
        history = {7:[3]*6}
        self.assertEqual(label_update(candidate,points,{1:4,2:3},history),'visible_supported')
        self.assertEqual(label_update(candidate,points,{1:4,2:5},history),'original_absent')


if __name__ == '__main__':
    unittest.main()
