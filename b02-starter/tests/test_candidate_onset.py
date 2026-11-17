from collections import Counter
from types import SimpleNamespace
import unittest
import numpy as np

from experiments.candidate_onset.audit import Audit,classify


def tracker():
    event=dict(batch=1,indices=[1],gated_indices=[1],eligible_indices=[1],track_features=np.zeros(98))
    return SimpleNamespace(events=[event],retired_edges=[],proposals=[],last_matches={1:1},trusted_updates={},
        stage_records=[dict(track_ids=[1],occupied_before=[])],retired_calls=[],state={'scan':2},
        gate=SimpleNamespace(policy=SimpleNamespace(enter=.525)))


class CandidateOnsetTest(unittest.TestCase):
    def test_classification_distinguishes_occupancy_cost_and_topk(self):
        t=tracker();before={1:dict(bank='active')}
        self.assertEqual(classify(1,[0],before,t,[.9,.9])[0],'main_motion_gate')
        t.stage_records[0]['occupied_before']=[0]
        self.assertEqual(classify(1,[0],before,t,[.9,.9])[0],'prior_stage_occupied')
        t.stage_records[0]['occupied_before']=[]
        t.events[0]['gated_indices'].append(0)
        self.assertEqual(classify(1,[0],before,t,[.9,.9])[0],'main_base_cost')
        t.events[0]['eligible_indices'].append(0)
        self.assertEqual(classify(1,[0],before,t,[.9,.9])[0],'main_topk')
        t.last_matches[1]=0
        self.assertEqual(classify(1,[0],before,t,[.9,.9])[0],'output_gate')

    def test_repeated_failed_frames_are_one_episode_and_purity_cannot_erase_it(self):
        a=Audit({7:10,8:10});a.history[1]=Counter({7:3});a.last_truth_output[1][7]=0
        t=tracker();points=[dict(PntID=10),dict(PntID=11)];labels={10:7,11:8}
        before={1:dict(bank='active',emissions=3,last_scan=0)}
        first=a.observe(1,1,points,labels,[dict(PntID=11,BatchID=1)],t,before,[.9,.9])
        self.assertEqual(len(first),1)
        self.assertTrue(first[0]['prior_sampling_rule_eligible'])
        self.assertEqual(first[0]['intervening_internal_updates'],{})
        self.assertEqual(a.observe(2,2,points,labels,[],t,before,[.9,.9]),[])
        self.assertEqual(a.counts['first_visible_failures'],1)
        a.observe(3,3,points,labels,[dict(PntID=10,BatchID=1)],t,before,[.9,.9])
        self.assertEqual(a.counts['episodes_recovered'],1)
        self.assertEqual(len(a.observe(4,4,points,labels,[],t,before,[.9,.9])),1)

    def test_true_end_of_observations_is_not_a_new_failed_episode(self):
        a=Audit({7:0});a.history[1]=Counter({7:3});a.last_truth_output[1][7]=0
        t=tracker();t.last_matches={}
        self.assertEqual(a.observe(1,1,[],{},[],t,{},[]),[])
        self.assertEqual(a.counts['episodes_started'],0)

    def test_anchor_uses_the_last_correct_output_not_the_last_other_identity(self):
        a=Audit({7:10,8:10});a.history[1]=Counter({7:4,8:1})
        a.last_truth_output[1].update({7:2,8:3})
        t=tracker();before={1:dict(bank='active',emissions=5,last_scan=3)}
        result=a.observe(4,4,[dict(PntID=10),dict(PntID=11)],{10:7,11:8},[],t,before,[.9,.9])
        self.assertEqual(result[0]['gap_since_good'],2)


if __name__=='__main__':unittest.main()
