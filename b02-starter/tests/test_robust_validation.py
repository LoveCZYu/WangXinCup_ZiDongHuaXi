import unittest
from collections import Counter

import numpy as np

from experiments.robust_validation.frozen import Probe
from experiments.robust_validation.oracles import graph_repair,perfect_identity,shadow_identity
from experiments.continuation.fusion import MemoryFusionTracker
from test_id_recovery import StrongAffinity,point
from test_prebirth import RejectMature


class RobustValidationTest(unittest.TestCase):
    def test_sufficient_statistics_match_existing_scorecard_and_paired_null(self):
        from b02.scorecard import build_scorecard
        from experiments.robust_validation.uncertainty import sufficient,metrics,paired_bootstrap
        scene={'truth_points':10,'matched_target_points':6,'clutter_outputs':1,'unknown_outputs':0,
               'per_target':[{'truth_id':7,'truth_points':4,'matched_points':3,'output_ids':{'1':2,'2':1},
                              'most_matched_points_with_one_id':2,'first_output_delay_circles':2},
                             {'truth_id':8,'truth_points':6,'matched_points':3,'output_ids':{'3':3},
                              'most_matched_points_with_one_id':3,'first_output_delay_circles':4}]}
        values=sufficient(scene,{'mixed_minority_truth_points':1})
        actual=metrics(values);expected=build_scorecard({'scenes':{'toy':scene}},'toy')['overall']['values']
        for key in actual:
            if key in expected:self.assertAlmostEqual(float(actual[key]),expected[key])
        ci=paired_bootstrap(np.stack([values,values]),np.stack([values,values]),count=100)
        self.assertTrue(all(row=={'lower':0.,'upper':0.} for row in ci.values()))

    def test_perfect_identity_preserves_points_and_allocates_same_frame_slots(self):
        outputs=[[{'PntID':1,'BatchID':8},{'PntID':2,'BatchID':9}],
                 [{'PntID':3,'BatchID':9},{'PntID':4,'BatchID':8}]]
        frames=[(0,[],{1:7,2:7}),(1,[],{3:7,4:0})]
        result,counts=perfect_identity(outputs,frames)
        self.assertEqual([[p['PntID'] for p in frame] for frame in result],[[1,2],[3,4]])
        self.assertEqual(result[0][0]['BatchID'],result[1][0]['BatchID'])
        self.assertNotEqual(result[0][0]['BatchID'],result[0][1]['BatchID'])
        self.assertEqual(counts['same_truth_same_frame_overflows'],1)

    def test_graph_repair_resolves_a_feasible_swap(self):
        outputs=[[{'PntID':1,'BatchID':1},{'PntID':2,'BatchID':2}],
                 [{'PntID':3,'BatchID':2},{'PntID':4,'BatchID':1}],
                 [{'PntID':5,'BatchID':1},{'PntID':6,'BatchID':2}]]
        frames=[(0,[],{1:7,2:8}),(1,[],{3:7,4:8}),(2,[],{5:7,6:8})]
        graphs=[[{1},{2}],[{1,2},{1,2}],[{1},{2}]]
        result,counts=graph_repair(outputs,frames,graphs)
        self.assertEqual([p['BatchID'] for p in result[1]],[1,2])
        self.assertEqual(counts['changed_points'],2)

    def test_graph_repair_cannot_steal_id_from_fixed_unscored_point(self):
        outputs=[[{'PntID':1,'BatchID':1}],[{'PntID':2,'BatchID':1}],
                 [{'PntID':3,'BatchID':2},{'PntID':4,'BatchID':1}]]
        frames=[(0,[],{1:7}),(1,[],{2:7}),(2,[],{3:7,4:0})]
        result,counts=graph_repair(outputs,frames,[[{1}],[{1}],[{1,2},{1}]])
        self.assertEqual(result,outputs)
        self.assertEqual(counts['collision_blocks'],1)

    def test_one_to_one_shadow_penalizes_one_id_covering_two_targets(self):
        summary={'truth_points':4,'matched_target_points':4,'per_target':[
            {'output_ids':{'1':2}},{'output_ids':{'1':2}}]}
        result=shadow_identity(summary)
        self.assertEqual(result['one_to_one_correct_points'],2)
        self.assertEqual(result['mixed_minority_points'],2)

    def test_retired_probe_preserves_outputs_and_records_accepted_edge(self):
        class Head:
            def score(self,rows):return np.zeros(len(rows))
        class Probed(Probe,MemoryFusionTracker):pass
        args=(RejectMature(),StrongAffinity(),Head(),{'amount':0.,'threshold':.8})
        plain=MemoryFusionTracker(*args,guard=.9);probed=Probed(*args,guard=.9)
        for scan in range(12):self.assertEqual(plain([point(scan)],[.99]),probed([point(scan)],[.99]))
        for _ in range(5):self.assertEqual(plain([],[]),probed([],[]))
        expected=plain([point(17)],[.99]);actual=probed([point(17)],[.99])
        self.assertEqual(expected,actual)
        self.assertIn((actual[0]['BatchID'],0),probed.retired_edges)
        self.assertEqual(plain.state,probed.state)


if __name__=='__main__':unittest.main()
