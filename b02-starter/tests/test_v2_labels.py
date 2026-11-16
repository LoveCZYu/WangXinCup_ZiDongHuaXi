import unittest
from experiments.v2_association.labels import continuity_owners


class ContinuityLabelTest(unittest.TestCase):
    def test_prior_emitted_owner_beats_duplicate_but_must_have_true_candidate(self):
        def event(batch,indices,age):
            return {'batch':batch,'indices':indices,'track_features':[0.]*35+[age]+[0.]*4}
        points=[{'PntID':10},{'PntID':11}];labels={10:7,11:0}
        events=[event(1,[0,1],.5),event(2,[0,1],1.),event(3,[0],3.),event(4,[0],2.)]
        histories={1:[7,7],2:[7,7],3:[7,8],4:[0,0]}
        counts={1:{7:30},2:{7:3},3:{7:100},4:{7:100}}
        self.assertEqual(continuity_owners(events,histories,counts,points,labels),{7:1})
        events[0]['indices']=[1]
        self.assertEqual(continuity_owners(events,histories,counts,points,labels),{7:2})


if __name__=='__main__':unittest.main()
