from collections import Counter
import unittest
from tests.test_candidate_onset import tracker
from experiments.candidate_onset.outcomes import OutcomeAudit


class CandidateOutcomesTest(unittest.TestCase):
    def setup_audit(self):
        audit=OutcomeAudit({7:10,8:10});audit.history[1]=Counter({7:3});audit.last_truth_output[1][7]=0
        return audit

    def test_short_other_id_segment_is_retained_after_original_id_recovers(self):
        a=self.setup_audit();t=tracker();before={1:dict(bank='active',emissions=3,last_scan=0)}
        points=[dict(PntID=10),dict(PntID=11)];labels={10:7,11:8}
        for frame in (1,2):a.observe(frame,frame,points,labels,[dict(PntID=10,BatchID=2)],t,before,[.9,.9])
        a.observe(3,3,points,labels,[dict(PntID=10,BatchID=1)],t,before,[.9,.9]);a.finalize()
        self.assertEqual(a.events[0]['other_id_points_before_recovery'],2)
        self.assertFalse(a.events[0]['replacement_three_plus'])
        self.assertTrue(a.events[0]['recovered'])
        self.assertEqual(a.counts['visible_any_other_id'],1)

    def test_persistent_replacement_is_counted_once(self):
        a=self.setup_audit();t=tracker();before={1:dict(bank='active',emissions=3,last_scan=0)}
        points=[dict(PntID=10),dict(PntID=11)];labels={10:7,11:8}
        for frame in (1,2,3):a.observe(frame,frame,points,labels,[dict(PntID=10,BatchID=2)],t,before,[.9,.9])
        a.finalize()
        self.assertEqual(a.counts['visible_replacement_three_plus'],1)
        self.assertFalse(a.events[0]['recovered'])
        self.assertTrue(a.events[0]['had_later_truth_observation'])


if __name__=='__main__':unittest.main()
