import json
import unittest
from b02.algorithm import _output
from b02.replay import replay_frames
from experiments.v2_association.run_stage1 import Discard
from experiments.v3_tracking.diagnose import reference_equal


class StoredReferenceTest(unittest.TestCase):
    def test_real_replay_summary_matches_its_json_stored_reference(self):
        frames=[]
        for circle in range(3):
            point=dict(Time=circle*5000,Azimuth=.1,Elevation=.2,Distance=.3,Amp=.4,SNR=.4,CircleID=circle,PntID=circle+1)
            frames.append((circle,[point],{circle+1:7}))
        stats=replay_frames(frames,Discard(),lambda points,state:[_output(points[0],42)])
        stored=json.loads(json.dumps(stats))
        self.assertTrue(reference_equal(stats,stored))
        stored['per_target'][0]['output_ids']={'42':2,'43':1}
        self.assertFalse(reference_equal(stats,stored))


if __name__=='__main__':unittest.main()
