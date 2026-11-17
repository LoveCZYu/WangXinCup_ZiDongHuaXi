from collections import Counter
from types import SimpleNamespace
import unittest
import numpy as np

from experiments.candidate_onset.policy import correction


def tracker():
    entries=[]
    for scan in range(1,5):
        entries.append((np.array([scan*.1,0.,0.],dtype=np.float32),None,scan,.99,.5))
    return SimpleNamespace(state={'tracks':{1:SimpleNamespace(position=(.5,1.,0.),velocity=(.1,.3,0.),last_scan=5,hits=8)}},
        emissions=Counter({1:6}),memories={1:SimpleNamespace(entries=entries)})


class CandidateRebaseTest(unittest.TestCase):
    def test_only_past_trusted_points_define_correction_at_last_physical_time(self):
        t=tracker();position,velocity=correction(t,1,6)
        np.testing.assert_allclose(position,[.5,0.,0.],atol=1e-7)
        np.testing.assert_allclose(velocity,[.1,0.,0.],atol=1e-7)
        self.assertEqual(t.state['tracks'][1].last_scan,5)
        self.assertEqual(t.state['tracks'][1].position,(.5,1.,0.))

    def test_fresh_trusted_state_is_unchanged_and_stale_memory_is_not_extrapolated(self):
        t=tracker();t.state['tracks'][1].last_scan=4
        self.assertIsNone(correction(t,1,5))
        t.state['tracks'][1].last_scan=5
        self.assertIsNone(correction(t,1,8))

    def test_immature_or_non_linear_memory_is_not_corrected(self):
        t=tracker();t.emissions[1]=2
        self.assertIsNone(correction(t,1,6))
        t.emissions[1]=6
        xyz,context,scan,score,margin=t.memories[1].entries[2]
        t.memories[1].entries[2]=(xyz+20,context,scan,score,margin)
        self.assertIsNone(correction(t,1,6))


if __name__=='__main__':unittest.main()
