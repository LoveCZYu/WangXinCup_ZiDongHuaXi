import unittest

import numpy as np

from experiments.candidate_onset.harmful_update import dominant_prefix, reference_features


class HarmfulUpdateTest(unittest.TestCase):
    def test_dominant_prefix_tolerates_one_bad_recent_identity(self):
        self.assertEqual(dominant_prefix([31,32,31,31,31,31,31,31]),31)
        self.assertIsNone(dominant_prefix([31,32,31,32,31,31,32,31]))
        self.assertIsNone(dominant_prefix([31]*5))

    def test_robust_reference_supports_irregular_history_times(self):
        def point(x):
            return dict(Azimuth=x,Elevation=0.,Distance=0.,Amp=.7,SNR=.4)

        history=[(point(t*.025),.9,t) for t in (1,3,4,7)]
        result=reference_features(history,point(.2),point(.25),8)
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(result[0],0.,places=5)
        self.assertGreater(result[1],result[0])


if __name__=='__main__':
    unittest.main()
