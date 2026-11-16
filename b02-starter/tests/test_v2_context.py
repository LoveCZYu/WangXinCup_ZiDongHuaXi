import unittest
import numpy as np
from experiments.v2_association.context import descriptors
from experiments.v2_association.tracker import Tracker
from test_v2_tracker import sequence


class ContextTest(unittest.TestCase):
    def test_context_is_permutation_and_common_signal_shift_invariant(self):
        points,scores=sequence(frames=1)[0];original=descriptors(points,scores)
        reverse=descriptors(list(reversed(points)),list(reversed(scores)))[::-1]
        np.testing.assert_allclose(original,reverse,atol=1e-6)
        changed=[{**p,'Amp':p['Amp']+.12,'SNR':p['SNR']+.12,'PntID':p['PntID']+12345} for p in points]
        np.testing.assert_allclose(original,descriptors(changed,scores),atol=1e-6)

    def test_context_collection_preserves_v1_output_and_state(self):
        a=Tracker('v1');b=Tracker('v1',contextual=True,feature_capture=True,capture=True)
        for points,scores in sequence():
            self.assertEqual(a(points,scores),b(points,scores));self.assertEqual(a.state,b.state)
            for event in b.events:
                self.assertEqual(event['track_features'].shape,(60,))
                self.assertEqual(event['edge_features'].shape,(len(event['indices']),50))
        self.assertEqual(set(b.context_history),set(b.state['tracks']))


if __name__=='__main__':unittest.main()
