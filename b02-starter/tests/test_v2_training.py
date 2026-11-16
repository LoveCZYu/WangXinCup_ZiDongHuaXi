import unittest
import numpy as np
from experiments.v2_association.train import select_objective


class PairObjectiveTest(unittest.TestCase):
    def test_pair_ranking_excludes_null_rows_without_training_dummy(self):
        groups={'track':np.zeros((3,40),dtype=np.float32),'edges':np.zeros((3,12,20),dtype=np.float32),
                'base':np.ones((3,13),dtype=np.float32),'valid':np.zeros((3,13),dtype=bool),
                'target':np.zeros((3,13),dtype=bool)}
        groups['valid'][:,:2]=True;groups['valid'][:,-1]=True
        groups['valid'][2,1]=False
        groups['target'][0,0]=True;groups['target'][1,-1]=True;groups['target'][2,0]=True
        selected=select_objective(groups,'pair_ranking')
        self.assertEqual(selected['track'].shape,(1,40))
        self.assertFalse(selected['valid'][0,-1])
        self.assertTrue(groups['valid'][0,-1])
        self.assertTrue((selected['target']<=selected['valid']).all())


if __name__=='__main__':unittest.main()
