from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.v3_tracking.train import load_groups


class PairDatasetTest(unittest.TestCase):
    def test_unknown_history_and_pure_vs_anchor_labels_stay_separate(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);valid=np.zeros((3,13),dtype=bool);valid[:,:2]=True;valid[:,-1]=True
            target=np.zeros((3,13),dtype=bool);target[0,-1]=True;target[1,0]=True;target[2,1]=True
            anchor=target.copy();anchor[0]=False;anchor[0,1]=True
            np.savez(root/'x.npz',track=np.zeros((3,60),dtype=np.float32),edges=np.zeros((3,12,50),dtype=np.float32),
                memory_track=np.zeros((3,98),dtype=np.float32),memory_edges=np.zeros((3,12,85),dtype=np.float32),
                base=np.ones((3,13),dtype=np.float32),valid=valid,target=target,known=np.array([False,True,False]),
                anchor_target=anchor,anchor_known=np.array([True,True,False]))
            pure,counts=load_groups(root,['x'])
            self.assertEqual(pure['track'].shape,(1,60));self.assertEqual(counts['known'],1)
            long,counts=load_groups(root,['x'],True,'anchor')
            self.assertEqual(long['track'].shape,(2,98));self.assertEqual(long['edges'].shape,(2,12,85))
            self.assertFalse(long['valid'][:,-1].any());self.assertTrue((long['target']<=long['valid']).all())


if __name__=='__main__':unittest.main()
