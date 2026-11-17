from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.competition_learning.model import Model, set_loss
from experiments.id_recovery.affinity import TRACK
from experiments.set_association.model import Identity, Rejection, full_pairs, geometry


class SetAssociationTest(unittest.TestCase):
    def test_full_history_reconstruction_and_function_preserving_start(self):
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        rows = rng.normal(size=(20, 101)).astype(np.float32)
        pair = rng.normal(size=(20, 96)).astype(np.float32)
        pair[:, :8] = rows[:, list(TRACK)]
        pair[:, 93:] = rows[:, 98:]
        expanded = full_pairs(pair, rows)
        prior = Model().eval()
        for full in (False, True):
            model = Identity(full).eval()
            x = torch.from_numpy(expanded if full else pair)
            model.initialize(prior, x)
            torch.testing.assert_close(model.edges(x), prior.edges(torch.from_numpy(pair)), atol=1e-6, rtol=1e-5)
        broken = pair.copy();broken[0, 0] += 1
        with self.assertRaises(AssertionError):full_pairs(broken, rows)

    def test_set_rejection_is_invariant_to_order_padding_and_uses_candidates(self):
        torch.manual_seed(42)
        model = Rejection(True).eval()
        # Test the function after learning, rather than the identically zero start.
        torch.nn.init.normal_(model.null[-1].weight)
        rows, edges = torch.randn(2, 101), torch.randn(2, 3, 96)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        expected = model(rows, edges, mask)
        perm = [2, 0, 1]
        torch.testing.assert_close(expected, model(rows, edges[:, perm], mask[:, perm]))
        padding = edges.clone();padding[~mask] = 500
        torch.testing.assert_close(expected, model(rows, padding, mask))
        changed = edges.clone();changed[mask] += 2
        self.assertFalse(torch.allclose(expected, model(rows, changed, mask)))
        self.assertTrue(torch.isfinite(model(rows, edges, torch.zeros_like(mask))).all())

    def test_label_visibility_cannot_change_candidate_features(self):
        d = {'edge_row':np.array([0, 0, 1]),'row':np.zeros((2,101)),
            'row_index':np.zeros((2, 12), int),'edge_known':np.array([True,False,True]),
            'row_allowed':np.array([True,True]),'edge_positive':np.array([True,False,False])}
        index, mask, supervised, _ = geometry(d)
        self.assertEqual(mask.sum(1).tolist(), [2,1])
        self.assertEqual(supervised.tolist(), [False,True])
        d['edge_known'][:] = True
        second_index, second_mask, _, _ = geometry(d)
        np.testing.assert_array_equal(index,second_index)
        np.testing.assert_array_equal(mask,second_mask)

    def test_rejection_training_does_not_change_pair_ranking(self):
        torch.manual_seed(42)
        prior = Model().eval()
        pair = torch.randn(2,1,96)
        fixed = prior.edges(pair).detach()
        rejection = Rejection(True)
        rejection.initialize(prior,torch.randn(10,101))
        rows, mask = torch.randn(2,101),torch.ones(2,1,dtype=torch.bool)
        optimizer = torch.optim.SGD(rejection.parameters(),lr=.2)
        for _ in range(4):
            null = rejection(rows,pair,mask)
            logits = torch.cat((fixed,null[:,None]),1)
            loss = set_loss(logits,torch.ones(2,2,dtype=torch.bool),
                torch.tensor([[False,True],[False,True]]))
            optimizer.zero_grad();loss.backward();optimizer.step()
        torch.testing.assert_close(prior.edges(pair),fixed)
        self.assertTrue(all(p.grad is None for p in prior.parameters()))
        self.assertTrue((rejection(rows,pair,mask) > 0).all())


if __name__ == '__main__':unittest.main()
