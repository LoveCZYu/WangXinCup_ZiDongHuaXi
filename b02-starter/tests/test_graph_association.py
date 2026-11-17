from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.competition_learning.model import Model
from experiments.competition_learning.samples import grouped_index
from experiments.graph_association.data import Tables,neighborhoods
from experiments.graph_association.model import Association,pool
from experiments.graph_association.runtime import GraphScorer


class GraphAssociationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.prior = Model().eval()
        self.pair,self.rows = torch.randn(4,96),torch.randn(3,101)
        self.er,self.ec = torch.tensor([0,0,1,2]),torch.tensor([0,1,0,2])

    def model(self,context,learned=False):
        model = Association(deepcopy(self.prior),context).eval()
        model.initialize(self.rows)
        if learned:
            torch.nn.init.normal_(model.residual[-1].weight,std=.1)
            torch.nn.init.normal_(model.null[-1].weight,std=.1)
        return model

    def test_both_arms_preserve_initial_pair_function_and_zero_null(self):
        for context in (False,True):
            model = self.model(context)
            pair,null = model.graph(self.pair,self.rows,self.er,self.ec)
            torch.testing.assert_close(pair,self.prior.edges(self.pair))
            torch.testing.assert_close(null,torch.zeros(3))
            self.assertTrue(all(not p.requires_grad for p in model.prior.parameters()))

    def test_cached_int32_group_indices_are_supported(self):
        model = self.model(True,True)
        pair,null = model.graph(self.pair,self.rows,self.er,self.ec)
        actual,n = model.graph(self.pair,self.rows,self.er.int(),self.ec.int())
        torch.testing.assert_close(pair,actual)
        torch.testing.assert_close(null,n)

    def test_sampled_neighbors_match_full_graph_and_ignore_padding(self):
        model = self.model(True,True)
        ri,rm = grouped_index(self.er.numpy(),3,3)
        ci,cm = grouped_index(self.ec.numpy(),3,3)
        rp,cp = self.pair[ri[self.er]],self.pair[ci[self.ec]]
        pair = model.edges(self.pair,rp,torch.from_numpy(rm[self.er]),cp,torch.from_numpy(cm[self.ec]))
        null = model.rejection(self.rows,self.pair[ri],torch.from_numpy(rm))
        gp,gn = model.graph(self.pair,self.rows,self.er,self.ec)
        torch.testing.assert_close(pair,gp,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(null,gn,atol=1e-6,rtol=1e-5)
        rp = rp.clone();rp[~torch.from_numpy(rm[self.er])] = 1000
        padded = model.edges(self.pair,rp,torch.from_numpy(rm[self.er]),cp,torch.from_numpy(cm[self.ec]))
        torch.testing.assert_close(pair,padded)

    def test_context_scores_are_edge_permutation_equivariant_and_see_competition(self):
        model = self.model(True,True)
        expected,null = model.graph(self.pair,self.rows,self.er,self.ec)
        permutation = torch.tensor([2,0,3,1])
        changed,n = model.graph(self.pair[permutation],self.rows,self.er[permutation],self.ec[permutation])
        torch.testing.assert_close(changed,expected[permutation],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(null,n,atol=1e-6,rtol=1e-5)
        no_competition,_ = model.graph(self.pair,self.rows,self.er,torch.arange(4))
        self.assertFalse(torch.allclose(expected,no_competition))
        control = self.model(False,True)
        a,_ = control.graph(self.pair,self.rows,self.er,self.ec)
        b,_ = control.graph(self.pair,self.rows,self.er,torch.arange(4))
        torch.testing.assert_close(a,b)

    def test_unknown_labels_only_change_loss_eligibility_not_neighborhoods(self):
        d = dict(edge_row=self.er.numpy(),edge_col=self.ec.numpy(),row=self.rows.numpy(),
            pair=self.pair.numpy(),row_index=np.zeros((3,12),int),edge_known=np.array([True,False,True,True]),
            row_allowed=np.ones(3,bool),col_allowed=np.ones(3,bool),edge_positive=np.array([True,False,False,True]))
        a = neighborhoods(d)
        d['edge_known'][:] = True
        b = neighborhoods(d)
        for key in ('ri','rm','ci','cm'):np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(a['rm'].sum(1).tolist(),[2,1,1])
        self.assertNotIn(0,a['positive_rows'])
        self.assertIn(0,b['positive_rows'])

    def test_runtime_retains_parent_scale_and_immature_rows(self):
        class Prior:
            model = None
            def __call__(self,events):return [(np.zeros(len(e['indices'])),0.) for e in events]
        class Learned:
            row_mean = torch.zeros(101)
            def graph(self,pair,rows,er,ec):return torch.full((len(pair),),-2.),torch.full((len(rows),),2.)
        tracker = SimpleNamespace(emissions=Counter({1:3}),last_emission={1:0},memories={},state={'scan':1})
        event = dict(batch=1,indices=[0],track_features=np.zeros(98),edge_features=np.zeros((1,85),np.float32),
            base_costs=[.1],dummy_cost=1.15)
        scorer = GraphScorer(Prior(),Learned(),tracker,.2)
        delta,null = scorer([event])[0]
        self.assertAlmostEqual(.1+.25*delta[0],.5)
        self.assertAlmostEqual(1.15+.25*null,-.55)
        tracker.emissions[1] = 2
        delta,null = scorer([event])[0]
        self.assertEqual(delta[0],0.)
        self.assertEqual(null,0.)

    def test_joint_training_updates_edge_and_null_without_updating_prior(self):
        data = dict(pair=self.pair.numpy(),row=self.rows.numpy(),edge_row=self.er.numpy(),edge_col=self.ec.numpy(),
            row_index=np.zeros((3,12),int),edge_known=np.ones(4,bool),row_allowed=np.ones(3,bool),
            col_allowed=np.ones(3,bool),edge_positive=np.array([True,False,False,True]),
            col_target=np.array([True,False,False,True]),known_ids=np.arange(4))
        model = self.model(True)
        table = Tables(data,'cpu')
        fixed = model.prior.edges(self.pair).detach().clone()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        for _ in range(2):
            loss = table.loss(model,table.batch(),fixed)
            self.assertTrue(torch.isfinite(loss))
            optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertGreater(float(model.residual[-1].weight.detach().abs().sum()),0.)
        self.assertGreater(float(model.null[-1].weight.detach().abs().sum()),0.)
        torch.testing.assert_close(model.prior.edges(self.pair),fixed)


if __name__ == '__main__':unittest.main()
