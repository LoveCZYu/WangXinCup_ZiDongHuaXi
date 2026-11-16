import unittest

import numpy as np
import torch

from experiments.joint_matching.model import PlanModel, components, feasible_plans


class JointPlanTests(unittest.TestCase):
    def test_joint_swap_and_unchosen_rows_preserve_the_point_set(self):
        graph = {1: {40, 41}, 2: {40, 41}, 3: {40}, 9: {99}}
        rows, points = next(components(graph))
        self.assertEqual(rows, (1, 2, 3))
        costs = {(row, point): .2 for row, indices in graph.items() for point in indices}
        costs[3, 40] = .5
        baseline = ((1, 40), (2, 41))
        options = list(feasible_plans(rows, points, graph, costs, baseline))
        self.assertEqual(options[0], (baseline, 0.))
        self.assertIn(((2, 40), (1, 41)), [plan for plan, _ in options])
        self.assertIn(((3, 40), (2, 41)), [plan for plan, _ in options])
        for plan, _ in options:
            self.assertEqual([point for _, point in plan], [40, 41])
            self.assertEqual(len(set(row for row, _ in plan)), 2)

    def test_zero_initialized_ranker_favors_the_prior(self):
        model = PlanModel()
        plans = torch.from_numpy(np.random.default_rng(1).normal(size=(3, 32)).astype(np.float32))
        result = model(plans, torch.tensor([0., .2, .5]))
        self.assertEqual(result.argmax().item(), 0)


if __name__ == '__main__':
    unittest.main()
