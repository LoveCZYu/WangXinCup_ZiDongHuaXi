"""Bounded, point-preserving joint assignment inside a single association stage."""
from collections import Counter, defaultdict
from itertools import permutations
from math import log1p

import numpy as np
import torch
from torch import nn

from experiments.v3_tracking.tracker import Tracker

MAX_OPTIONS = 6
FEATURES = 32
COST_BUDGET = .50


def components(edges):
    reverse = defaultdict(set)
    for row, points in edges.items():
        for point in points:
            reverse[point].add(row)
    seen = set()
    for seed in sorted(edges):
        if seed in seen:
            continue
        stack, rows, points = [seed], set(), set()
        while stack:
            row = stack.pop()
            if row in rows:
                continue
            rows.add(row)
            for point in edges[row]:
                if point not in points:
                    points.add(point)
                    stack.extend(reverse[point] - rows)
        seen.update(rows)
        if 2 <= len(rows) <= 3 and 1 <= len(points) <= len(rows):
            yield tuple(sorted(rows)), tuple(sorted(points))


def feasible_plans(rows, points, edges, cost, baseline):
    origin = sum(cost[(row, point)] for row, point in baseline)
    yield baseline, 0.
    for owners in permutations(rows, len(points)):
        plan = tuple(zip(owners, points))
        if plan == baseline or any(point not in edges[row] for row, point in plan):
            continue
        change = sum(cost[(row, point)] for row, point in plan) - origin
        if change <= COST_BUDGET:
            yield plan, max(change, 0.)


class PlanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(FEATURES))
        self.register_buffer('std', torch.ones(FEATURES))
        self.net = nn.Sequential(nn.Linear(FEATURES, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, plans, costs):
        return self.net(((plans-self.mean)/self.std).clamp(-8, 8)).squeeze(-1).tanh()*2.0 - costs/.25


class JointTracker(Tracker):
    def __init__(self, scorer, *, model=None, rule=False, collect=False):
        super().__init__(scorer, memory=True)
        self.plan_model = model
        self.rule = rule
        self.collect = collect
        self.groups = []
        self.emissions = Counter()
        self.last_emission = {}
        self.plan_counts = Counter()

    def assign(self, track_ids, scan, positions, grid, occupied, points, scores):
        start = len(self.events)
        matched = super().assign(track_ids, scan, positions, grid, occupied, points, scores)
        if not (self.collect or self.rule or self.plan_model is not None):
            return matched
        # A confirmed track's low-score assignment remains locked. The stage's
        # selected strong observations are reassigned jointly, with unused rows allowed.
        if not track_ids or not self.confirmed(track_ids[0]):
            return matched
        owners = {point: row for row, point in matched.items() if scores[point] >= self.gate.policy.enter}
        if not owners:
            return matched
        graph, values, evidence = {}, {}, {}
        for event in self.events[start:]:
            row = event['batch']
            if row in matched and matched[row] not in owners:
                continue
            edges = set()
            for j, point in enumerate(event['indices']):
                if point in owners:
                    edges.add(point)
                    values[row, point] = event['effective_costs'][j] - event['effective_dummy']
                    evidence[row, point] = (event['edge_features'][j], event['base_costs'][j])
            if edges:
                graph[row] = edges
        proposals = []
        for rows, observations in components(graph):
            original = tuple((owners[point], point) for point in observations)
            options = list(feasible_plans(rows, observations, graph, values, original))
            if len(options) < 2:
                continue
            features = np.stack([self.features(plan, evidence, scan, len(rows), options[0][0]) for plan, _ in options])
            costs = np.asarray([cost for _, cost in options], dtype=np.float32)
            self.plan_counts['groups'] += 1
            self.plan_counts['alternatives'] += len(options)-1
            proposals.append((rows, options, features, costs))
        if self.collect:
            self.groups.extend(proposals)
        if not proposals or (not self.rule and self.plan_model is None):
            return matched
        if self.plan_model is not None:
            with torch.no_grad():
                logits = self.plan_model(torch.from_numpy(np.concatenate([p[2] for p in proposals])),
                                         torch.from_numpy(np.concatenate([p[3] for p in proposals]))).numpy()
        cursor = 0
        for rows, options, features, costs in proposals:
            if self.rule:
                # First half is the mean evidence vector; reward mature emitters
                # that also fit the recent velocity and stable memory descriptor.
                scores_joint = -costs/.25 + .9*features[:, 8] - .5*features[:, 3] - .5*features[:, 6]
            else:
                scores_joint = logits[cursor:cursor+len(options)]
            cursor += len(options)
            chosen = int(np.argmax(scores_joint))
            if chosen and scores_joint[chosen] > scores_joint[0]+.08:
                for row in rows:
                    matched.pop(row, None)
                matched.update(options[chosen][0])
                self.plan_counts['changed_groups'] += 1
        return matched

    def features(self, plan, evidence, scan, row_count, original):
        items = []
        original_rows = set(original)
        for row, point in plan:
            edge, base = evidence[row, point]
            observations = self.emissions[row]
            elapsed = scan - self.last_emission.get(row, scan-16)
            memory = self.memories.get(row)
            track = self.state['tracks'][row]
            items.append([base, float(edge[9]), float(np.linalg.norm(edge[6:9])), float(np.linalg.norm(edge[80:83])),
                          float(np.linalg.norm(edge[0:3])), float(np.linalg.norm(edge[50:60])),
                          float(np.abs(edge[70:80]).mean()), float(edge[11]), min(log1p(observations)/5, 2.),
                          min(elapsed/8, 2.), min(track.hits/50, 2.), min(len(memory.entries)/16, 1.) if memory else 0.,
                          float(observations > 0), float((row, point) in original_rows),
                          len(plan)/3, row_count/3])
        array = np.asarray(items, dtype=np.float32)
        return np.concatenate((array.mean(0), array.max(0)))

    def __call__(self, points, scores):
        self.groups = []
        outputs = super().__call__(points, scores)
        for output in outputs:
            row = output['BatchID']
            self.emissions[row] += 1
            self.last_emission[row] = self.state['scan']
        for row in self.deleted:
            self.emissions.pop(row, None)
            self.last_emission.pop(row, None)
        return outputs
