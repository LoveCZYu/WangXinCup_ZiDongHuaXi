"""Experimental column-wise identity/motion penalties; frozen point model."""
from collections import Counter, defaultdict
from math import log

import numpy as np

from experiments.continuation.fusion import MemoryFusionTracker
from experiments.id_recovery.affinity import pack
from experiments.identity_competition.motion import memory_motion
from experiments.v2_association.features import normalized

MODES = ('baseline', 'identity', 'motion', 'combined', 'volume', 'identity_volume')


class ColumnScorer:
    """Compare mature IDs that share a strong point, even with one edge each.

    Existing solver applies its fixed .25 residual scale. Each penalty is
    nonnegative and the best evidence in a column receives zero penalty.
    No candidate edge, dummy cost or output gate is added here.
    """
    def __init__(self, prior, affinity, tracker, mode):
        assert mode in MODES
        self.prior = prior
        self.model = prior.model
        self.affinity = affinity
        self.tracker = tracker
        self.mode = mode
        self.counts = Counter()

    def __call__(self, events):
        original = self.prior(events)
        result = [(np.asarray(edge).copy(), null) for edge, null in original]
        if self.mode == 'baseline':
            return result
        groups = defaultdict(list)
        for row, event in enumerate(events):
            batch = event['batch']
            if self.tracker.emissions.get(batch, 0) < 3:
                continue
            for j, index in enumerate(event['indices']):
                if event['edge_features'][j, 11] >= self.tracker.gate.policy.enter:
                    groups[index].append((row, j, batch))
        groups = {index: rows for index, rows in groups.items() if len(rows) >= 2}
        self.counts['conflict_points'] += len(groups)
        keys = [(index, *row) for index, rows in groups.items() for row in rows]
        if not keys:
            return result
        if self.mode in ('identity', 'combined', 'identity_volume'):
            logits = self.affinity.score(pack(
                np.stack([events[row]['track_features'] for _, row, _, _ in keys]),
                np.stack([events[row]['edge_features'][j] for _, row, j, _ in keys])))
            values = {(row, j): float(logit) for (_, row, j, _), logit in zip(keys, logits, strict=True)}
            for index, rows in groups.items():
                best = max(values[row, j] for row, j, _ in rows)
                for row, j, _ in rows:
                    penalty = float(np.clip(best-values[row, j], 0., 4.))
                    result[row][0][j] += penalty
                    self.counts['identity_penalized_edges'] += penalty > 0
        if self.mode in ('volume', 'identity_volume'):
            scan = self.tracker.state['scan']
            for index, rows in groups.items():
                # A wider search region discounts the baseline distance. This
                # fixed diagnostic penalty tests the missing gate-volume cost;
                # the gate radius is not a calibrated noise covariance.
                costs = [3.*log(min(1.+.2*(scan-self.tracker.state['tracks'][batch].last_scan-1), 1.6))
                         for _, _, batch in rows]
                best = min(costs)
                for (row, j, _), cost in zip(rows, costs, strict=True):
                    result[row][0][j] += cost-best
                    self.counts['volume_penalized_edges'] += cost > best
        if self.mode in ('motion', 'combined'):
            scan = self.tracker.state['scan']
            predictions = {}
            for _, _, _, batch in keys:
                if batch not in predictions:
                    memory = self.tracker.memories.get(batch)
                    predictions[batch] = memory_motion(memory, scan) if memory is not None else None
            for index, rows in groups.items():
                if any(predictions[batch] is None for _, _, batch in rows):
                    continue
                point = normalized(self.tracker.research_points[index])
                costs = []
                for row, j, batch in rows:
                    age = scan-self.tracker.state['tracks'][batch].last_scan
                    radius = min(1.+.2*(age-1), 1.6)
                    costs.append(float(np.square(point-predictions[batch][0]).sum()/radius**2))
                best = min(costs)
                for (row, j, _), cost in zip(rows, costs, strict=True):
                    penalty = float(np.clip(cost-best, 0., 4.))
                    result[row][0][j] += penalty
                    self.counts['motion_penalized_edges'] += penalty > 0
                self.counts['motion_qualified_points'] += 1
        return result


class CompetitionTracker(MemoryFusionTracker):
    def __init__(self, scorer, affinity, head, fusion, mode):
        super().__init__(scorer, affinity, head, fusion, guard=.90)
        self.model_scorer = ColumnScorer(self.model_scorer, affinity, self, mode)

    def __call__(self, points, scores):
        self.research_points = points
        return super().__call__(points, scores)


def install(tracker, mode):
    """Use only on a copied private fixture, leaving the original unchanged."""
    tracker.__class__ = CompetitionTracker
    tracker.model_scorer = ColumnScorer(tracker.model_scorer, tracker.affinity, tracker, mode)
    return tracker
