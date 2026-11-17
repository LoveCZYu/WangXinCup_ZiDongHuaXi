"""Require positive absolute affinity before it may influence a contested point."""
from collections import defaultdict

import numpy as np

from experiments.candidate_onset.mixed import install as install_mixed
from experiments.id_recovery.affinity import pack

MIN_AFFINITY = 0.0


class EvidenceGuard:
    def __init__(self, prior, tracker):
        self.prior = prior
        self.model = prior.model
        self.tracker = tracker

    def __call__(self, events):
        deltas = self.prior(events)
        results = [(np.asarray(edges).copy(), null) for edges, null in deltas]
        contenders = defaultdict(list)
        for row, event in enumerate(events):
            batch = event["batch"]
            if self.tracker.emissions.get(batch, 0) < 3:
                continue
            for edge, point in enumerate(event["indices"]):
                if event["edge_features"][edge, 11] >= self.tracker.gate.policy.enter:
                    contenders[point].append((row, edge))

        columns = [column for column in contenders.values() if len(column) >= 2]
        edges = [edge for column in columns for edge in column]
        if not edges:
            return results
        features = pack(
            np.stack([events[row]["track_features"] for row, _ in edges]),
            np.stack([events[row]["edge_features"][edge] for row, edge in edges]),
        )
        scores = dict(zip(edges, self.tracker.affinity.score(features), strict=True))
        for column in columns:
            best = max(float(scores[key]) for key in column)
            self.tracker.motion_counts["contested_identity_columns"] += 1
            if best < MIN_AFFINITY:
                self.tracker.motion_counts["weak_identity_columns_preserved"] += 1
                continue
            for row, edge in column:
                results[row][0][edge] += float(np.clip(best - scores[row, edge], 0.0, 4.0))
        return results


def install(tracker):
    install_mixed(tracker, "backup")
    tracker.model_scorer = EvidenceGuard(tracker.model_scorer, tracker)
    return tracker
