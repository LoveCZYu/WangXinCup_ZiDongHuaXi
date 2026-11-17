"""Causal state-only experiment driven by the fixed development-scene edge screen."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.candidate_onset.propagation import MIN_AFFINITY, restore
from experiments.candidate_onset.propagation_summary import features
from experiments.candidate_onset.propagation_work import capture_prefix
from experiments.id_recovery.affinity import pack


class Screen:
    def __init__(self, path):
        model = json.loads(Path(path).read_bytes())
        self.mean = np.asarray(model["mean"])
        self.scale = np.asarray(model["scale"])
        self.coef = np.asarray(model["coefficients"])
        self.threshold = model["threshold"]

    def flagged(self, tracker, scores, eligible):
        edges = {}
        for event in tracker.events:
            batch = event["batch"]
            if batch not in eligible:
                continue
            index = tracker.last_matches.get(batch)
            if index is not None and scores[index] >= .525 and index in event["indices"]:
                edges[batch] = (index, event, event["indices"].index(index))
        if not edges:
            return []
        tx = np.stack([event["track_features"] for _,event,_ in edges.values()])
        ex = np.stack([event["edge_features"][i] for _,event,i in edges.values()])
        affinities = tracker.affinity.score(pack(tx,ex))
        selected = []
        for (batch,(index,event,_)), affinity, track_features, edge_features in zip(
                edges.items(), affinities, tx, ex, strict=True):
            if affinity < MIN_AFFINITY:
                continue
            sample = dict(affinity=float(affinity), detector_score=float(scores[index]),
                          track_features=track_features, edge_features=edge_features,
                          candidate_count=len(event["indices"]))
            z = (np.asarray(features(sample))-self.mean)/self.scale
            if float(z @ self.coef[:-1]+self.coef[-1]) > self.threshold:
                selected.append((batch,index))
        return selected


class Quarantine:
    def __init__(self, tracker, screen, full):
        self.tracker, self.screen, self.full = tracker, screen, full
        self.counters = Counter()
        self.applied = []

    def __call__(self, points, scores):
        tracker = self.tracker
        active = tracker.state.get("tracks", {})
        eligible = {batch for batch in active
                    if tracker.emissions.get(batch, 0) >= 4 and len(tracker.history.get(batch, ())) >= 4}
        prefix = capture_prefix(tracker, eligible)
        output = tracker(points,scores)
        self.applied = []
        published = {(item["BatchID"], item["PntID"]) for item in output}
        for batch,index in self.screen.flagged(tracker,scores,eligible):
            if (batch, points[index]["PntID"]) not in published:
                continue
            if restore(tracker,prefix,batch,full=self.full):
                self.applied.append((batch,index))
        self.counters["rollbacks"] += len(self.applied)
        return output
