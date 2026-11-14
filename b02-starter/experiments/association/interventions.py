"""Causal tracking interventions and their fixed combination."""
from functools import partial
from heapq import nsmallest
from itertools import product
from math import floor

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from b02.algorithm import DEFAULT_CONFIG, process_circle
from experiments.association.assignment import assign
from experiments.point_scorer_expanded.policy import OutputGate, Policy


def priority_assign(track_ids, tracks, scan, positions, grid, occupied, config, *, scores, strong_hits):
    """Keep original motion candidates, adding a bounded per-track evidence penalty.

    A track with six prior score>=0.50 observations pays no extra real-edge cost;
    one without such observations pays 0.15. Dummy cost remains 1.15. This tests
    competition between unequal-evidence tracks, not learned identity matching.
    """
    rows, columns, costs, pending, point_columns = [], [], [], [], {}
    for batch in track_ids:
        track = tracks[batch]
        elapsed = scan-track.last_scan
        predicted = tuple(x+v*elapsed for x,v in zip(track.position, track.velocity))
        radius = min(1+.2*(elapsed-1), 1.6)
        neighborhood = [range(floor(x-radius), floor(x+radius)+1) for x in predicted]
        edges = []
        for cell in product(*neighborhood):
            for index in grid.get(cell, ()):
                if index in occupied:
                    continue
                distance = sum((a-b)**2 for a,b in zip(predicted, positions[index]))
                if distance <= radius**2:
                    cost = distance/radius**2+.50*(1-scores[index])
                    if cost < 1.15:
                        edges.append((cost,index))
        if not edges:
            continue
        row = len(pending)
        pending.append(batch)
        penalty = .15*(1-min(strong_hits.get(batch,0),6)/6)
        for cost,index in nsmallest(config.candidates_per_track, edges):
            rows.append(row)
            columns.append(point_columns.setdefault(index,len(point_columns)))
            costs.append(cost+penalty+1e-8)
    if not pending:
        return {}
    point_count, count = len(point_columns), len(pending)
    rows.extend(range(count))
    columns.extend(range(point_count,point_count+count))
    costs.extend([1.15+1e-8]*count)
    matrix = coo_matrix((costs,(rows,columns)),shape=(count,point_count+count)).tocsr()
    ri,ci = min_weight_full_bipartite_matching(matrix)
    indices = list(point_columns)
    matches = {pending[r]:indices[c] for r,c in zip(ri,ci) if c<point_count}
    occupied.update(matches.values())
    return matches


class Tracker:
    def __init__(self, variant, policy=None):
        if variant not in ('control','evidence_priority','confidence_velocity','combined'):
            raise ValueError(variant)
        self.variant = variant
        self.use_priority = variant in ('evidence_priority','combined')
        self.use_velocity = variant in ('confidence_velocity','combined')
        self.state = {}
        self.strong_hits = {}
        self.gate = OutputGate(policy or Policy('frozen_050_005',.50,.05,3))

    def __call__(self, points, scores):
        old = self.state.get('tracks',{})
        old_ids = set(old)
        velocities = {batch:t.velocity for batch,t in old.items()} if self.use_velocity else {}
        matches = {}
        if self.use_priority:
            method = partial(priority_assign,scores=scores,strong_hits=self.strong_hits)
        else:
            method = partial(assign,scores=scores,score_weight=.50)

        def assignment(*args):
            chosen = method(*args)
            matches.update(chosen)
            return chosen

        raw = process_circle(points,self.state,assignment=assignment)
        tracks = self.state['tracks']
        if self.use_velocity:
            for batch,index in matches.items():
                trust = min(1,scores[index]/.50)
                before = velocities[batch]
                tracks[batch].velocity = tuple(v+trust*(updated-v) for v,updated in zip(before,tracks[batch].velocity))
        if self.use_priority:
            self.strong_hits = {batch:count for batch,count in self.strong_hits.items() if batch in tracks}
            for batch,index in matches.items():
                self.strong_hits[batch] = min(6,self.strong_hits.get(batch,0)+int(scores[index]>=.50))
            new_ids = sorted(set(tracks)-old_ids)
            occupied = set(matches.values())
            free = (index for index in range(len(points)) if index not in occupied)
            born = nsmallest(len(new_ids),free,key=lambda i:(-points[i]['Amp'],i))
            assert len(born)==len(new_ids)
            for batch,index in zip(new_ids,born):
                self.strong_hits[batch] = int(scores[index]>=.50)
        self.last_raw = raw
        score_map = dict(zip((p['PntID'] for p in points),scores))
        return self.gate.apply(raw,score_map,self.state['scan']-1)
