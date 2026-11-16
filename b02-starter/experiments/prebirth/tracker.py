"""Intercept an unpublished track's confirmation using its observed prefix.

No delay, future frame, point synthesis, or truth input is introduced. Confirmed
matches from the combined control have priority over these proposals.
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from b02.algorithm import POSITION
from experiments.id_recovery.affinity import pack
from experiments.id_recovery.tracker import CONFIGS as RECOVERY, RecoveryTracker, Retired, cutoff
from experiments.v2_association.context import augment
from experiments.v2_association.features import features
from experiments.v3_tracking.memory import Memory


@dataclass(frozen=True)
class Prebirth:
    retired: bool = False
    probability: float = .90
    temporal: float = .50
    velocity: float = .60
    margin: float = .25
    apply: bool = True
    short: bool = False
    motion_scale: float = 1.


CONFIGS = {'control': None, 'live': Prebirth(), 'both': Prebirth(retired=True),
           'both_relaxed': Prebirth(retired=True, probability=.80, temporal=.85, velocity=1.),
           'audit_only': Prebirth(retired=True, probability=.80, temporal=.85, velocity=1., apply=False),
           'short_live': Prebirth(short=True),
           'short_both': Prebirth(retired=True, short=True),
           'short_relaxed': Prebirth(retired=True, short=True, probability=.80, temporal=.85, velocity=1.),
           'short_wide': Prebirth(retired=True, short=True, probability=.80, temporal=.85, velocity=1., motion_scale=1.75),
           'wide_confident': Prebirth(retired=True, short=True, probability=.95, temporal=.85, velocity=1., motion_scale=1.75)}


def prefix_geometry(saved, history, position, scan, scales, motion_scale=1.):
    """Agreement of two past observations and the present with the old motion."""
    observed = [(np.asarray([point[k]/scale for k, scale in zip(POSITION, scales)]), when)
                for point, _, when in history[-2:]]
    if len(observed) < 2:
        return None
    observed.append((np.asarray(position), scan))
    old = saved.track
    radius = min(1.+.10*max(0, scan-old.last_scan-1), 1.8)*motion_scale
    errors = [float(np.square(p-np.asarray(old.position)-np.asarray(old.velocity)*(t-old.last_scan)).sum())/radius**2
              for p, t in observed]
    times = np.asarray([t-scan for _, t in observed], dtype=float)
    times -= times.mean()
    xyz = np.stack([p for p, _ in observed])
    velocity = (times[:, None]*(xyz-xyz.mean(0))).sum(0)/max(float(np.square(times).sum()), 1.)
    disagreement = float(np.square(velocity-np.asarray(old.velocity)).sum())
    return float(np.mean(errors)), errors[-1], disagreement, radius


class PrebirthTracker(RecoveryTracker):
    def __init__(self, scorer, affinity, config):
        super().__init__(scorer, affinity, RECOVERY['combined'])
        self.prebirth = config
        self.first_matches = {}
        self.proposals = []
        self.short_retired = {}

    def __call__(self, points, scores):
        self.proposals = []
        saved = {}
        if self.prebirth is not None and self.prebirth.short:
            scan = self.state.get('scan', 0)+1
            for batch, old in list(self.short_retired.items()):
                if scan-old.track.last_scan > 14:
                    del self.short_retired[batch]
            for batch, track in self.state.get('tracks', {}).items():
                if not 1 <= self.emissions[batch] <= 2 or scan-track.last_scan <= self.config.max_missed:
                    continue
                if sum(score >= .5 for _, score, _ in self.history[batch]) < 2:
                    continue
                saved[batch] = Retired(track, self.history[batch], self.context_history[batch],
                    self.memories.get(batch, Memory()), self.strong_hits[batch], self.quality_hits[batch],
                    self.birth_scan[batch], self.emissions[batch], self.last_emission[batch])
        result = super().__call__(points, scores)
        for batch in self.deleted:
            if batch in saved:
                self.short_retired[batch] = saved[batch]
                self.counters['short_archived'] += 1
        return result

    def assign(self, track_ids, scan, positions, grid, occupied, points, scores):
        stage = self.assignment_stage
        matched = super().assign(track_ids, scan, positions, grid, occupied, points, scores)
        if not stage:
            self.first_matches = matched
        if self.prebirth is not None:
            used = set(self.first_matches) | set(matched)
            self.intercept(matched, used, scan, positions, points, scores)
        return matched

    def old_candidates(self, used, scan):
        for batch, track in self.state['tracks'].items():
            minimum = 1 if self.prebirth.short else 3
            if batch in used or self.emissions[batch] < minimum or scan-self.last_emission[batch] > 14:
                continue
            memory = self.memories.get(batch)
            short = self.prebirth.short and self.emissions[batch] < 3
            if short:
                if sum(score >= .5 for _, score, _ in self.history[batch]) < 2:
                    continue
                memory = memory or Memory()
            elif memory is None or len(memory.entries) < 2:
                continue
            yield batch, False, Retired(track, self.history[batch], self.context_history[batch], memory,
                                        self.strong_hits[batch], self.quality_hits[batch], self.birth_scan[batch],
                                        self.emissions[batch], self.last_emission[batch])
        if self.prebirth.retired:
            banks = [self.retired, self.short_retired] if self.prebirth.short else [self.retired]
            for batch, saved in ((batch, saved) for bank in banks for batch, saved in bank.items()):
                if batch not in used:
                    yield batch, True, saved

    def intercept(self, matched, used, scan, positions, points, scores):
        newborns = [(batch, index) for batch, index in matched.items()
                    if not self.emissions[batch] and self.state['tracks'][batch].hits >= 2
                    and scores[index] >= self.gate.policy.enter
                    and any(score >= .5 for _, score, _ in self.history[batch][-2:])]
        if not newborns:
            return
        self.counters['first_output_candidates'] += len(newborns)
        keys, txs, exs = [], [], []
        for old, retired, saved in list(self.old_candidates(used, scan)):
            candidates = []
            for new, index in newborns:
                if self.birth_scan[new] < saved.last_emission-2:
                    continue
                geometry = prefix_geometry(saved, self.history[new], positions[index], scan, self.config.gate,
                                           self.prebirth.motion_scale)
                if geometry is None:
                    continue
                temporal, current, velocity, radius = geometry
                if current > .90 or temporal > 1.25 or velocity > 2.:
                    continue
                candidates.append((new, index, temporal, current, velocity, radius))
            if not candidates:
                continue
            indices = [row[1] for row in candidates]
            costs = [row[3]+.5*(1-scores[row[1]]) for row in candidates]
            radius = candidates[0][-1]
            tx, ex = features(saved.track, saved.history, scan, saved.birth_scan, saved.strong_hits,
                              saved.quality_hits, points, scores, indices, costs, radius)
            tx, ex = augment(tx, ex, saved.context_history, self.current_context, indices)
            tx, ex = saved.memory.augment(tx, ex, self.current_context[indices], scan)
            keys.extend((old, retired, *row[:-1]) for row in candidates)
            txs.extend([tx]*len(candidates))
            exs.extend(ex)
        if not keys:
            return
        self.counters['prebirth_proposals'] += len(keys)
        logits = self.affinity.score(pack(np.stack(txs), np.stack(exs)))
        options = defaultdict(list)
        competing = defaultdict(list)
        for (old, retired, new, index, temporal, current, velocity), logit in zip(keys, logits, strict=True):
            strength = float(np.clip(logit, -6, 6))-1.2*temporal-.5*velocity
            accepted = (logit >= cutoff(self.prebirth.probability) and temporal <= self.prebirth.temporal
                        and velocity <= self.prebirth.velocity)
            proposal = dict(old=old, retired=retired, new=new, index=index, logit=float(logit),
                            temporal=temporal, current=current, velocity=velocity, strength=strength,
                            eligible=bool(accepted), selected=False)
            self.proposals.append(proposal)
            self.counters['prebirth_eligible'] += bool(accepted)
            options[old].append(proposal)
            competing[new].append(proposal)
        ranked = []
        for old, choices in options.items():
            choices.sort(key=lambda p: p['strength'], reverse=True)
            best = choices[0]
            if not best['eligible']:
                continue
            if len(choices)>1 and best['strength']-choices[1]['strength'] < self.prebirth.margin:
                continue
            if any(p['old'] != old and best['strength']-p['strength'] < self.prebirth.margin for p in competing[best['new']]):
                continue
            ranked.append(best)
        for row in sorted(ranked, key=lambda p: p['strength'], reverse=True):
            old, new, index = row['old'], row['new'], row['index']
            if old in used or matched.get(new) != index:
                continue
            if not self.prebirth.apply:
                continue
            if row['retired']:
                self.restore(old)
            del matched[new]
            matched[old] = index
            used.add(old)
            row['selected'] = True
            operation = 'prebirth_retired' if row['retired'] else 'prebirth_live'
            if self.emissions[old] < 3:
                operation = operation.replace('prebirth_', 'prebirth_short_')
            self.decisions.append((operation, old, index, new))

    def restore(self, batch):
        saved = self.retired.pop(batch) if batch in self.retired else self.short_retired.pop(batch)
        for mapping, value in ((self.state['tracks'], saved.track), (self.history, saved.history),
                               (self.context_history, saved.context_history), (self.memories, saved.memory),
                               (self.strong_hits, saved.strong_hits), (self.quality_hits, saved.quality_hits),
                               (self.birth_scan, saved.birth_scan), (self.emissions, saved.emission_count),
                               (self.last_emission, saved.last_emission)):
            mapping[batch] = value
