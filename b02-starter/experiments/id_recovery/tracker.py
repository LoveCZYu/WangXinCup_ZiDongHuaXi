"""Separate causal interventions for live unmatched and recently deleted IDs."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log, log1p

import numpy as np

from experiments.id_recovery.affinity import pack
from experiments.v2_association.context import augment
from experiments.v2_association.features import features
from experiments.v3_tracking.tracker import Tracker


@dataclass(frozen=True)
class Recovery:
    active: float | None = None
    margin: float = .45
    retired: float | None = None
    retention: int = 0
    contest_new: bool = False


CONFIGS = {
    'baseline': Recovery(),
    'active_strict': Recovery(active=.95, margin=.75),
    'active': Recovery(active=.85, margin=.45),
    'retired_strict': Recovery(retired=.95, retention=8),
    'retired': Recovery(retired=.88, retention=14),
    'retired_unclaimed_or_new': Recovery(retired=.95, retention=14, contest_new=True),
    'combined': Recovery(active=.95, margin=.75, retired=.88, retention=14),
}


@dataclass
class Retired:
    track: object
    history: list
    context_history: list
    memory: object
    strong_hits: int
    quality_hits: float
    birth_scan: int
    emission_count: int
    last_emission: int


def cutoff(probability):
    return log(probability/(1-probability))


class RecoveryTracker(Tracker):
    def __init__(self, scorer, affinity, config):
        super().__init__(scorer, memory=True)
        self.affinity = affinity
        self.recovery = config
        self.emissions = Counter()
        self.last_emission = {}
        self.retired = {}
        self.decisions = []
        self.counters = Counter()
        self.assignment_stage = 0

    def assign(self, track_ids, scan, positions, grid, occupied, points, scores):
        stage = self.assignment_stage
        self.assignment_stage += 1
        start = len(self.events)
        matched = super().assign(track_ids, scan, positions, grid, occupied, points, scores)
        if stage:
            return matched
        if self.recovery.active is not None:
            self.reconnect_active(matched, self.events[start:], scan, scores)
        if self.recovery.retired is not None:
            new_matches = {index: batch for batch, index in matched.items() if not self.emissions[batch]} if self.recovery.contest_new else {}
            candidates = self.reconnect_retired(positions, occupied.difference(new_matches), points, scores, scan)
            for index in candidates.values():
                if index in new_matches:
                    del matched[new_matches[index]]
                    self.counters['retired_replaced_new_match'] += 1
            matched.update(candidates)
            occupied.update(candidates.values())
        return matched

    def reconnect_active(self, matched, events, scan, scores):
        proposals, prior, _ = self.active_options(matched, events, scan, scores)
        if not proposals:
            return
        confirmed = []
        for (batch, index, _, winner), (affinity, rival) in zip(proposals, prior, strict=True):
            old_support = min(log1p(self.emissions[batch]), 5.)
            rival_support = min(log1p(self.emissions[winner]), 5.)
            difference = float(np.clip(affinity, -6, 6) - np.clip(rival, -6, 6))
            advantage = difference + .70*(old_support-rival_support)
            if affinity >= cutoff(self.recovery.active) and advantage >= self.recovery.margin:
                confirmed.append((advantage, batch, index, winner))
        self.apply_transfers(matched, confirmed)

    def active_options(self, matched, events, scan, scores):
        winners = {index: batch for batch, index in matched.items() if scores[index] >= self.gate.policy.enter}
        if not winners:
            return [], [], []
        proposals = []
        rival_keys = set()
        lookup = {event['batch']: event for event in events}
        for event in events:
            batch = event['batch']
            if batch in matched or self.emissions[batch] < 2 or scan-self.last_emission[batch] > 6:
                continue
            for j, index in enumerate(event['indices']):
                winner = winners.get(index)
                if winner is None:
                    continue
                proposals.append((batch, index, j, winner))
                rival_keys.add((winner, index))
        if not proposals:
            return [], [], []
        rivals = sorted(rival_keys)
        keys = [(batch, index, j) for batch, index, j, _ in proposals]
        keys += [(batch, index, lookup[batch]['indices'].index(index)) for batch, index in rivals]
        packed = pack(np.stack([lookup[b]['track_features'] for b, _, _ in keys]),
                      np.stack([lookup[b]['edge_features'][j] for b, _, j in keys]))
        logits = self.affinity.score(packed)
        right = dict(zip(rivals, logits[len(proposals):], strict=True))
        rival_rows = {key: i for i, key in enumerate(rivals, len(proposals))}
        self.counters['active_proposals'] += len(proposals)
        prior = [(float(affinity), float(right[winner, index])) for (_, index, _, winner), affinity in zip(proposals, logits[:len(proposals)], strict=True)]
        paired = np.concatenate((packed[:len(proposals)], packed[[rival_rows[winner, index] for _, index, _, winner in proposals]]), axis=1)
        return proposals, prior, paired

    def apply_transfers(self, matched, confirmed):
        used = set()
        for _, batch, index, winner in sorted(confirmed, reverse=True):
            if batch in used or matched.get(winner) != index:
                continue
            del matched[winner]
            matched[batch] = index
            used.add(batch)
            self.decisions.append(('active', batch, index, winner))

    def reconnect_retired(self, positions, occupied, points, scores, scan):
        available = [i for i, score in enumerate(scores) if score >= self.gate.policy.enter and i not in occupied]
        if not self.retired or not available:
            return {}
        observations = np.asarray([positions[i] for i in available], dtype=np.float32)
        keys, tracks, edges = [], [], []
        for batch, saved in sorted(self.retired.items()):
            age = scan-saved.track.last_scan
            predicted = np.asarray(saved.track.position) + np.asarray(saved.track.velocity)*age
            radius = min(1.+.10*(age-1), 1.8)
            errors = np.square(observations-predicted).sum(1)/radius**2
            selected = [j for j in np.argsort(errors, kind='stable')[:6] if errors[j] <= .75]
            if not selected:
                continue
            indices = [available[j] for j in selected]
            costs = [float(errors[j]) + .5*(1-scores[available[j]]) for j in selected]
            track_x, edge_x = features(saved.track, saved.history, scan, saved.birth_scan, saved.strong_hits,
                                       saved.quality_hits, points, scores, indices, costs, radius)
            track_x, edge_x = augment(track_x, edge_x, saved.context_history, self.current_context, indices)
            track_x, edge_x = saved.memory.augment(track_x, edge_x, self.current_context[indices], scan)
            keys.extend((batch, index, age, float(errors[j])) for index, j in zip(indices, selected, strict=True))
            tracks.extend([track_x]*len(indices))
            edges.extend(edge_x)
        if not keys:
            return {}
        self.counters['retired_proposals'] += len(keys)
        logits = self.affinity.score(pack(np.stack(tracks), np.stack(edges)))
        proposed = defaultdict(list)
        competing = defaultdict(list)
        for (batch, index, age, error), logit in zip(keys, logits, strict=True):
            evidence = float(np.clip(logit, -6, 6)) - 1.2*error - .10*max(0, age-6)
            proposal = (evidence, batch, index, float(logit))
            proposed[batch].append(proposal)
            competing[index].append(proposal)
        restored = {}
        for batch, options in proposed.items():
            options.sort(reverse=True)
            strength, _, index, absolute = options[0]
            others = competing[index]
            if absolute < cutoff(self.recovery.retired):
                continue
            if len(options) > 1 and strength-options[1][0] < .25:
                continue
            if any(other != batch and strength-value < .25 for value, other, _, _ in others):
                continue
            saved = self.retired.pop(batch)
            self.state['tracks'][batch] = saved.track
            self.history[batch] = saved.history
            self.context_history[batch] = saved.context_history
            self.memories[batch] = saved.memory
            self.strong_hits[batch] = saved.strong_hits
            self.quality_hits[batch] = saved.quality_hits
            self.birth_scan[batch] = saved.birth_scan
            self.emissions[batch] = saved.emission_count
            self.last_emission[batch] = saved.last_emission
            restored[batch] = index
            self.decisions.append(('retired', batch, index, None))
        return restored

    def __call__(self, points, scores):
        self.assignment_stage = 0
        self.decisions = []
        saved = {}
        if self.recovery.retired is not None:
            scan = self.state.get('scan', 0)+1
            for batch, old in list(self.retired.items()):
                if scan-old.track.last_scan > self.recovery.retention:
                    del self.retired[batch]
            for batch, track in self.state.get('tracks', {}).items():
                memory = self.memories.get(batch)
                if self.emissions[batch] < 3 or memory is None or len(memory.entries) < 2:
                    continue
                if scan-track.last_scan <= self.config.max_missed or scan-self.last_emission[batch] > self.recovery.retention:
                    continue
                saved[batch] = Retired(track, self.history[batch], self.context_history[batch], memory,
                                       self.strong_hits[batch], self.quality_hits[batch], self.birth_scan[batch],
                                       self.emissions[batch], self.last_emission[batch])
        result = super().__call__(points, scores)
        for output in result:
            batch = output['BatchID']
            self.emissions[batch] += 1
            self.last_emission[batch] = self.state['scan']
        for batch in self.deleted:
            if batch in saved:
                self.retired[batch] = saved[batch]
                self.counters['archived'] += 1
            self.emissions.pop(batch, None)
            self.last_emission.pop(batch, None)
        return result
