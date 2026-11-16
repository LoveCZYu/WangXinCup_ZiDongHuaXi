"""Separate candidate admission from scoring, then learn from causal prefixes."""
from collections import defaultdict
from dataclasses import dataclass
from math import log1p

import numpy as np

from b02.algorithm import POSITION
from experiments.id_recovery.affinity import FEATURE_COUNT, pack
from experiments.id_recovery.tracker import cutoff
from experiments.prebirth.tracker import CONFIGS as PREBIRTH, PrebirthTracker, prefix_geometry
from experiments.v2_association.context import augment
from experiments.v2_association.features import features
from experiments.v3_tracking.memory import Memory

MAX_OLD = 12
EXTRA_DIM = 41
DIMENSION = 2*FEATURE_COUNT+EXTRA_DIM


@dataclass(frozen=True)
class Mode:
    admission: float = 1.75
    scoring: float = 1.75
    learned: bool = False


MODES = {'wide_shadow': Mode(), 'narrow_shadow': Mode(1., 1.),
         'gate_only': Mode(1.75, 1.), 'score_only': Mode(1., 1.75),
         'learned': Mode(learned=True), 'learned_safe': Mode(learned=True)}


def pair_features(tracker, saved, indices, costs, radius, points, scores, scan):
    tx, ex = features(saved.track, saved.history, scan, saved.birth_scan, saved.strong_hits,
                      saved.quality_hits, points, scores, indices, costs, radius)
    tx, ex = augment(tx, ex, saved.context_history, tracker.current_context, indices)
    tx, ex = saved.memory.augment(tx, ex, tracker.current_context[indices], scan)
    return pack(np.stack([tx]*len(indices)), ex)


class ContinuationTracker(PrebirthTracker):
    def __init__(self, scorer, affinity, mode='wide_shadow', model=None, threshold=None):
        super().__init__(scorer, affinity, PREBIRTH['short_wide'])
        self.mode = MODES[mode]
        self.mode_name = mode
        self.continuation_model = model
        self.threshold = threshold
        self.samples = []

    def __call__(self, points, scores):
        self.samples = []
        return super().__call__(points, scores)

    def candidates(self, matched, used, scan, positions, points, scores):
        newborns = [(batch, index) for batch, index in matched.items()
                    if not self.emissions[batch] and self.state['tracks'][batch].hits >= 2
                    and scores[index] >= self.gate.policy.enter
                    and any(score >= .5 for _, score, _ in self.history[batch][-2:])]
        self.counters['first_output_candidates'] += len(newborns)
        rows, canonical, scoring = [], [], []
        for old, retired, saved in list(self.old_candidates(used, scan)):
            candidates = []
            for new, index in newborns:
                if self.birth_scan[new] < saved.last_emission-2:
                    continue
                geometry = prefix_geometry(saved, self.history[new], positions[index], scan, self.config.gate,
                                           self.mode.admission)
                temporal, current, velocity, radius = geometry
                if current > .90 or temporal > 1.25 or velocity > 2.:
                    continue
                candidates.append(dict(old=old, retired=retired, saved=saved, new=new, index=index,
                                       temporal=temporal, current=current, velocity=velocity, radius=radius))
            if not candidates:
                continue
            base_radius = candidates[0]['radius']/self.mode.admission
            scoring_radius = base_radius*self.mode.scoring
            indices = [row['index'] for row in candidates]
            costs = [row['current']*(self.mode.admission/self.mode.scoring)**2+.5*(1-scores[row['index']]) for row in candidates]
            scoring.extend(pair_features(self, saved, indices, costs, scoring_radius, points, scores, scan))
            costs = [row['current']*self.mode.admission**2+.5*(1-scores[row['index']]) for row in candidates]
            canonical.extend(pair_features(self, saved, indices, costs, base_radius, points, scores, scan))
            rows.extend(candidates)
        if not rows:
            return []
        canonical = np.stack(canonical)
        scoring = np.stack(scoring)
        logits = self.affinity.score(scoring)
        canonical_logits = self.affinity.score(canonical)
        for row, x, logit, standard in zip(rows, canonical, logits, canonical_logits, strict=True):
            row.update(old_features=x, logit=float(logit), canonical_logit=float(standard))
            scoring_temporal = row['temporal']*(self.mode.admission/self.mode.scoring)**2
            row['strength'] = float(np.clip(logit, -6, 6))-1.2*scoring_temporal-.5*row['velocity']
            row['eligible'] = bool(logit >= cutoff(.80) and row['temporal'] <= .85 and row['velocity'] <= 1.)
            row['selected'] = False
        self.counters['prebirth_proposals'] += len(rows)
        return rows

    def vectorize(self, rows, scan, positions, points, scores):
        by_new, by_old = defaultdict(list), defaultdict(list)
        for row in rows:
            by_new[row['new']].append(row)
            by_old[row['old']].append(row)
        new_features = {}
        for new, group in by_new.items():
            index = group[0]['index']
            from experiments.id_recovery.tracker import Retired
            saved = Retired(self.state['tracks'][new], self.history[new], self.context_history[new],
                            self.memories.get(new, Memory()), self.strong_hits[new], self.quality_hits[new],
                            self.birth_scan[new], 0, scan)
            elapsed = scan-saved.track.last_scan
            radius = min(1.+.2*(elapsed-1), 1.6)
            predicted = np.asarray(saved.track.position)+np.asarray(saved.track.velocity)*elapsed
            cost = float(np.square(np.asarray(positions[index])-predicted).sum())/radius**2+.5*(1-scores[index])
            new_features[new] = pair_features(self, saved, [index], [cost], radius, points, scores, scan)[0]
        for row in rows:
            old, new, index, saved = row['old'], row['new'], row['index'], row['saved']
            prefix = [*self.history[new][-2:], (points[index], scores[index], scan)]
            xyz = np.asarray([[p[k]/scale for k, scale in zip(POSITION, self.config.gate)] for p, _, _ in prefix])
            times = np.asarray([when for _, _, when in prefix], dtype=float)
            residual = xyz-np.asarray(saved.track.position)-(times-saved.track.last_scan)[:, None]*np.asarray(saved.track.velocity)
            centered = times-times.mean()
            velocity = (centered[:, None]*(xyz-xyz.mean(0))).sum(0)/max(float(np.square(centered).sum()), 1.)
            mean_signal = np.mean([[p['Amp'], p['SNR']] for p, _, _ in saved.history], axis=0)
            signal = np.asarray([[p['Amp'], p['SNR']] for p, _, _ in prefix])-mean_signal
            alternatives = [r['logit'] for r in by_new[new] if r is not row]
            old_alternatives = [r['logit'] for r in by_old[old] if r is not row]
            extra = [*(residual/4).ravel(), *((times-scan)/8), *(score for _, score, _ in prefix),
                     *signal.ravel(), *((velocity-np.asarray(saved.track.velocity))/4),
                     row['temporal']*self.mode.admission**2, row['current']*self.mode.admission**2,
                     float(np.square(residual).sum(1).max())/(row['radius']/self.mode.admission)**2, row['velocity'],
                     row['logit'], row['canonical_logit'], (scan-saved.track.last_scan)/14,
                     (scan-saved.last_emission)/14, (scan-self.birth_scan[new])/14,
                     log1p(saved.emission_count), saved.emission_count/max(1, saved.track.hits),
                     float(row['retired']), float(saved.emission_count<3), len(saved.memory.entries)/16,
                     len(by_new[new])/MAX_OLD, (row['logit']-max(alternatives, default=0.))/6,
                     (row['logit']-max(old_alternatives, default=0.))/6]
            assert len(extra) == EXTRA_DIM, len(extra)
            row['features'] = np.concatenate((row['old_features'], new_features[new], np.asarray(extra, dtype=np.float32)))
            assert row['features'].shape == (DIMENSION,) and np.isfinite(row['features']).all()

    def intercept(self, matched, used, scan, positions, points, scores):
        rows = self.candidates(matched, used, scan, positions, points, scores)
        if not rows:
            return
        self.vectorize(rows, scan, positions, points, scores)
        self.samples.extend(rows)
        if self.mode.learned:
            assert self.continuation_model is not None and self.threshold is not None
            by_new = defaultdict(list)
            for row in rows:
                by_new[row['new']].append(row)
                row['eligible'] = False
            retained = [row for group in by_new.values() for row in sorted(group, key=lambda r:r['strength'], reverse=True)[:MAX_OLD]]
            for row in rows:
                row['strength'] = -float('inf')
            logits = self.continuation_model.score(np.stack([row['features'] for row in retained]))
            for row, logit in zip(retained, logits, strict=True):
                row['strength'] = float(logit)
                row['eligible'] = bool(logit >= cutoff(self.threshold))
        options, competing = defaultdict(list), defaultdict(list)
        for row in rows:
            self.proposals.append(row)
            self.counters['prebirth_eligible'] += row['eligible']
            options[row['old']].append(row)
            competing[row['new']].append(row)
        ranked = []
        for old, choices in options.items():
            choices.sort(key=lambda row:row['strength'], reverse=True)
            best = choices[0]
            if not best['eligible']:
                continue
            if len(choices)>1 and best['strength']-choices[1]['strength']<.25:
                continue
            if any(row['old']!=old and best['strength']-row['strength']<.25 for row in competing[best['new']]):
                continue
            ranked.append(best)
        for row in sorted(ranked, key=lambda r:r['strength'], reverse=True):
            old, new, index = row['old'], row['new'], row['index']
            if old in used or matched.get(new)!=index:
                continue
            if row['retired']:
                self.restore(old)
            del matched[new]
            matched[old]=index
            used.add(old)
            row['selected']=True
            name='continuation_retired' if row['retired'] else 'continuation_live'
            self.decisions.append((name, old, index, new))
