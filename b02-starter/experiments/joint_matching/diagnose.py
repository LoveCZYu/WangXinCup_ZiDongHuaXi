"""Read-only attribution and fixed-output identity counterfactuals on calibration."""
from collections import Counter, defaultdict

from b02.algorithm import POSITION


class FragmentationAudit:
    def __init__(self):
        self.support = defaultdict(Counter)
        self.batch_truth = defaultdict(Counter)
        self.first = {}
        self.hits = []

    def observe(self, circle, points, labels, outputs, tracker, before):
        events = {event['batch']: event for event in tracker.events}
        lookup = {point['PntID']: i for i, point in enumerate(points)}
        indexed = {index: batch for batch, index in tracker.last_matches.items()}
        matched = []
        for output in outputs:
            truth = labels[output['PntID']]
            if truth <= 0:
                continue
            batch = output['BatchID']
            index = lookup[output['PntID']]
            counts = self.support[truth]
            old = max(counts, key=counts.get) if counts else None
            cause = 'first_detection'
            if old is not None and batch != old:
                matched_index = tracker.last_matches.get(old)
                matched_truth = labels[points[matched_index]['PntID']] if matched_index is not None else None
                event = events.get(old, {})
                if old not in before:
                    cause = 'old_id_absent'
                elif index in event.get('indices', []):
                    cause = 'surviving_candidate'
                elif index in event.get('eligible_indices', []):
                    cause = 'top_k_excluded'
                elif index in event.get('gated_indices', []):
                    cause = 'cost_gate_excluded'
                else:
                    predicted, radius = before[old]
                    squared = sum((points[index][key]/scale - p)**2 for key, scale, p in zip(POSITION, tracker.config.gate, predicted))
                    if squared <= radius**2:
                        cause = 'taken_in_earlier_stage'
                    elif squared <= (1.5*radius)**2:
                        cause = 'outside_gate_near'
                    else:
                        cause = 'outside_gate_far'
                rival = 'unmatched' if matched_truth is None else ('same_truth' if matched_truth == truth else 'other_truth' if matched_truth > 0 else 'clutter_or_unknown')
            else:
                rival = None
            event_info = {'cause': cause, 'circle': circle, 'old': old, 'receiver': indexed.get(index),
                          'old_match': rival, 'new_id_prior_other_truth': any(value != truth for value in self.batch_truth[batch]),
                          'old_support': counts.get(old, 0)}
            self.first.setdefault((truth, batch), event_info)
            self.hits.append((circle, truth, batch))
            matched.append((truth, batch))
        for truth, batch in matched:
            self.support[truth][batch] += 1
            self.batch_truth[batch][truth] += 1

    def snapshot(self, tracker):
        needed = {max(values, key=values.get) for values in self.support.values() if values}
        scan = tracker.state.get('scan', 0) + 1
        result = {}
        for batch, track in tracker.state.get('tracks', {}).items():
            if batch in needed:
                elapsed = scan-track.last_scan
                result[batch] = (tuple(x+v*elapsed for x, v in zip(track.position, track.velocity)), min(1+.2*(elapsed-1), 1.6))
        return result

    def result(self):
        histogram, counts = Counter(), Counter()
        fragments = []
        for truth, ids in self.support.items():
            main = max(ids, key=ids.get)
            histogram[len(ids)] += 1
            counts['dominant_90pct_and_extra'] += len(ids) > 1 and 10*ids[main] >= 9*sum(ids.values())
            for batch, size in ids.items():
                if batch != main:
                    row = {**self.first[truth, batch], 'truth': truth, 'batch': batch, 'main': main, 'points': size}
                    fragments.append(row)
        return {'histogram': dict(histogram), 'short_fragment_counts': {str(limit): sum(row['points'] <= limit for row in fragments) for limit in (1, 3, 5, 10)},
                'dominant_90pct_and_extra': counts['dominant_90pct_and_extra'],
                'first_causes': dict(Counter(row['cause'] for row in fragments)), 'fragments': fragments}

    def repair(self, outputs, frames, fragments, allowed=None, max_size=None):
        """Diagnostic only: retroactive secondary-ID repair on identical output points."""
        mappings = {(f['truth'], f['batch']): f['main'] for f in fragments
                    if (allowed is None or f['cause'] in allowed) and (max_size is None or f['points'] <= max_size)}
        changed = conflicts = 0
        repaired = []
        for prediction, (_, _, labels) in zip(outputs, frames, strict=True):
            occupied = {p['BatchID'] for p in prediction}
            current = []
            for point in prediction:
                old = point['BatchID']
                t = labels[point['PntID']]
                new = mappings.get((t, old), old)
                if new != old:
                    if new in occupied:
                        conflicts += 1
                    else:
                        occupied.add(new)
                        changed += 1
                        point = {**point, 'BatchID': new}
                current.append(point)
            repaired.append(current)
        return repaired, {'mapped_fragments': len(mappings), 'reassigned_points': changed, 'collision_skips': conflicts}


IN_GATE = {'surviving_candidate'}
WITH_GATE = IN_GATE | {'outside_gate_near', 'top_k_excluded', 'cost_gate_excluded', 'taken_in_earlier_stage'}
