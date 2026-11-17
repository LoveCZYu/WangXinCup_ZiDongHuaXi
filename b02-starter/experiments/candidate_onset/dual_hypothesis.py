"""Replay two local assignments from the same causal track prefix."""
import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from b02.algorithm import POSITION
from b02.replay import _scans, open_scenes
from experiments.candidate_onset.mixed import install
from experiments.candidate_onset.propagation import anchor, sha
from experiments.candidate_onset.update_stream import published_edges, risk, update_class
from experiments.robust_validation.frozen import Components
from experiments.v2_association.run_stage1 import save


@dataclass(frozen=True)
class TrackState:
    position: np.ndarray
    velocity: np.ndarray
    signal: np.ndarray
    last_scan: int

    def update(self, scan, observation):
        position, signal, _ = observation
        velocity = .65 * self.velocity + .35 * (position - self.position) / (scan - self.last_scan)
        return TrackState(position, velocity, self.signal, scan)

    def costs(self, scan, observations):
        xyz, signal, score = observations
        gap = scan - self.last_scan
        radius = min(1 + .2 * (gap - 1), 1.6)
        predicted = self.position + self.velocity * gap
        residual = np.square(xyz - predicted).sum(1) / radius**2
        signature = np.abs(signal - self.signal).mean(1)
        cost = residual + .30 * np.minimum(signature, 2) + .25 * (1 - score)
        return np.where(residual <= 1, cost, np.inf)


def observe(points, scores, gate):
    xyz = np.asarray([[p[key] / scale for key, scale in zip(POSITION, gate)] for p in points]).reshape(-1, 3)
    return xyz, np.asarray([[p['Amp'], p['SNR']] for p in points]).reshape(-1, 2), np.asarray(scores)


def take(observations, index):
    return tuple(a[index] for a in observations)


def advance(states, scan, observations, forbidden):
    """Find the best two-row assignment including independent miss options."""
    choices = []
    for state in states:
        costs = state.costs(scan, observations)
        allowed = [(float(costs[i]), i) for i in np.argsort(costs, kind='stable')[:2] if costs[i] < 1.15]
        allowed.append((1.15, -1))
        choices.append(allowed)
    cost, a, b = min((ac + bc, a, b) for ac, a in choices[0] for bc, b in choices[1] if a != b or a == -1)
    next_states = tuple(state if index == -1 else state.update(scan, take(observations, index))
                        for state, index in zip(states, (a, b), strict=True))
    return next_states, cost, tuple(None if i == -1 else int(forbidden[i]) for i in (a, b))


def trajectory(states, scan, start, assignment, future):
    states = tuple(state if i is None else state.update(scan, take(start, i))
                   for state, i in zip(states, assignment, strict=True))
    steps = []
    for delta, (obs, indices) in enumerate(future, 1):
        states, cost, matched = advance(states, scan + delta, obs, indices)
        steps.append((cost, matched))
    return steps


def make_states(tracker):
    return {batch: TrackState(np.asarray(track.position), np.asarray(track.velocity),
                              np.mean([[p['Amp'], p['SNR']] for p, _, _ in tracker.history[batch][-4:]], axis=0),
                              track.last_scan)
            for batch, track in tracker.state.get('tracks', {}).items()
            if tracker.emissions.get(batch, 0) >= 4 and len(tracker.history.get(batch, ())) >= 4}


def alternative(events, matches, mature, winner, index):
    """Choose one rival using only the solver's recorded current-scan costs."""
    chosen = None
    winner_row = events.get(winner, {})
    for rival, row in events.items():
        if rival == winner or rival not in mature or index not in row['indices']:
            continue
        old = matches.get(rival)
        swap = old is not None and old in winner_row['indices']
        if old is not None and old not in row['indices']:
            continue
        gain = cost(row, index) - cost(row, old)
        gain += cost(winner_row, old if swap else None) - cost(winner_row, index)
        item = (gain, int(rival), old, swap)
        if chosen is None or item[:2] < chosen[:2]:
            chosen = item
    return chosen


def cost(row, index):
    if index is None:
        return row['effective_dummy']
    return row['effective_costs'][row['indices'].index(index)]


def collect(frames, raw_scores, offsets, tracker, model, horizon):
    history = defaultdict(lambda: deque(maxlen=8))
    counts = Counter()
    pending, rows = defaultdict(list), []
    for t, (circle, points, labels) in enumerate(frames):
        scores = raw_scores[offsets[t]:offsets[t + 1]].tolist()
        # Capture only eligible state, before the ordinary tracker can update it.
        initial = make_states(tracker)
        output = tracker(points, scores)
        keys, x = published_edges(tracker, points, scores, initial, output)
        xyz = observe(points, scores, tracker.config.gate)
        # Keep the background fixed to the ordinary replay. Both hypotheses can
        # use observations claimed by either focal ID, and unassigned observations.
        matches = tracker.last_matches
        for row, start, assignment in pending.pop(t, ()):
            winner, rival = row['pair']
            occupied = {j for batch, j in matches.items() if batch not in row['pair']}
            keep = np.asarray([j for j, score in enumerate(scores) if score >= .525 and j not in occupied], dtype=int)
            row['future'].append((take(xyz, keep), keep))
            if len(row['future']) < horizon:
                pending[t + 1].append((row, start, assignment))
                continue
            future = row.pop('future')
            initial_states, scan, start_obs = start
            baseline, challenger = (trajectory(initial_states, scan, start_obs, arm, future) for arm in assignment)
            row['future_advantage'] = float(sum(a[0] - b[0] for a, b in zip(baseline, challenger, strict=True)))
            row['two_hypothesis_score'] = row['future_advantage'] - .50 * row['onset_gap']
            row['steps'] = [[list(step) for step in arm] for arm in (baseline, challenger)]
            rows.append(row)

        events = {event['batch']: event for event in tracker.events if 'effective_costs' in event}
        single = risk(model, x)
        for (winner, index), flag_score in zip(keys, single, strict=True):
            primary = labels[points[index]['PntID']]
            kind = update_class(history.get(winner, ()), primary)
            counts[f'eligible_{kind}'] += 1
            if t + horizon >= len(frames) or winner not in events or index not in events[winner]['indices']:
                continue
            proposal = alternative(events, matches, initial, winner, index)
            if proposal is None:
                continue
            gap, rival, old, swap = proposal
            rival_truth = anchor(history.get(rival, ()))
            original_truth = anchor(history.get(winner, ()))
            useful = bool(kind == 2 and rival_truth == primary)
            if useful and old is not None:
                old_truth = labels[points[old]['PntID']]
                useful = old_truth >= 0 and (not swap or old_truth == original_truth)
            row = dict(circle=circle, index=t, pair=[int(winner), rival], point=int(points[index]['PntID']),
                       primary_truth=primary, original_truth=original_truth, rival_truth=rival_truth,
                       kind=kind, recoverable=useful, swap=swap, onset_gap=float(gap),
                       single_score=float(flag_score), future=[])
            start = ((initial[winner], initial[rival]), tracker.state['scan'], xyz)
            pending[t + 1].append((row, start, ((index, old), (old if swap else None, index))))
        for item in output:
            label = labels[item['PntID']]
            if label >= 0:
                history[item['BatchID']].append(label)
    assert not pending
    return dict(counts), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'policy', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--role', choices=('development', 'work'), required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.protocol.read_bytes())
    root = Path(__file__).resolve().parents[2]
    assert all(sha(root / name) == digest for name, digest in plan['sources'].items())
    assert sha(args.policy) == plan['policy_sha256'] and sha(args.freeze) == plan['parent_freeze_sha256']
    assert sha(args.data) == plan[args.role]['data_sha256']
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    split = json.loads((root / 'splits/v3_tracking.json').read_bytes())
    names = plan[args.role]['scenes']
    allowed = split['fit64'] if args.role == 'development' else components.freeze['work_validation16']
    assert set(names) <= set(allowed) and not set(names) & set(split['sealed_excluded'])
    scene = names[args.index]
    cache_path = args.cache / (scene + '.npz')
    manifest = json.loads((args.cache / 'manifest.json').read_bytes())
    assert sha(cache_path) == manifest['scenes'][scene]['cache_sha256']
    with open_scenes(args.data) as (_, opener), opener(scene) as source:
        frames = list(_scans(source))
    with np.load(cache_path) as cache:
        scores, offsets = cache['point_scores'], cache['point_offsets']
    assert len(offsets) == len(frames) + 1
    model = json.loads(args.policy.read_bytes())
    tracker = install(components.build('candidate'), 'combined')
    counts, rows = collect(frames, scores, offsets, tracker, model, plan['horizon'])
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / (scene + '.json'), dict(scene=scene, role=args.role, counts=counts, rows=rows,
                                          protocol_sha256=sha(args.protocol)))
    print(scene, counts, 'contests', len(rows), 'recoverable', sum(r['recoverable'] for r in rows), flush=True)


if __name__ == '__main__':
    main()
