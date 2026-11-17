"""Attribute the frozen combination's minority outputs to episodes and stages."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import numpy as np
import torch

from b02.replay import _scans, open_scenes, replay_frames
from experiments.candidate_onset.audit import classify
from experiments.candidate_onset.mixed import install
from experiments.candidate_onset.observe import build
from experiments.candidate_onset.propagation import sha
from experiments.candidate_onset.update_stream_report import TIMINGS
from experiments.robust_validation.frozen import Components
from experiments.v2_association.run_stage1 import Diagnostic, Discard, save


def owners(summary):
    hist = defaultdict(Counter)
    for target in summary['per_target']:
        for batch, count in target['output_ids'].items():
            hist[int(batch)][target['truth_id']] += count
    # A deterministic tie break preserves the scorer's minority point total.
    # These tracks remain unassigned to a causal category.
    return {b: (min(labels, key=lambda t: (-labels[t], t)),
                sum(c == max(labels.values()) for c in labels.values()) > 1)
            for b, labels in hist.items() if len(labels) > 1}


def opportunity(batch, truth, right, before, tracker, scores, owners_of_indices):
    if not right:
        return 'expected_point_absent', 'expected_point_absent', False
    reason, sources = classify(batch, right, before, tracker, scores)
    if any(sources.values()):
        taken = any(i in owners_of_indices and owners_of_indices[i] != batch for i in right)
        return 'available_but_not_selected', reason, taken
    return 'missing_candidate', reason, False


def split_episodes(rows, missed_scans):
    """Count repeated wrong identities, with a fixed tolerated observation gap."""
    episodes = []
    open_episode = {}
    for row in rows:
        batch, truth = row['batch'], row['truth']
        if truth == row['expected']:
            open_episode.pop(batch, None)
            continue
        prev = open_episode.get(batch)
        if (prev is None or prev['wrong_truth'] != truth
                or row['frame'] - prev['end'] > missed_scans + 1):
            prev = dict(batch=batch, wrong_truth=truth, expected=row['expected'], start=row['frame'],
                        end=row['frame'], count=0, onset=row, reciprocal=0)
            episodes.append(prev)
            open_episode[batch] = prev
        prev['count'] += 1
        prev['end'] = row['frame']
        prev['reciprocal'] += row['reciprocal']
    return episodes


def report_episodes(episodes, long_min):
    start, length_weighted, decomposition = (Counter() for _ in range(3))
    long_points = cross_points = 0
    for event in episodes:
        first = event['onset']
        basis = ('ambiguous_dominant' if first['tied'] else 'unestablished_identity'
                 if not first['seen_expected'] else first['category'])
        start[basis] += 1
        length_weighted[basis] += event['count']
        decomposition[basis] += 1
        decomposition['continued'] += event['count'] - 1
        if event['count'] >= long_min:
            long_points += event['count'] - 1
            if event['reciprocal']:
                cross_points += event['count'] - 1
    return dict(starts=dict(start), weighted_from_first_error=dict(length_weighted),
                decomposition=dict(decomposition), continued_long=long_points,
                continued_long_reciprocal=cross_points)


def collect(frames, cached, tracker, reference, plan):
    dom = owners(reference['summary'])
    records, frozen_outputs = [], []
    histories = defaultdict(lambda: deque(maxlen=8))
    seen = set()
    diagnostic = Diagnostic()
    index = 0

    def run(points, _):
        nonlocal index
        circle, _, labels = frames[index]
        scores = cached(index)
        prior = {b: {'bank': 'active'} for b in tracker.state.get('tracks', {}) if b in dom}
        for bank in ('retired', 'short_retired'):
            for b in getattr(tracker, bank):
                if b in dom:
                    prior.setdefault(b, {'bank': bank})
        output = tracker(points, scores)
        frozen_outputs.append([(p['BatchID'], p['PntID']) for p in output])
        raw = defaultdict(list)
        for i, point in enumerate(points):
            label = labels[point['PntID']]
            if label > 0:
                raw[label].append(i)
        truth_by_batch = {p['BatchID']: labels[p['PntID']] for p in output}
        all_owners = {i:b for b,i in tracker.last_matches.items()}
        output_by_truth = defaultdict(list)
        for batch, true in truth_by_batch.items():
            output_by_truth[true].append(batch)
        for point in output:
            b = point['BatchID']
            true = truth_by_batch[b]
            if true <= 0 or b not in dom:
                continue
            want, tied = dom[b]
            row = dict(frame=index, circle=circle, batch=b, expected=want, truth=true, tied=tied,
                       seen_expected=b in seen, pure_prefix=len(histories[b]) >= 4 and set(histories[b]) == {want})
            if true == want:
                seen.add(b)
            else:
                status, reason, occupied = opportunity(b, want, raw[want], prior, tracker, scores, all_owners)
                counterpart = any(other in dom and dom[other][0] == true and not dom[other][1]
                                  for other in output_by_truth[want] if other != b)
                row.update(category=status, reason=reason, true_point_taken=occupied, reciprocal=counterpart,
                           active_before=prior.get(b, {}).get('bank') == 'active', point=point['PntID'],
                           internal_same=tracker.last_matches.get(b) is not None and
                           points[tracker.last_matches[b]]['PntID'] == point['PntID'])
            records.append(row)
            histories[b].append(true)
        diagnostic.observe(points, labels, output, tracker, set(prior), circle)
        index += 1
        return output

    summary = replay_frames(frames, Discard(), run)
    observed = json.loads(json.dumps(summary))
    different = [key for key in observed if key not in TIMINGS and observed[key] != reference['summary'][key]]
    assert not different, different
    mixed = diagnostic.result()['counts']['mixed_minority_truth_points']
    assert mixed == reference['identity']['mixed_minority_truth_points']
    minority = [row for row in records if row['truth'] != row['expected']]
    assert len(minority) == mixed
    rollups = {}
    for gap in plan['gaps']:
        episodes = split_episodes(records, gap)
        rollup = report_episodes(episodes, plan['long_min'])
        assert sum(rollup['decomposition'].values()) == mixed
        rollups[str(gap)] = rollup
    episodes = split_episodes(records, plan['main_gap'])
    return dict(mixed=mixed, rollups=rollups, current_state_counts=Counter(
        ('ambiguous_dominant' if row['tied'] else 'unestablished_identity' if not row['seen_expected'] else row['category'])
        for row in minority), detail_counts=Counter(row['reason'] for row in minority), episodes=episodes,
        frozen_outputs=frozen_outputs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'reference', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.protocol.read_bytes())
    root = Path(__file__).resolve().parents[2]
    assert all(sha(root / key) == h for key,h in plan['sources'].items())
    assert sha(args.data) == plan['data_sha256'] and sha(args.freeze) == plan['parent_freeze_sha256']
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    split = json.loads((root/'splits/v3_tracking.json').read_bytes())
    names = plan['scenes']
    assert set(names) == set(components.freeze['work_validation16']) and not set(names) & set(split['sealed_excluded'])
    scene = names[args.index]
    reference = json.loads((args.reference / (scene+'.json')).read_bytes())
    assert reference['scene'] == scene and reference['protocol_sha256'] == plan['census_protocol_sha256']
    manifest = json.loads((args.cache/'manifest.json').read_bytes())
    path = args.cache / (scene+'.npz')
    assert sha(path) == manifest['scenes'][scene]['cache_sha256']
    with open_scenes(args.data) as (_, opener), opener(scene) as source:
        frames = list(_scans(source))
    with np.load(path) as cache:
        scores, offsets = cache['point_scores'], cache['point_offsets']
    assert len(offsets) == len(frames) + 1

    def cached(i):
        return scores[offsets[i]:offsets[i+1]].tolist()

    report = collect(frames, cached, install(build(components), 'combined'), reference['variants']['census'], plan)
    report.update(scene=scene, protocol_sha256=sha(args.protocol))
    save(args.out / (scene+'.json'), report)
    print(scene, report['mixed'], report['rollups'][str(plan['main_gap'])]['decomposition'], flush=True)


if __name__ == '__main__':
    main()
