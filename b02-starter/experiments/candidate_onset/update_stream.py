"""Census and counterfactual replay of the published high-affinity update stream."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import numpy as np
import torch

from b02.replay import _scans, open_scenes, replay_frames
from experiments.candidate_onset.mixed import install
from experiments.candidate_onset.propagation import MIN_AFFINITY, anchor, restore, sha
from experiments.candidate_onset.propagation_summary import features
from experiments.candidate_onset.propagation_work import capture_prefix
from experiments.id_recovery.affinity import pack
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.v2_association.run_stage1 import Diagnostic, save

CLASSES = ('unresolved', 'correct', 'other_target', 'clutter')


def published_edges(tracker, points, scores, eligible, output):
    positions = {point['PntID']: i for i, point in enumerate(points)}
    published = {item['BatchID']: positions[item['PntID']] for item in output
                 if item['BatchID'] in eligible}
    accepted = {}
    for event in tracker.events:
        batch = event['batch']
        index = published.get(batch)
        if (index is not None and index == tracker.last_matches.get(batch)
                and scores[index] >= .525 and index in event['indices']):
            accepted[batch] = index, event, event['indices'].index(index)
    if not accepted:
        return [], np.empty((0, 10))
    tx = np.stack([event['track_features'] for _, event, _ in accepted.values()])
    ex = np.stack([event['edge_features'][j] for _, event, j in accepted.values()])
    affinities = tracker.affinity.score(pack(tx, ex))
    keys, rows = [], []
    for (batch, (index, event, _)), affinity, track, edge in zip(
            accepted.items(), affinities, tx, ex, strict=True):
        if affinity >= MIN_AFFINITY:
            row = features(dict(affinity=float(affinity), detector_score=float(scores[index]),
                                track_features=track, edge_features=edge,
                                candidate_count=len(event['indices'])))
            keys.append((batch, index))
            rows.append(row)
    return keys, np.asarray(rows).reshape(-1, 10)


def update_class(history, truth):
    expected = anchor(history)
    if expected is None or truth < 0:
        return 0
    if truth == expected:
        return 1
    return 2 if truth > 0 else 3


def risk(model, x):
    coef = np.asarray(model['coefficients'])
    return ((x - model['mean']) / model['scale']) @ coef[:-1] + coef[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--role', choices=('development', 'work'), required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--policy', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.protocol.read_bytes())
    root = Path(__file__).resolve().parents[2]
    for name, digest in plan['sources'].items():
        assert sha(root / name) == digest, name
    assert sha(args.data) == plan[args.role]['data_sha256']
    assert sha(args.freeze) == plan['parent_freeze_sha256']
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
    report = dict(scene=scene, protocol_sha256=sha(args.protocol), variants={})
    if args.policy is None:
        modes = {'census': False}
        model = None
    else:
        assert args.role == 'work'
        model = json.loads(args.policy.read_bytes())
        assert model['protocol_sha256'] == report['protocol_sha256']
        report['policy_sha256'] = sha(args.policy)
        modes = dict(motion=False, full=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for mode, full in modes.items():
        tracker = install(components.build('candidate'), 'combined')
        histories = defaultdict(lambda: deque(maxlen=8))
        counts = Counter()
        birth, diagnostic = BirthAudit(), Diagnostic()
        stored = []
        index = 0

        def run(points, _):
            nonlocal index
            circle, _, labels = frames[index]
            scan_scores = scores[offsets[index]:offsets[index + 1]].tolist()
            index += 1
            active = set(tracker.state.get('tracks', {}))
            eligible = {batch for batch in active if tracker.emissions.get(batch, 0) >= 4
                        and len(tracker.history.get(batch, ())) >= 4}
            prefix = capture_prefix(tracker, eligible) if model is not None else None
            output = tracker(points, scan_scores)
            keys, x = published_edges(tracker, points, scan_scores, eligible, output)
            flags = risk(model, x) > model['threshold'] if model is not None else np.zeros(len(x), dtype=bool)
            for (batch, j), features_row, flagged in zip(keys, x, flags, strict=True):
                kind = update_class(histories.get(batch, ()), labels[points[j]['PntID']])
                counts['eligible_' + CLASSES[kind]] += 1
                if model is None:
                    stored.append((kind, circle, batch, points[j]['PntID'], *features_row))
                elif flagged and restore(tracker, prefix, batch, full=full):
                    counts['rollback_' + CLASSES[kind]] += 1
            for point in output:
                truth = labels[point['PntID']]
                if truth >= 0:
                    histories[point['BatchID']].append(truth)
            birth.observe(points, labels, output, tracker, circle)
            diagnostic.observe(points, labels, output, tracker, active, circle)
            return output

        trace_path = args.out / f'{scene}-{mode}.jsonl'
        with trace_path.open('w', encoding='utf-8') as trace:
            summary = replay_frames(frames, trace, run)
        if model is None:
            np.savez_compressed(args.out / (scene + '.npz'), updates=np.asarray(stored).reshape(-1, 14))
            trace_path.unlink()
        report['variants'][mode] = dict(summary=summary, operations=dict(counts),
                                       identity=diagnostic.result()['counts'], birth_audit=birth.result())
        print(scene, mode, dict(counts), flush=True)
    save(args.out / (scene + '.json'), report)


if __name__ == '__main__':
    main()
