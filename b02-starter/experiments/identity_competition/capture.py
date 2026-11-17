"""Capture reproducible switch onsets on fitting scene train_093 only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.replay import _scans, open_scenes
from experiments.robust_validation.frozen import Components
from experiments.v2_association.run_stage1 import save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    scene = 'train_093'
    assert scene in plan['fit64'] and scene not in plan['sealed_excluded']
    cache = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['complete'] and cache['data_sha256'] == sha(args.data)
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    tracker = components.build('candidate')
    last = {}
    counts = Counter()
    with open_scenes(args.data) as (_, opener):
        with opener(scene) as source:
            frames = list(_scans(source))
    with np.load(args.cache/'fit'/f'{scene}.npz') as cached:
        scores, offsets = cached['point_scores'], cached['point_offsets']
    for frame, (circle, points, labels) in enumerate(frames):
        before = pickle.dumps(tracker, protocol=5)
        values = scores[offsets[frame]:offsets[frame+1]].tolist()
        prediction = tracker(points, values)
        lookup = {p['PntID']: i for i, p in enumerate(points)}
        unique = Counter(labels[p['PntID']] for p in points)
        edges = {e['batch']: set(e['indices']) for e in tracker.events}
        for point in prediction:
            truth = labels[point['PntID']]
            if truth <= 0 or unique[truth] != 1:
                continue
            batch = point['BatchID']
            previous = last.get(truth)
            if previous:
                old, when, run = previous
                if batch != old and frame-when <= 2 and run >= 3 and lookup[point['PntID']] in edges.get(old, ()):
                    path = args.out/f'case-{counts["captured"]:02d}.pkl'
                    raw = pickle.dumps({'before': before, 'points': points, 'scores': values,
                        'labels': labels, 'observed_outputs': prediction}, protocol=5)
                    path.write_bytes(raw)
                    save(path.with_suffix('.json'), {'scene': scene, 'circle': circle, 'frame': frame,
                        'truth_id_diagnostic_only': truth, 'point_id': point['PntID'],
                        'expected_previous_id': old, 'observed_new_id': batch,
                        'prior_same_id_hits': run, 'observation_gap': frame-when,
                        'fixture_sha256': hashlib.sha256(raw).hexdigest(), 'data_sha256': sha(args.data),
                        'freeze_sha256': sha(args.freeze), 'fitting_pool_only': True})
                    counts['captured'] += 1
                    print('captured', path.name, circle, old, batch, flush=True)
            last[truth] = (batch, frame, previous[2]+1 if previous and previous[0] == batch else 1)
        if counts['captured'] >= 2:
            break
    save(args.out/'capture-report.json', {'scene': scene, 'counts': dict(counts), 'new_training': False})
    assert counts['captured'] > 0, 'No suitable onset found; do not invent a failing fixture'


if __name__ == '__main__':
    main()
