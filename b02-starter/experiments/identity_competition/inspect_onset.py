"""One-variable counterfactuals on real and minimised identity-switch fixtures."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.replay import _scans, open_scenes
from experiments.identity_competition.repro import replay
from experiments.v2_association.features import normalized
from experiments.v2_association.model import Scorer
from experiments.v2_association.run_stage1 import save
from experiments.id_recovery.affinity import pack
from experiments.identity_competition.motion import memory_motion


class AbsoluteEvidence:
    def __init__(self, prior, real=False, null=False):
        self.prior = prior
        self.model = prior.model
        self.raw = Scorer(self.model, 'cpu', prior.scale, 'residual')
        self.real = real
        self.null = null

    def __call__(self, events):
        relative = self.prior(events)
        absolute = self.raw(events)
        return [(a if self.real else r, b if self.null else n)
                for (r, n), (a, b) in zip(relative, absolute, strict=True)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    with open_scenes(args.data) as (_, opener):
        with opener('train_093') as source:
            frames = list(_scans(source))
    history_labels = {(pt['CircleID'], pt['PntID']): labels[pt['PntID']]
                      for _, pts, labels in frames for pt in pts}
    memory_labels = [{normalized(pt).tobytes(): labels[pt['PntID']] for pt in pts}
                     for _, pts, labels in frames]
    result = {}
    for path in sorted(args.fixtures.glob('*.pkl')):
        manifest, packet, baseline_after, original = replay(path)
        before = pickle.loads(packet['before'])
        old, new, target = manifest['expected_previous_id'], manifest['observed_new_id'], manifest['point_id']
        rows = {}
        for batch in (old, new):
            track = before.state['tracks'].get(batch)
            event = next((e for e in baseline_after.events if e['batch'] == batch), None)
            h = before.history.get(batch, [])
            memory = before.memories.get(batch)
            row = {'active': track is not None, 'emissions': before.emissions[batch],
                   'history_labels': dict(Counter(history_labels.get((pt['CircleID'], pt['PntID']), 'not_found') for pt, _, _ in h)),
                   'memory_labels': dict(Counter(memory_labels[e[2]-1].get(e[0].tobytes(), 'not_found') for e in memory.entries)) if memory else {}}
            if track:
                row.update(hits=track.hits, age=before.state['scan']+1-track.last_scan,
                           velocity=list(map(float, track.velocity)))
            if event:
                evidence = before.model_scorer([event])[0]
                absolute = Scorer(before.model_scorer.model, 'cpu', 1., 'residual')([event])[0]
                affinity = before.affinity.score(pack(np.stack([event['track_features']]*len(event['indices'])), event['edge_features']))
                row['edges'] = [{'point': packet['points'][idx]['PntID'], 'label': packet['labels'][packet['points'][idx]['PntID']],
                    'base_cost': float(event['base_costs'][j]), 'effective_cost': float(event['effective_costs'][j]),
                    'dummy': float(event['effective_dummy']), 'relative_delta': float(evidence[0][j]),
                    'absolute_delta': float(absolute[0][j]), 'absolute_null': float(absolute[1])}
                    for j, idx in enumerate(event['indices'])]
                for item, value in zip(row['edges'], affinity, strict=True):
                    item['identity_affinity_logit'] = float(value)
            rows[str(batch)] = row
        tests = {}
        for mode in ('baseline', 'geometry_only', 'absolute_real', 'absolute_null', 'absolute_both', 'memory_position', 'memory_velocity'):
            tracker = deepcopy(before)
            if mode == 'geometry_only':
                tracker.model_scorer.scale = 0.
            if mode.startswith('absolute_'):
                tracker.model_scorer = AbsoluteEvidence(tracker.model_scorer,
                    real=mode in ('absolute_real', 'absolute_both'), null=mode in ('absolute_null', 'absolute_both'))
            if mode.startswith('memory_'):
                track = tracker.state['tracks'].get(old)
                memory = tracker.memories.get(old)
                motion = memory_motion(memory, track.last_scan) if track and memory else None
                if motion is not None:
                    if mode == 'memory_position': track.position = tuple(motion[0])
                    else: track.velocity = tuple(motion[1])
            prediction = tracker(packet['points'], packet['scores'])
            assigned = next((pt['BatchID'] for pt in prediction if pt['PntID'] == target), None)
            tests[mode] = {'target_id': assigned, 'same_previous_id': assigned == old,
                           'output_count': len(prediction), 'selected_points_changed':
                           {pt['PntID'] for pt in prediction} != {pt['PntID'] for pt in original}}
        result[path.stem] = {'manifest': manifest, 'scorer_strategy': before.model_scorer.strategy,
                             'rows': rows, 'counterfactuals': tests}
    save(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
