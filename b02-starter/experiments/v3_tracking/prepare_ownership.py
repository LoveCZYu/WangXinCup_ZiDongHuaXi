"""Capture competition for the same observation on causal memory-model rollouts."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.v2_association.run_stage1 import historical_identity, save, sha
from experiments.v3_tracking.ownership import EDGE_DIM, TRACK_DIM, OwnershipTracker
from experiments.v3_tracking.runtime import load_scorer

MAX_OWNERS = 8


def examples(graph, points, scores, labels, histories, emitted, counts):
    columns = defaultdict(list)
    for row, (batch, indices) in enumerate(zip(graph['batches'], graph['indices'])):
        identity = historical_identity(histories.get(batch, ()))
        for edge, index in enumerate(indices):
            truth = labels[points[index]['PntID']]
            if scores[index] >= .5 and truth > 0 and identity is not None:
                support = emitted.get(batch, {}).get(truth, 0)
                columns[index].append((graph['prior'][row][edge], row, edge, identity == truth, support))
    for competitors in columns.values():
        if len(competitors) < 2:
            continue
        candidates = sorted(competitors)[:MAX_OWNERS]
        support = [item[4] for item in candidates if item[3]]
        if not support:
            continue
        maximum = max(support)
        targets = [item[3] and item[4] == maximum for item in candidates]
        if all(targets):
            continue
        track = np.zeros((MAX_OWNERS, TRACK_DIM), dtype=np.float32)
        edge = np.zeros((MAX_OWNERS, EDGE_DIM), dtype=np.float32)
        prior = np.zeros(MAX_OWNERS, dtype=np.float32)
        mask = np.arange(MAX_OWNERS) < len(candidates)
        target = np.zeros(MAX_OWNERS, dtype=bool)
        target[:len(candidates)] = targets
        for i, (cost, row, j, _, _) in enumerate(candidates):
            track[i], edge[i], prior[i] = graph['track'][row], graph['edges'][row][j], cost
        counts['columns'] += 1
        counts['prior_wrong'] += not targets[0]
        counts['has_single_candidate_track'] += any(len(graph['indices'][item[1]]) == 1 for item in candidates)
        yield track, edge, prior, mask, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--shards', type=int, required=True)
    args = parser.parse_args()
    import torch
    from b02.replay import _scans, open_scenes
    torch.set_num_threads(2)
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    cache = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['fit'] == plan['fit64'] and cache['calibration'] == plan['calibration']
    assert cache['data_sha256'] == sha(args.data) and 0 <= args.index < args.shards
    scorer, memory, _, _ = load_scorer(args.prior)
    assert memory
    args.out.mkdir(parents=True, exist_ok=True)
    record = {'scenes': {}, 'prior_sha256': sha(args.prior), 'split_sha256': sha(ROOT/'splits/v3_tracking.json')}
    names = plan['fit64'] + plan['calibration']
    with open_scenes(args.data) as (_, opener):
        for name in names[args.index::args.shards]:
            role = 'fit' if name in plan['fit64'] else 'calibration'
            with opener(name) as source:
                frames = list(_scans(source))
            with np.load(args.cache/role/f'{name}.npz') as data:
                scores, offsets = data['point_scores'], data['point_offsets']
            assert len(offsets) == len(frames) + 1
            starts = np.linspace(0, len(frames)-8, 8, dtype=int)
            sampled = {i for start in starts for i in range(start, start+8)}
            tracker = OwnershipTracker(scorer)
            histories = defaultdict(lambda: deque(maxlen=8))
            emitted = defaultdict(Counter)
            rows, counts = [], Counter()
            for i, (_, points, labels) in enumerate(frames):
                tracker.graph_capture = i in sampled
                point_scores = scores[offsets[i]:offsets[i+1]].tolist()
                outputs = tracker(points, point_scores)
                for graph in tracker.graphs:
                    rows.extend(examples(graph, points, point_scores, labels, histories, emitted, counts))
                for batch, j in (tracker.last_matches | tracker.last_births).items():
                    histories[batch].append(labels[points[j]['PntID']])
                for point in outputs:
                    emitted[point['BatchID']][labels[point['PntID']]] += 1
                for batch in tracker.deleted:
                    histories.pop(batch, None)
                    emitted.pop(batch, None)
            folder = args.out/role
            folder.mkdir(exist_ok=True)
            path = folder/f'{name}.npz'
            assert not path.exists()
            keys = ('track', 'edges', 'prior', 'valid', 'target')
            arrays = {key: np.stack(values) for key, values in zip(keys, zip(*rows))}
            np.savez_compressed(path, **arrays)
            record['scenes'][name] = dict(counts)
            print(name, dict(counts), flush=True)
    save(args.out/f'shard-{args.index}.json', record)


if __name__ == '__main__':
    main()
