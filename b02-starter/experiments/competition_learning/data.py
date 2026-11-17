"""Capture unchanged frozen rollouts; label only in this external collector."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.replay import _scans, open_scenes
from experiments.competition_learning.features import inputs, MAX_COLUMN
from experiments.identity_competition.preflight import prefix_anchor
from experiments.robust_validation.frozen import Components
from experiments.v2_association.run_stage1 import save, sha


class Tape:
    def __init__(self, prior, tracker):
        self.prior, self.tracker, self.model = prior, tracker, prior.model
        self.calls = []

    def __call__(self, events):
        result = self.prior(events)
        rows = []
        for event in events:
            if self.tracker.emissions.get(event['batch'], 0) < 3:
                continue
            rx, ex = inputs(event, self.tracker)
            rows.append((event['batch'], tuple(event['indices']), rx, ex,
                         np.asarray(event['edge_features'][:, 11]).copy()))
        if rows:self.calls.append(rows)
        return result


def label_call(call, points, labels, histories, owners, output, counts):
    """Append one assignment stage; every feature was captured before updates."""
    rows = []
    for batch, indices, rx, ex, scores in call:
        anchor = prefix_anchor(histories.get(batch, Counter()))
        if anchor is not None:
            rows.append((batch, indices, rx, ex, scores, anchor))
    if not rows:return
    columns = defaultdict(list)
    for batch, indices, rx, ex, scores, anchor in rows:
        row_id = len(output['row'])
        output['row'].append(rx)
        truths = [labels[points[i]['PntID']] for i in indices]
        known = [truth >= 0 for truth in truths]
        positive = [truth == anchor for truth in truths]
        output['row_allowed'].append(any(positive) or all(known))
        counts['rows'] += 1
        counts['single_candidate_rows'] += len(indices) == 1
        counts['null_rows'] += all(known) and not any(positive)
        counts['unknown_ambiguous_rows'] += not any(positive) and not all(known)
        for j, (idx, truth) in enumerate(zip(indices, truths, strict=True)):
            edge_id = len(output['pair'])
            output['pair'].append(ex[j])
            output['edge_known'].append(known[j])
            output['edge_positive'].append(positive[j])
            output['edge_row'].append(row_id)
            output['col_target'].append(False)
            columns[idx].append((edge_id, batch, truth, positive[j], scores[j]))
    for index, entries in columns.items():
        group = len(output['col_allowed'])
        known = entries[0][2] >= 0
        allowed = known and len(entries) <= MAX_COLUMN
        output['col_allowed'].append(allowed)
        output['col_eval'].append(bool(allowed and len(entries) >= 2 and entries[0][4] >= .525))
        positives = [row for row in entries if row[3]]
        truth = entries[0][2]
        previous = owners.get(truth)
        canonical = max(previous, key=lambda b: (previous[b], -b)) if previous else None
        preferred = [row for row in positives if row[1] == canonical]
        targets = preferred or positives
        for edge_id, _, _, _, _ in targets:output['col_target'][edge_id] = True
        for edge_id, _, _, _, _ in entries:output['edge_col'].append((edge_id, group))
        counts['columns'] += 1
        counts['strong_competing_columns'] += output['col_eval'][-1]
        counts['oversize_columns'] += len(entries) > MAX_COLUMN
        counts['canonical_duplicate_columns'] += len(positives) > 1 and bool(preferred)
    assert len(output['edge_col']) == len(output['pair'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    for name, digest in protocol['collection_sources'].items():assert sha(ROOT/name) == digest, name
    assert sha(args.freeze) == protocol['parent_freeze_sha256']
    cache = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['complete'] and cache['data_sha256'] == sha(args.data) == protocol['data_sha256']
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    names = protocol['training24'] + protocol['validation8']
    assert len(set(names)) == 32 and set(names) <= set(plan['fit64'])
    assert not set(names) & set(plan['calibration']+plan['confirmation16']+plan['sealed_excluded'])
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    assert cache['point_weights_sha256'] == components.freeze['artifacts']['point']
    args.out.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    with open_scenes(args.data) as (_, opener):
        for scene in names[args.index::args.shards]:
            with opener(scene) as source:frames = list(_scans(source))
            cache_path = args.cache/'fit'/f'{scene}.npz'
            with np.load(cache_path) as stored:scores, offsets = stored['point_scores'], stored['point_offsets']
            assert len(offsets) == len(frames)+1
            tracker = components.build('candidate')
            tape = Tape(tracker.model_scorer, tracker)
            tracker.model_scorer = tape
            shadow = components.build('candidate') if scene == names[0] else None
            histories, owners = defaultdict(Counter), defaultdict(Counter)
            output = {key: [] for key in ('pair', 'row', 'edge_known', 'edge_positive', 'edge_row', 'edge_col',
                       'row_allowed', 'col_target', 'col_allowed', 'col_eval')}
            counts = Counter()
            for frame, (_, points, labels) in enumerate(frames):
                tape.calls = []
                values = scores[offsets[frame]:offsets[frame+1]].tolist()
                prediction = tracker(points, values)
                if shadow is not None:assert prediction == shadow(points, values), (scene, frame)
                for call in tape.calls:label_call(call, points, labels, histories, owners, output, counts)
                for point in prediction:
                    truth, batch = labels[point['PntID']], point['BatchID']
                    if truth >= 0:histories[batch][truth] += 1
                    if truth > 0:owners[truth][batch] += 1
            assert output['pair'] and output['row']
            columns = np.empty(len(output['pair']), dtype=np.int32)
            for edge, group in output.pop('edge_col'):columns[edge] = group
            arrays = {key: np.asarray(value, dtype=np.float32 if key in ('pair', 'row') else
                np.int32 if key == 'edge_row' else bool) for key, value in output.items()}
            arrays['edge_col'] = columns
            assert np.all(arrays['edge_positive'] <= arrays['edge_known'])
            assert np.all(arrays['col_target'] <= arrays['edge_positive'])
            dest = args.out/f'{scene}.npz'
            np.savez_compressed(dest, **arrays, protocol_sha256=sha(args.protocol))
            save(args.out/f'{scene}.json', {'scene': scene, 'counts': {k: int(v) for k, v in counts.items()}, 'protocol_sha256': sha(args.protocol),
                'sample_sha256': sha(dest), 'point_cache_sha256': sha(cache_path), 'tape_shadow_checked': shadow is not None})
            print(scene, 'rows', counts['rows'], 'edges', len(columns), 'seconds', round(monotonic()-started, 1), flush=True)


if __name__ == '__main__':main()
