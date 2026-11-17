"""Read-only candidate and rejection audit; no labels enter the tracker."""
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
from experiments.robust_validation.frozen import Components
from experiments.v2_association.run_stage1 import save, sha


def prefix_anchor(counts):
    """A diagnostic identity based only on previous known output labels."""
    positive = {key: value for key, value in counts.items() if key > 0}
    if not positive:
        return None
    best = max(positive, key=lambda key: (positive[key], -key))
    if positive[best] < 3 or positive[best] < .8 * sum(counts.values()):
        return None
    return best


class CompetitionAudit:
    def __init__(self):
        self.by_id = defaultdict(Counter)
        self.by_truth = defaultdict(Counter)
        self.counts = Counter()

    def observe(self, points, labels, outputs, tracker, bank_before, scores):
        # All state here is external and label-dependent. Never write tracker state.
        anchors = {batch: anchor for batch, counts in self.by_id.items()
                   if (anchor := prefix_anchor(counts)) is not None}
        c = self.counts
        edges = defaultdict(set)
        gated = defaultdict(set)
        by_index = defaultdict(set)
        for event in tracker.events:
            batch = event['batch']
            edges[batch].update(event['indices'])
            gated[batch].update(event['gated_indices'])
        for batch, index in tracker.retired_edges:
            edges[batch].add(index)
        for row in tracker.proposals:
            edges[row['old']].add(row['index'])
        for batch, indices in edges.items():
            for index in indices:
                by_index[index].add(batch)
        point_index = {point['PntID']: index for index, point in enumerate(points)}
        assigned = {point['PntID']: point['BatchID'] for point in outputs}
        published_now = {point['BatchID']: labels[point['PntID']] for point in outputs}
        present = {value for value in labels.values() if value > 0}
        for batch in set(edges) | set(tracker.last_matches):
            anchor = anchors.get(batch)
            if anchor is None:
                continue
            c['anchored_recorded_rows'] += 1
            right = {i for i, point in enumerate(points) if labels[point['PntID']] == anchor}
            has_right = bool(edges[batch] & right)
            c['anchor_present_rows'] += anchor in present
            c['anchor_candidate_rows'] += has_right
            selected = tracker.last_matches.get(batch)
            if selected is None:
                c['null_with_correct_candidate' if has_right else 'null_without_correct_candidate'] += 1
                continue
            truth = labels[points[selected]['PntID']]
            status = ('correct' if truth == anchor else 'clutter' if truth == 0 else
                      'unknown' if truth == -1 else 'other_target')
            c['row_selected_' + status] += 1
            c[('with_correct_candidate_' if has_right else 'without_correct_candidate_') + status] += 1
            bucket = ('enter' if scores[selected] >= tracker.gate.policy.enter else
                      'maintain' if scores[selected] >= tracker.gate.policy.maintain else 'below_maintain')
            c['row_selected_' + status + '_' + bucket] += 1
            if batch in tracker.trusted_updates:
                c['trusted_update_' + status] += 1
        for index, point in enumerate(points):
            truth = labels[point['PntID']]
            if truth <= 0:
                continue
            c['all_truth_points'] += 1
            competitors = {batch for batch in by_index[index] if batch in self.by_id}
            c['truth_points_multiple_published_candidate_ids'] += len(competitors) > 1
            previous = self.by_truth.get(truth)
            if not previous:
                continue
            canonical = max(previous, key=lambda batch: (previous[batch], -batch))
            if previous[canonical] < 3:
                continue
            c['prefix_canonical_points'] += 1
            chosen = assigned.get(point['PntID'])
            if chosen == canonical:
                c['prefix_canonical_hits'] += 1
                continue
            c['prefix_canonical_misses'] += 1
            if index in edges[canonical]:
                reason = 'recorded_edge'
            elif index in gated[canonical]:
                reason = 'geometry_only'
            elif canonical in bank_before:
                reason = 'in_bank_no_recorded_edge'
            else:
                reason = 'outside_banks'
            c['prefix_miss_' + reason] += 1
            c['prefix_miss_' + ('other_output_id' if chosen is not None else 'not_output')] += 1
            c['prefix_miss_below_maintain'] += scores[index] < tracker.gate.policy.maintain
            other = published_now.get(canonical)
            c['prefix_miss_canonical_outputs_other_target'] += other is not None and other > 0 and other != truth
            internal = tracker.last_matches.get(canonical)
            if internal is not None:
                internal_truth = labels[points[internal]['PntID']]
                c['prefix_miss_canonical_internal_clutter'] += internal_truth == 0
                c['prefix_miss_canonical_internal_other_target'] += internal_truth > 0 and internal_truth != truth
        # Update identity references last, so the current label cannot define its own anchor.
        for point in outputs:
            truth = labels[point['PntID']]
            batch = point['BatchID']
            if truth >= 0:
                self.by_id[batch][truth] += 1
            if truth > 0:
                self.by_truth[truth][batch] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    assert protocol['source_sha256'] == sha(Path(__file__))
    assert protocol['frozen_protocol_sha256'] == sha(args.freeze)
    plan = json.loads((ROOT / 'splits/v3_tracking.json').read_text(encoding='utf-8'))
    names = protocol['fit_preflight8']
    assert len(names) == 8 and set(names) <= set(plan['fit64'])
    assert not set(names) & set(plan['calibration'] + plan['confirmation16'] + plan['sealed_excluded'])
    cache = json.loads((args.cache / 'manifest.json').read_text(encoding='utf-8'))
    assert cache['data_sha256'] == sha(args.data) == protocol['data_sha256']
    assert set(names) <= set(cache['fit'])
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    assert cache['point_weights_sha256'] == components.freeze['artifacts']['point']
    assert 0 <= args.index < args.shards
    args.out.mkdir(parents=True, exist_ok=False)
    start = monotonic()
    with open_scenes(args.data) as (_, opener):
        for scene in names[args.index::args.shards]:
            with opener(scene) as source:
                frames = list(_scans(source))
            cache_path = args.cache / 'fit' / f'{scene}.npz'
            with np.load(cache_path) as payload:
                scores, offsets = payload['point_scores'], payload['point_offsets']
            assert len(offsets) == len(frames) + 1
            tracker = components.build('candidate', probe=True)
            audit = CompetitionAudit()
            outputs = 0
            for frame, (_, points, labels) in enumerate(frames):
                banks = set(tracker.state.get('tracks', {})) | set(tracker.retired) | set(tracker.short_retired)
                values = scores[offsets[frame]:offsets[frame+1]].tolist()
                prediction = tracker(points, values)
                audit.observe(points, labels, prediction, tracker, banks, values)
                outputs += len(prediction)
            c = audit.counts
            assert c['prefix_canonical_hits'] + c['prefix_canonical_misses'] == c['prefix_canonical_points']
            assert sum(c['prefix_miss_' + name] for name in ('recorded_edge', 'geometry_only',
                'in_bank_no_recorded_edge', 'outside_banks')) == c['prefix_canonical_misses']
            save(args.out / f'{scene}.json', {'scene': scene, 'counts': dict(c), 'outputs': outputs,
                'circles': len(frames), 'protocol_sha256': sha(args.protocol),
                'cache_sha256': sha(cache_path), 'fitting_pool_only': True, 'no_training': True})
            print(scene, 'preflight complete', round(monotonic()-start, 1), flush=True)


if __name__ == '__main__':
    main()
