"""Eight-file calibration only; labels are restricted to external diagnostics."""
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
from b02.replay import _scans, open_scenes, replay_frames
from experiments.id_recovery.affinity import load_affinity
from experiments.id_recovery.replay import classify_recovery
from experiments.prebirth.tracker import CONFIGS, PrebirthTracker
from experiments.v2_association.run_stage1 import Diagnostic, Discard, save, sha
from experiments.v3_tracking.diagnose import reference_equal
from experiments.joint_matching.diagnose import FragmentationAudit


class BirthAudit:
    def __init__(self):
        self.truth_ids = defaultdict(Counter)
        self.first_truth = {}
        self.first_output = {}
        self.id_first = {}
        self.counts = Counter()

    def observe(self, points, labels, outputs, tracker, scan):
        for truth in set(labels.values()):
            if truth > 0:
                self.first_truth.setdefault(truth, scan)
        for p in outputs:
            truth, batch = labels[p['PntID']], p['BatchID']
            if truth > 0:
                self.first_output.setdefault(truth, scan)
                self.id_first.setdefault((truth, batch), scan)
                self.truth_ids[truth][batch] += 1
        for proposal in tracker.proposals:
            truth = labels[points[proposal['index']]['PntID']]
            if proposal['selected']:
                self.counts['prebirth_first_truth_output'] += truth > 0 and self.first_output.get(truth) == scan

    def result(self):
        histogram = Counter()
        for truth, counts in self.truth_ids.items():
            primary = max(counts, key=counts.get)
            for batch, count in counts.items():
                if batch != primary:
                    histogram[str(count) if count <= 3 else '4+'] += 1
        delays = {str(t): self.first_output[t]-when for t, when in self.first_truth.items() if t in self.first_output}
        return {'extra_id_lengths': dict(histogram), 'extra_ids_1or2': histogram['1']+histogram['2'],
                'first_truth_delays': delays, **dict(self.counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'reference', 'out', 'affinity'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--method', choices=sorted(CONFIGS), required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--scene', nargs='+')
    parser.add_argument('--trace-scene', default='train_032')
    args = parser.parse_args()
    torch.set_num_threads(2)
    from experiments.v3_tracking.runtime import load_scorer
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    cache = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['data_sha256'] == sha(args.data) and cache['calibration'] == plan['calibration']
    assert not set(cache['calibration']) & set(plan['sealed_excluded']+plan['confirmation16'])
    names = args.scene or plan['calibration']
    assert len(set(names)) == len(names) and set(names) <= set(plan['calibration'])
    assert 0 <= args.index < args.shards
    reference = json.loads(args.reference.read_text(encoding='utf-8'))['scenes']
    scorer, memory, _, _ = load_scorer(args.prior)
    assert memory
    affinity = load_affinity(args.affinity, args.prior)
    args.out.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    with open_scenes(args.data) as (_, opener):
        for scene in names[args.index::args.shards]:
            with opener(scene) as source:
                frames = list(_scans(source))
            with np.load(args.cache/'calibration'/f'{scene}.npz') as data:
                scores, offsets = data['point_scores'], data['point_offsets']
            assert len(offsets) == len(frames)+1
            tracker = PrebirthTracker(scorer, affinity, CONFIGS[args.method])
            diagnostic, audit, decisions = Diagnostic(), BirthAudit(), Counter()
            fragments = FragmentationAudit() if args.audit else None
            published, saved_outputs = set(), []
            index = 0

            def run(points, _state):
                nonlocal index
                circle, _, labels = frames[index]
                before = set(tracker.state.get('tracks', {}))
                snapshot = fragments.snapshot(tracker) if fragments else None
                retired_before = set(tracker.retired)
                prior_keys = set(fragments.first) if fragments else None
                predictions = tracker(points, scores[offsets[index]:offsets[index+1]].tolist())
                index += 1
                classify_recovery(tracker, points, labels, diagnostic.owners, predictions, decisions)
                if fragments:
                    fragments.observe(circle, points, labels, predictions, tracker, snapshot)
                    for key in set(fragments.first)-prior_keys:
                        row = fragments.first[key]
                        row['first_published_id'] = key[1] not in published
                        row['old_retired'] = row['old'] in retired_before
                    saved_outputs.append(predictions)
                    for proposal in tracker.proposals:
                        truth = labels[points[proposal['index']]['PntID']]
                        owners = diagnostic.owners.get(proposal['old'], {})
                        old_truth = max(owners, key=owners.get) if owners else None
                        same = truth > 0 and truth == old_truth
                        decisions['proposal_same_truth' if same else 'proposal_other_truth'] += 1
                        decisions['eligible_same_truth' if same else 'eligible_other_truth'] += proposal['eligible']
                    published.update(p['BatchID'] for p in predictions)
                audit.observe(points, labels, predictions, tracker, circle)
                diagnostic.observe(points, labels, predictions, tracker, before, circle)
                return predictions

            stream = (args.out/f'{scene}.jsonl').open('w', encoding='utf-8') if scene == args.trace_scene else Discard()
            try:
                summary = replay_frames(frames, stream, run)
            finally:
                if hasattr(stream, 'close'):
                    stream.close()
            if args.method in ('control', 'audit_only'):
                assert reference_equal(summary, reference[scene]), scene
            result = {'scene': scene, 'variants': {args.method: summary},
                 'identity': diagnostic.result()['counts'], 'plans': dict(decisions+tracker.counters), 'birth_audit': audit.result()}
            if fragments:
                breakdown = fragments.result()
                selected = [f for f in breakdown['fragments'] if f['points'] <= 2]
                breakdown['short2_first_events'] = dict(Counter((
                    ('fresh_' if f['first_published_id'] else 'already_published_') +
                    ('retired' if f['old_retired'] else f['cause']) + '_' + str(f['old_match'])) for f in selected))
                ceilings = {}
                for name, selected in (('all_short2', selected), ('fresh_short2', [f for f in selected if f['first_published_id']]),
                                       ('fresh_all_lengths', [f for f in breakdown['fragments'] if f['first_published_id']])):
                    repaired, operations = fragments.repair(saved_outputs, frames, selected)
                    output = iter(repaired)
                    corrected = replay_frames(frames, Discard(), lambda _p, _s: next(output))
                    ceilings[name] = {'summary': corrected, 'operations': operations}
                result['fragment_audit'] = breakdown
                result['diagnostic_oracles'] = ceilings
            save(args.out/f'{scene}.json', result)
            print(scene, args.method, dict(decisions), round(monotonic()-started, 1), flush=True)


if __name__ == '__main__':
    main()
