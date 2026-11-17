"""One-event counterfactuals for the joint-cost reassignments."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from b02.replay import _scans, open_scenes, replay_frames
from experiments.candidate_onset import joint_cost_replay as joint
from experiments.candidate_onset.joint_cost_replay import apply_proposals
from experiments.candidate_onset.mixed import install
from experiments.candidate_onset.propagation import bank, measure, sha
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.v2_association.run_stage1 import Diagnostic, save

GROUPS = dict(short=('history', 'context_history'), memory=('memories',),
              support=('quality_hits', 'strong_hits'))
MODES = ('joint', 'veto', 'motion', 'short', 'memory', 'support', 'all_association')


def cut_selected(matches, proposals, cutoff, pair, index):
    accepted = apply_proposals(matches, proposals, cutoff)
    focal = [p for p in accepted if (p.winner, p.rival, p.point) == (*pair, index)]
    assert len(focal) == 1
    p, = focal
    matches[p.winner] = p.point
    if p.other is not None:
        matches[p.rival] = p.other
    else:
        del matches[p.rival]
    assert len(set(matches.values())) == len(matches)
    return [q for q in accepted if q is not p]


def replace_state(tracker, control, pair, mode):
    fields = tuple(f for group in GROUPS.values() for f in group) if mode == 'all_association' else GROUPS.get(mode, ())
    for batch in pair:
        track, state = bank(tracker, batch)
        other, baseline = bank(control, batch)
        assert state == baseline == 'active', (batch, state, baseline)
        if mode in ('motion', 'all_association'):
            for attr in ('position', 'velocity', 'last_scan'):
                setattr(track, attr, getattr(other, attr))
        if mode in ('support', 'all_association'):
            track.hits = other.hits
        for field in fields:
            src, dest = getattr(control, field), getattr(tracker, field)
            if batch in src:
                dest[batch] = deepcopy(src[batch])
            else:
                dest.pop(batch, None)


def clone(tracker):
    # Learned weights and scorers do not change during replay. The column
    # scorer wrapper still needs its own pointer to each cloned tracker.
    shared = (tracker.affinity, tracker.head, tracker.model_scorer.prior)
    return deepcopy(tracker, {id(item): item for item in shared})


def candidates(tracker, batch, points, labels, truth):
    rows = []
    for event in tracker.events:
        if event['batch'] != batch:
            continue
        indices = event['indices']
        occupied = {i: b for b, i in tracker.last_matches.items()}
        relevant = []
        for j, index in enumerate(indices):
            label = labels[points[index]['PntID']]
            if label != truth and index != tracker.last_matches.get(batch):
                continue
            relevant.append(dict(point=points[index]['PntID'], label=label, owner=occupied.get(index),
                                 residual=float(event['edge_features'][j, 9]),
                                 effective_cost=event.get('effective_costs', event['base_costs'])[j],
                                 base_cost=event['base_costs'][j]))
        rows.append(dict(gated_truth=sum(labels[points[i]['PntID']] == truth for i in event['gated_indices']),
                         relevant=relevant, n_candidates=len(indices), dummy=event.get('effective_dummy')))
    return rows


def count_pair(output, labels, truths, tallies):
    for item in output:
        batch = item['BatchID']
        if batch not in truths:
            continue
        truth = labels[item['PntID']]
        if truth > 0:
            tallies[batch][truth] += 1


def trace_keys(output):
    return tuple((p['BatchID'], p['PntID']) for p in output)


def before_scan(tracker, batch):
    track, bank_name = bank(tracker, batch)
    return dict(bank=bank_name, position=track.position if track else None,
                velocity=track.velocity if track else None, last_scan=track.last_scan if track else None)


def replay(tracker, frames, scores, offsets, mode, fixture, expected, out):
    circle_at, pair, focal_point = fixture['circle'], tuple(fixture['pair']), fixture['point']
    truths = dict(zip(pair, fixture['truths'], strict=True))
    i = 0
    intervention_count = 0
    diagnostic, births = Diagnostic(), BirthAudit()
    tallies = {batch: Counter() for batch in pair}
    post_tallies = {batch: Counter() for batch in pair}
    window = []

    def handler(points, _):
        nonlocal i, intervention_count
        circle, _, labels = frames[i]
        scan_scores = scores[offsets[i]:offsets[i + 1]].tolist()
        active = set(tracker.state.get('tracks', {}))
        before = {b: before_scan(tracker, b) for b in pair} if circle_at <= circle <= circle_at + 30 else {}
        if circle == circle_at and mode != 'joint':
            index = next(j for j, p in enumerate(points) if p['PntID'] == focal_point)
            control = tracker if mode == 'veto' else clone(tracker)
            with patch.object(joint, 'apply_proposals', lambda m, p, c: cut_selected(m, p, c, pair, index)):
                result = control(points, scan_scores)
            intervention_count += 1
            if mode == 'veto':
                output = result
                assert (pair[0], focal_point) in trace_keys(output)
            else:
                output = tracker(points, scan_scores)
                assert (pair[1], focal_point) in trace_keys(output)
                replace_state(tracker, control, pair, mode)
        else:
            output = tracker(points, scan_scores)
        if mode == 'joint' or circle < circle_at or (circle == circle_at and mode != 'veto'):
            assert trace_keys(output) == expected[i], (mode, circle)
        i += 1
        count_pair(output, labels, truths, tallies)
        if circle > circle_at:
            count_pair(output, labels, truths, post_tallies)
        if before:
            window.append(dict(circle=circle, old=before, tracks={b: dict(
                **measure(tracker, output, points, labels, b, truths[b]),
                matched_point=points[tracker.last_matches[b]]['PntID'] if b in tracker.last_matches else None,
                trusted_update=b in tracker.trusted_updates, candidates=candidates(tracker,b,points,labels,truths[b])) for b in pair}))
        births.observe(points, labels, output, tracker, circle)
        diagnostic.observe(points, labels, output, tracker, active, circle)
        return output

    with (out / (mode + '.jsonl')).open('w', encoding='utf8') as stream:
        summary = replay_frames(frames, stream, handler)
    assert intervention_count == (mode != 'joint')
    mixed = sum(counts.total() - max(counts.values(), default=0) for counts in tallies.values())
    return dict(mode=mode, summary=summary, identity=diagnostic.result()['counts'], birth_audit=births.result(),
                mixed_pair=mixed, pair_truth_counts=tallies, pair_post_onset_truth_counts=post_tallies, window=window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'calibration', 'reference', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.protocol.read_bytes())
    root = Path(__file__).resolve().parents[2]
    assert all(sha(root / name) == digest for name, digest in plan['sources'].items())
    assert sha(args.data) == plan['data_sha256'] and sha(args.freeze) == plan['parent_freeze_sha256']
    assert sha(args.calibration) == plan['calibration_sha256']
    assert sha(args.reference) == plan['reference_sha256']
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    split = json.loads((root / 'splits/v3_tracking.json').read_bytes())
    scene = plan['fixture']['scene']
    assert scene in components.freeze['work_validation16'] and scene not in split['sealed_excluded']
    manifest = json.loads((args.cache / 'manifest.json').read_bytes())
    path = args.cache / (scene + '.npz')
    assert sha(path) == manifest['scenes'][scene]['cache_sha256']
    with open_scenes(args.data) as (_, opener), opener(scene) as source:
        frames = list(_scans(source))
    with np.load(path) as cached:
        scores, offsets = cached['point_scores'], cached['point_offsets']
    expected = [trace_keys(json.loads(line)['tracks']) for line in args.reference.read_text(encoding='utf8').splitlines()]
    assert len(expected) == len(frames) == len(offsets)-1
    tracker = install(components.build('candidate'), 'combined')
    tracker.__class__ = joint.JointCostTracker
    tracker.joint_cutoff = json.loads(args.calibration.read_bytes())['cutoff']
    args.out.mkdir(parents=True, exist_ok=True)
    report = replay(tracker, frames, scores, offsets, args.mode, plan['fixture'], expected, args.out)
    report['protocol_sha256'] = sha(args.protocol)
    if args.mode == 'joint':
        assert report['mixed_pair'] == plan['fixture']['reproduced_pair_mixed']
    save(args.out / (args.mode + '.json'), report)
    print(args.mode, report['mixed_pair'], report['identity']['mixed_minority_truth_points'], flush=True)


if __name__ == '__main__':
    main()
