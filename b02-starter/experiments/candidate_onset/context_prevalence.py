"""Audit whether correct context writes can cause a later identity error."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import numpy as np
import torch

from b02.replay import _scans, open_scenes, replay_frames
from experiments.candidate_onset.joint_event import clone, trace_keys
from experiments.candidate_onset.mixed import install
from experiments.candidate_onset.propagation import anchor, sha
from experiments.id_recovery.affinity import pack
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.v2_association.run_stage1 import Diagnostic, save


def context_prefix(tracker):
    return {batch: list(history) for batch, history in tracker.context_history.items()
            if tracker.emissions.get(batch, 0) >= 4 and len(history) >= 4}


def writes(tracker, output, points, scores, prefix):
    emitted = {p['BatchID']: p['PntID'] for p in output}
    events = {e['batch']: e for e in tracker.events if e['indices']}
    rows = []
    for batch, index in tracker.last_matches.items():
        row = events.get(batch)
        if (batch not in prefix or emitted.get(batch) != points[index]['PntID'] or scores[index] < .525
                or row is None or index not in row['indices']):
            continue
        edge = row['edge_features'][row['indices'].index(index)]
        previous = np.asarray(prefix[batch][-4:])
        change = float(np.linalg.norm(tracker.current_context[index] - previous.mean(0)))
        rows.append((batch, index, change, row['track_features'], edge))
    if not rows:
        return []
    affinity = tracker.affinity.score(pack(np.stack([row[3] for row in rows]), np.stack([row[4] for row in rows])))
    return [(batch, index, delta, float(score)) for (batch, index, delta, _, _), score in zip(rows, affinity, strict=True)]


def observe(frames, cached, make, plan):
    tracker = make()
    history = defaultdict(lambda: deque(maxlen=8))
    records = defaultdict(list)
    reference, all_scores = [], []
    for index, (_, points, labels) in enumerate(frames):
        prefix = context_prefix(tracker)
        scores = cached(index)
        output = tracker(points, scores)
        reference.append(trace_keys(output))
        updates = {b: (delta, affinity) for b, _, delta, affinity in writes(tracker, output, points, scores, prefix)}
        all_scores.extend(updates.values())
        for item in output:
            batch, label = item['BatchID'], labels[item['PntID']]
            records[batch].append((index, label, anchor(history[batch]), updates.get(batch)))
            if label >= 0:
                history[batch].append(label)
    return reference, records, np.asarray(all_scores, dtype=np.float32).reshape(-1, 2)


def pick_cases(records, length, plan):
    ahead, cap, lag = (plan[key] for key in ('horizon', 'cases_per_scene', 'precursor_scans'))
    errors, controls = [], []
    seen = set()
    for batch, events in records.items():
        for pos, (index, truth, expected, feature) in enumerate(events):
            if expected is None or index + ahead >= length:
                continue
            if truth > 0 and truth != expected:
                key = batch, expected, truth
                if key in seen:
                    continue
                seen.add(key)
                candidates = [(j, f) for j, q, a, f in events[max(0, pos-lag):pos]
                              if 1 <= index-j <= lag and q == expected == a and f is not None]
                if candidates:
                    j, f = max(candidates, key=lambda x: (x[1][0], x[0]))
                    errors.append(dict(batch=batch, start=j, onset=index, expected=expected, assigned=truth,
                                       delta=f[0], affinity=f[1], kind='precursor'))
            elif truth == expected and feature is not None and feature[0] > 0:
                follow = [q for j, q, _, _ in events[pos+1:pos+ahead+1] if j <= index+ahead]
                if len(follow) >= 3 and all(q == expected for q in follow):
                    controls.append(dict(batch=batch, start=index, onset=index, expected=expected, assigned=truth,
                                         delta=feature[0], affinity=feature[1], kind='control'))
    errors.sort(key=lambda r: (r['onset'], r['batch']))
    chosen = [errors[j] for j in np.linspace(0, len(errors)-1, min(cap, len(errors)), dtype=int)] if errors else []
    selected, used = [], set()
    for event in chosen:
        # Match on both decision features, so easy normal updates do not make
        # the counterfactual look safer than it is around crossing targets.
        possible = (r for r in controls if r['batch'] not in used)
        match = min(possible, default=None, key=lambda r: abs(r['delta']-event['delta']) + .10*abs(r['affinity']-event['affinity']))
        if match is not None:
            selected.append(match)
            used.add(match['batch'])
    return chosen + selected, len(errors), len(controls)


def outcome(output, labels, batch, truth):
    selected = [labels[p['PntID']] for p in output if p['BatchID'] == batch]
    return 'correct' if truth in selected else 'other_target' if any(q > 0 for q in selected) else 'silent_or_other'


def counterfactual(frames, cached, make, plan, chosen, reference):
    tracker = make()
    waiting = defaultdict(list)
    for event in chosen:
        waiting[event['start']].append(event)
    for i, (_, points, labels) in enumerate(frames):
        if i > max(waiting, default=-1):
            break
        copies = [(event, clone(tracker)) for event in waiting.get(i, ())]
        output = tracker(points, cached(i))
        assert trace_keys(output) == reference[i], i
        for event, branch in copies:
            batch, start, truth = event['batch'], event['start'], event['expected']
            saved = list(branch.context_history[batch])
            same = branch(points, cached(i))
            assert trace_keys(same) == reference[i] and len(saved) >= 4
            branch.context_history[batch] = saved
            before, after = [], []
            end = event['onset'] + plan['horizon'] + 1
            for j in range(start+1, end):
                _, new_points, new_labels = frames[j]
                altered = branch(new_points, cached(j))
                before.append(outcome([{'BatchID': b, 'PntID': p} for b, p in reference[j]], new_labels, batch, truth))
                after.append(outcome(altered, new_labels, batch, truth))
            event['original'], event['held'] = before, after
            event['wrong_reduction'] = before.count('other_target') - after.count('other_target')
            event['correct_gain'] = after.count('correct') - before.count('correct')
            event['onset_prevented'] = event['kind'] == 'precursor' and after[event['onset']-start-1] != 'other_target'
    return chosen


def full_replay(frames, cached, tracker, policy, trace):
    audit, birth = Diagnostic(), BirthAudit()
    counts = Counter()
    index = 0

    def run(points, _):
        nonlocal index
        circle, _, labels = frames[index]
        scores = cached(index)
        index += 1
        active = set(tracker.state.get('tracks', {}))
        prefix = context_prefix(tracker)
        result = tracker(points, scores)
        for batch, _, delta, affinity in writes(tracker, result, points, scores, prefix):
            counts['eligible'] += 1
            if delta >= policy['delta_min'] and affinity <= policy['affinity_max']:
                tracker.context_history[batch] = prefix[batch]
                counts['withheld'] += 1
        audit.observe(points, labels, result, tracker, active, circle)
        birth.observe(points, labels, result, tracker, circle)
        return result

    summary = replay_frames(frames, trace, run)
    return dict(summary=summary, identity=audit.result()['counts'], operations=dict(counts), birth_audit=birth.result())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--role', choices=('development', 'work'), required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.protocol.read_bytes())
    digest = sha(args.protocol)
    root = Path(__file__).resolve().parents[2]
    assert all(sha(root / name) == h for name, h in plan['sources'].items())
    assert sha(args.data) == plan[args.role]['data_sha256'] and sha(args.freeze) == plan['parent_freeze_sha256']
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    split = json.loads((root / 'splits/v3_tracking.json').read_bytes())
    names = plan[args.role]['scenes'] if args.policy is None else plan['full_scenes']
    assert args.policy is None or args.role == 'work'
    allowed = split['fit64'] if args.role == 'development' else components.freeze['work_validation16']
    assert set(names) <= set(allowed) and not set(names) & set(split['sealed_excluded'])
    scene = names[args.index]
    cache = args.cache / (scene + '.npz')
    manifest = json.loads((args.cache / 'manifest.json').read_bytes())
    assert sha(cache) == manifest['scenes'][scene]['cache_sha256']
    with open_scenes(args.data) as (_, opener), opener(scene) as source:
        frames = list(_scans(source))
    with np.load(cache) as stored:
        scores, offsets = stored['point_scores'], stored['point_offsets']
    assert len(offsets) == len(frames)+1

    def cached(i):
        return scores[offsets[i]:offsets[i+1]].tolist()

    def make():
        return install(components.build('candidate'), 'combined')

    args.out.mkdir(parents=True, exist_ok=True)
    if args.policy is None:
        expected, records, features = observe(frames, cached, make, plan)
        chosen, candidates, normal = pick_cases(records, len(frames), plan)
        rows = counterfactual(frames, cached, make, plan, chosen, expected)
        np.savez_compressed(args.out/(scene+'.npz'), features=features)
        report = dict(precursor_candidates=candidates, normal_candidates=normal, rows=rows, writes=len(features))
        print(scene, candidates, len(rows), sum(r['wrong_reduction'] > 0 and r['correct_gain'] >= 0 for r in rows), flush=True)
    else:
        policy = json.loads(args.policy.read_bytes())
        assert policy['protocol_sha256'] == digest and policy['enabled']
        with (args.out/(scene+'.jsonl')).open('w', encoding='utf8') as trace:
            report = full_replay(frames, cached, make(), policy, trace)
        report['policy_sha256'] = sha(args.policy)
        print(scene, report['identity']['mixed_minority_truth_points'], report['operations'], flush=True)
    save(args.out/(scene+'.json'), dict(report, scene=scene, role=args.role, protocol_sha256=digest))


if __name__ == '__main__':
    main()
