"""Hold a doubtful state update when a supported alternative is visible now."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import numpy as np
import torch

from b02.replay import _scans, open_scenes, replay_frames
from experiments.candidate_onset.mix_attribution import owners, split_episodes
from experiments.candidate_onset.mixed import install
from experiments.candidate_onset.propagation import anchor, restore, sha
from experiments.candidate_onset.propagation_work import capture_prefix
from experiments.id_recovery.affinity import pack
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.v2_association.run_stage1 import Diagnostic, Discard, save

FEATURES = ('assigned_affinity', 'alternative_affinity', 'assigned_residual', 'alternative_residual',
            'velocity_advantage', 'signal_advantage', 'learned_advantage', 'detector_advantage',
            'rival_recipient', 'rival_affinity_advantage', 'alternative_count')


def eligible_ids(tracker):
    last = tracker.state.get('scan', 0)
    return {batch for batch, track in tracker.state.get('tracks', {}).items()
            if track.last_scan == last and tracker.emissions.get(batch, 0) >= 4
            and len(tracker.history.get(batch, ())) >= 4}


def proposals(tracker, points, scores, output, eligible):
    matched = tracker.last_matches
    emitted = {p['BatchID']: p['PntID'] for p in output}
    owner = {i: b for b, i in matched.items()}
    recorded = {}
    for event in tracker.events:
        recorded.setdefault(event['batch'], event)
    options = {}
    tx, ex = [], []
    for batch in sorted(eligible):
        index, row = matched.get(batch), recorded.get(batch)
        if (index is None or emitted.get(batch) != points[index]['PntID'] or scores[index] < .525
                or row is None or index not in row['indices'] or len(row['indices']) < 2):
            continue
        edges = {j: k for k, j in enumerate(row['indices']) if scores[j] >= .525}
        if len(edges) < 2:
            continue
        book = {}
        for point, ordinal in edges.items():
            book[point] = len(tx)
            tx.append(row['track_features'])
            ex.append(row['edge_features'][ordinal])
        options[batch] = (index, row, edges, book)
    if not options:
        return []
    logits = tracker.affinity.score(pack(np.stack(tx), np.stack(ex)))
    rows = []
    for batch, (index, row, edges, book) in options.items():
        # Require current input support for a distinct, plausible observation.
        plausible = [q for q in edges if q != index and logits[book[q]] >= 0
                     and ex[book[q]][9] <= .64]
        if not plausible:
            continue
        alt = max(plausible, key=lambda q: (logits[book[q]]-ex[book[q]][9], -q))
        rival = options.get(owner.get(alt))
        reciprocal = rival is not None and index in rival[3] and alt in rival[3]
        rival_advantage = float(logits[rival[3][index]]-logits[rival[3][alt]]) if reciprocal else 0.
        a, b = ex[book[index]], ex[book[alt]]
        costs = row['effective_costs']
        x = [logits[book[index]], logits[book[alt]], a[9], b[9],
             np.linalg.norm(a[6:9])-np.linalg.norm(b[6:9]), np.linalg.norm(a[16:18])-np.linalg.norm(b[16:18]),
             costs[edges[index]]-costs[edges[alt]], scores[alt]-scores[index],
             float(reciprocal), rival_advantage, np.log1p(len(plausible))]
        rows.append((batch, index, alt, tuple(plausible), [float(value) for value in x]))
    return rows


def label_update(row, points, labels, history):
    batch, index, alternative, plausible, _ = row
    expected = anchor(history.get(batch, ()))
    truth = labels[points[index]['PntID']]
    if expected is None or truth < 0:
        return 'unresolved'
    if truth == expected:
        return 'correct'
    if truth == 0:
        return 'clutter'
    if labels[points[alternative]['PntID']] == expected:
        return 'visible_supported'
    if any(labels[points[j]['PntID']] == expected for j in plausible):
        return 'visible_other_supported'
    return 'visible_unsupported' if expected in labels.values() else 'original_absent'


def score(model, matrix):
    coef = np.asarray(model['coefficients'])
    return ((matrix-np.asarray(model['mean']))/np.asarray(model['scale'])) @ coef[:-1] + coef[-1]


def run(frames, cached, tracker, mode, model, stream):
    counters, histories = Counter(), defaultdict(lambda: deque(maxlen=8))
    audit, birth = Diagnostic(), BirthAudit()
    audit_rows, track_rows, actions = [], [], []
    used = set()
    i = 0

    def handler(points, _):
        nonlocal i
        circle, _, labels = frames[i]
        scores = cached(i)
        i += 1
        active = set(tracker.state.get('tracks', {}))
        eligible = eligible_ids(tracker)
        before = capture_prefix(tracker, eligible) if mode != 'census' else None
        output = tracker(points, scores)
        candidates = proposals(tracker, points, scores, output, eligible)
        kinds = [label_update(row, points, labels, histories) for row in candidates]
        for kind in kinds:
            counters['eligible_'+kind] += 1
        if mode == 'census':
            audit_rows.extend(dict(circle=circle, batch=row[0], label=kind, x=row[-1])
                              for row,kind in zip(candidates,kinds,strict=True))
        elif candidates:
            if mode == 'automatic':
                flags = score(model, np.asarray([row[-1] for row in candidates])) > model['cutoff']
            else:
                flags = [kind in ('visible_supported','visible_other_supported') for kind in kinds]
            for row, kind, flagged in zip(candidates, kinds, flags, strict=True):
                if not flagged:
                    continue
                batch = row[0]
                assert batch in eligible
                if mode == 'oracle':
                    key = (batch, anchor(histories[batch]), labels[points[row[1]]['PntID']])
                    if key in used:
                        continue
                    used.add(key)
                if restore(tracker,before,batch,full=True):
                    counters['hold_'+kind] += 1
                    actions.append(dict(circle=circle,batch=batch,kind=kind))
        for point in output:
            label = labels[point['PntID']]
            batch = point['BatchID']
            if label >= 0:
                histories[batch].append(label)
            if label > 0:
                track_rows.append((i-1,batch,label))
        audit.observe(points,labels,output,tracker,active,circle)
        birth.observe(points,labels,output,tracker,circle)
        return output

    summary = replay_frames(frames,stream,handler)
    dominant = owners(summary)
    episodes = split_episodes([dict(frame=j,batch=b,truth=q,expected=dominant[b][0],reciprocal=False)
                               for j,b,q in track_rows if b in dominant], 3)
    identity = audit.result()['counts']
    assert sum(r['count'] for r in episodes) == identity['mixed_minority_truth_points']
    return dict(summary=summary, identity=identity, birth_audit=birth.result(), operations=dict(counters),
                continued=sum(r['count']-1 for r in episodes), actions=actions, rows=audit_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data','cache','prior','affinity','head','fusion','freeze','protocol','out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--policy', type=Path)
    p.add_argument('--role', choices=('development','work'), required=True)
    p.add_argument('--index', type=int, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    plan, digest = json.loads(args.protocol.read_bytes()), sha(args.protocol)
    root = Path(__file__).resolve().parents[2]
    assert all(sha(root/name)==h for name,h in plan['sources'].items())
    assert sha(args.data)==plan[args.role]['data_sha256'] and sha(args.freeze)==plan['parent_freeze_sha256']
    components = Components(args.prior,args.affinity,args.head,args.fusion,args.freeze)
    split = json.loads((root/'splits/v3_tracking.json').read_bytes())
    names = plan[args.role]['scenes']
    allowed = split['fit64'] if args.role == 'development' else components.freeze['work_validation16']
    assert set(names)<=set(allowed) and not set(names)&set(split['sealed_excluded'])
    scene = names[args.index]
    cache_path = args.cache/(scene+'.npz')
    manifest = json.loads((args.cache/'manifest.json').read_bytes())
    assert sha(cache_path)==manifest['scenes'][scene]['cache_sha256']
    with open_scenes(args.data) as (_, opener), opener(scene) as source:
        frames = list(_scans(source))
    with np.load(cache_path) as cache:
        scores, offsets = cache['point_scores'], cache['point_offsets']
    assert len(offsets)==len(frames)+1

    def cached(i):
        return scores[offsets[i]:offsets[i+1]].tolist()

    args.out.mkdir(parents=True,exist_ok=True)
    model = json.loads(args.policy.read_bytes()) if args.policy else None
    if model is not None:
        assert args.role=='work' and model['protocol_sha256']==digest and model['features']==list(FEATURES)
    modes = ('census',) if model is None else ('automatic','oracle')
    variants = {}
    for mode in modes:
        tracker = install(components.build('candidate'),'combined')
        if mode=='automatic':
            with (args.out/(scene+'.jsonl')).open('w',encoding='utf8') as trace:
                result = run(frames,cached,tracker,mode,model,trace)
        else:
            result = run(frames,cached,tracker,mode,model,Discard())
        variants[mode] = result
        print(scene,mode,result['identity']['mixed_minority_truth_points'],result['operations'],flush=True)
    save(args.out/(scene+'.json'),dict(scene=scene,protocol_sha256=digest,policy_sha256=sha(args.policy) if args.policy else None,
                                     variants=variants))


if __name__=='__main__':
    main()
