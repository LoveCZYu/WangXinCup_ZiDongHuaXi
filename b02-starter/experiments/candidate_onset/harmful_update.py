"""Learn an update hold from paired real-scene continuation outcomes."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import numpy as np
import torch

from b02.replay import _scans, open_scenes, replay_frames
from experiments.candidate_onset.joint_event import clone, trace_keys
from experiments.candidate_onset.mix_attribution import owners, split_episodes
from experiments.candidate_onset.mixed import install
from experiments.candidate_onset.propagation import restore, sha
from experiments.candidate_onset.propagation_work import capture_prefix
from experiments.candidate_onset.visible_onset_hold import FEATURES as PAIR_FEATURES, eligible_ids, proposals, score
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.v2_association.features import normalized
from experiments.v2_association.run_stage1 import Diagnostic, save

FEATURES = (*PAIR_FEATURES, 'picked_robust_residual', 'alternative_robust_residual', 'prefix_fit_residual',
            'last_prefix_residual', 'picked_signal_deviation', 'alternative_signal_deviation', 'prefix_age')


def dominant_prefix(labels):
    if len(labels) < 6:
        return None
    (truth, count), = Counter(labels).most_common(1)
    return truth if truth > 0 and count >= .75*len(labels) else None


def reference_features(history, point, alternative, scan):
    xyz = np.stack([normalized(p) for p, _, _ in history])
    times = np.asarray([t-scan for _, _, t in history],dtype=float)
    step = np.diff(times)
    valid = step > 0
    speed = np.median(np.diff(xyz,axis=0)[valid]/step[valid,None],axis=0) if valid.any() else np.zeros(3)
    position = np.median(xyz-times[:,None]*speed,axis=0)
    residual = np.linalg.norm(xyz-position-times[:,None]*speed,axis=1)
    signal = np.median([[p['Amp'],p['SNR']] for p, _, _ in history],axis=0)
    return [float(v) for v in (
        np.linalg.norm(normalized(point)-position), np.linalg.norm(normalized(alternative)-position),
        np.median(residual), residual[-1], np.linalg.norm(np.asarray([point['Amp'],point['SNR']])-signal),
        np.linalg.norm(np.asarray([alternative['Amp'],alternative['SNR']])-signal), -times[0])]


def candidates(tracker, points, scores, output, past):
    return [(b,i,alt,plausible,[*x,*reference_features(past[b][-8:],points[i],points[alt],tracker.state['scan'])])
            for b,i,alt,plausible,x in proposals(tracker,points,scores,output,past)]


def classify(row, points, labels, past):
    b,i,_,plausible,_=row
    expected=dominant_prefix(past.get(b,()))
    current=labels[points[i]['PntID']]
    if expected is None or current < 0:
        return expected,current,'unresolved'
    if current==expected:
        return expected,current,'correct'
    if current==0:
        return expected,current,'clutter'
    if any(labels[points[j]['PntID']]==expected for j in plausible):
        return expected,current,'visible_wrong'
    return expected,current,'unsupported' if expected in labels.values() else 'absent'


def observe(frames, cached, make):
    tracker=make()
    history=defaultdict(lambda:deque(maxlen=8))
    reference, rows = [], []
    for i,(circle,points,labels) in enumerate(frames):
        scores=cached(i)
        past={b:list(tracker.history[b]) for b in eligible_ids(tracker)}
        output=tracker(points,scores)
        reference.append(trace_keys(output))
        for row in candidates(tracker,points,scores,output,past):
            b,index,*_=row
            expected,current,kind=classify(row,points,labels,history)
            rows.append(dict(frame=i,circle=circle,batch=b,expected=expected,current=current,kind=kind,x=row[-1],
                             point=points[index]['PntID']))
        for b,point in reference[-1]:
            label=labels[point]
            if label>=0:
                history[b].append(label)
    return reference, rows


def continuation(keys, labels, batch, truth):
    got = next((labels[p] for b,p in keys if b==batch),None)
    return 'correct' if got==truth else 'other_target' if got is not None and got>0 else 'other'


def window(frames, reference, row, horizon):
    return [continuation(reference[i],frames[i][2],row['batch'],row['expected'])
            for i in range(row['frame']+1,row['frame']+horizon+1)]


def select_cases(frames, reference, rows, plan):
    horizon, cap = plan['horizon'], plan['cases_per_kind']
    seeds = [row for row in rows if row['kind']=='visible_wrong' and row['frame']+horizon<len(frames)]
    ranked=sorted(seeds,key=lambda row:(-window(frames,reference,row,horizon).count('other_target'),row['frame']))
    selected, seen = [],set()
    for row in ranked:
        key=(row['batch'],row['expected'],row['current'])
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected)>=cap:
            break
    controls=[row for row in rows if row['kind']=='correct' and row['frame']+horizon<len(frames)]
    if controls:
        matrix=np.asarray([row['x'] for row in controls])
        scale=np.maximum(matrix.std(0),.05)
        used=set()
        for seed in list(selected):
            distances=np.square((matrix-seed['x'])/scale).sum(1)
            candidates_sorted=np.argsort(distances,kind='stable')
            found=next((controls[i] for i in candidates_sorted if controls[i]['batch'] not in used),None)
            if found is not None:
                selected.append(found)
                used.add(found['batch'])
    return selected


def pair_replay(frames,cached,make,plan,chosen,reference):
    tracker=make()
    by_frame=defaultdict(list)
    for row in chosen:
        by_frame[row['frame']].append(row)
    horizon=plan['horizon']
    for i in range(max(by_frame,default=-1)+1):
        _,points,_=frames[i]
        for row in by_frame[i]:
            branch=clone(tracker)
            batch=row['batch']
            old=capture_prefix(branch,(batch,))
            assert trace_keys(branch(points,cached(i)))==reference[i] and restore(branch,old,batch,full=True)
            old_tail=window(frames,reference,row,horizon)
            new_tail=[]
            for j in range(i+1,i+horizon+1):
                _,next_points,labels=frames[j]
                keys=trace_keys(branch(next_points,cached(j)))
                new_tail.append(continuation(keys,labels,batch,row['expected']))
            fewer=old_tail.count('other_target')-new_tail.count('other_target')
            gain=new_tail.count('correct')-old_tail.count('correct')
            row.update(wrong_reduction=fewer,correct_gain=gain,beneficial=fewer>=2 and gain>=0)
        assert trace_keys(tracker(points,cached(i)))==reference[i],i
    return chosen


def full(frames,cached,tracker,model,trace):
    histories=defaultdict(lambda:deque(maxlen=8))
    counter=Counter()
    rows, actions = [], []
    diag,birth=Diagnostic(),BirthAudit()
    i=0

    def run(points,_):
        nonlocal i
        circle,_,labels=frames[i]
        scores=cached(i)
        i+=1
        active=set(tracker.state.get('tracks',{}))
        past={b:list(tracker.history[b]) for b in eligible_ids(tracker)}
        saved=capture_prefix(tracker,past)
        output=tracker(points,scores)
        options=candidates(tracker,points,scores,output,past)
        if options:
            values=score(model,np.asarray([row[-1] for row in options]))
            for row, value in zip(options,values,strict=True):
                _,_,kind=classify(row,points,labels,histories)
                counter['eligible_'+kind]+=1
                if value>model['cutoff'] and restore(tracker,saved,row[0],full=True):
                    counter['hold_'+kind]+=1
                    actions.append(dict(circle=circle,batch=row[0],kind=kind))
        for b,p in trace_keys(output):
            label=labels[p]
            if label>=0:
                histories[b].append(label)
            if label>0:
                rows.append((i-1,b,label))
        diag.observe(points,labels,output,tracker,active,circle)
        birth.observe(points,labels,output,tracker,circle)
        return output

    summary=replay_frames(frames,trace,run)
    identity=diag.result()['counts']
    modal=owners(summary)
    runs=split_episodes([dict(frame=j,batch=b,truth=q,expected=modal[b][0],reciprocal=False)
                        for j,b,q in rows if b in modal],3)
    assert sum(r['count'] for r in runs)==identity['mixed_minority_truth_points']
    return dict(summary=summary,identity=identity,birth_audit=birth.result(),operations=dict(counter),actions=actions,
                continued=sum(r['count']-1 for r in runs))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','cache','prior','affinity','head','fusion','freeze','protocol','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--role',choices=('fit','validation','work'),required=True)
    p.add_argument('--index',type=int,required=True)
    p.add_argument('--policy',type=Path)
    args=p.parse_args()
    torch.set_num_threads(1)
    plan=json.loads(args.protocol.read_bytes())
    root=Path(__file__).resolve().parents[2]
    assert all(sha(root/name)==h for name,h in plan['sources'].items())
    assert sha(args.data)==plan[args.role]['data_sha256'] and sha(args.freeze)==plan['freeze_sha256']
    components=Components(args.prior,args.affinity,args.head,args.fusion,args.freeze)
    split=json.loads((root/'splits/v3_tracking.json').read_bytes())
    names=plan[args.role]['scenes']
    allowed=components.freeze['work_validation16'] if args.role=='work' else split['fit64']
    assert set(names)<=set(allowed) and not set(names)&set(split['sealed_excluded'])
    scene=names[args.index]
    manifest=json.loads((args.cache/'manifest.json').read_bytes())
    path=args.cache/((scene)+'.npz')
    assert sha(path)==manifest['scenes'][scene]['cache_sha256']
    with open_scenes(args.data) as (_,opener),opener(scene) as source:
        frames=list(_scans(source))
    with np.load(path) as cache:
        values,offsets=cache['point_scores'],cache['point_offsets']
    assert len(offsets)==len(frames)+1

    def cached(i):
        return values[offsets[i]:offsets[i+1]].tolist()

    def make():
        return install(components.build('candidate'),'combined')

    args.out.mkdir(parents=True,exist_ok=True)
    if args.role=='work':
        policy=json.loads(args.policy.read_bytes())
        assert policy['protocol_sha256']==sha(args.protocol) and policy['feature_names']==list(FEATURES)
        with (args.out/(scene+'.jsonl')).open('w',encoding='utf8') as trace:
            report=full(frames,cached,make(),policy,trace)
        report['policy_sha256']=sha(args.policy)
        print(scene,report['identity']['mixed_minority_truth_points'],report['operations'],flush=True)
    else:
        reference, rows=observe(frames,cached,make)
        chosen=select_cases(frames,reference,rows,plan)
        samples=pair_replay(frames,cached,make,plan,chosen,reference)
        report=dict(census=rows,samples=samples)
        print(scene,len(rows),len(samples),sum(row['beneficial'] for row in samples),flush=True)
    save(args.out/(scene+'.json'),dict(report,scene=scene,protocol_sha256=sha(args.protocol)))


if __name__=='__main__':
    main()
