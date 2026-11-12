"""Paired, bounded simulated-pretraining versus random-initialization experiment."""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--project',type=Path,required=True)
p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--pretrained',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--steps',type=int,default=500)
p.add_argument('--seconds',type=int,default=840)
a=p.parse_args()
sys.path[:0]=[str(a.upstream),str(a.project),str(a.project/'experiments/track_mt3_stage2')]
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from adapter import make_model,transfer,make_window,associate,IgnoreUnknownCriterion,HISTORY
from b02.replay import open_scenes,_scans,_checked_outputs,_identity
from track_mt3.tracking import OnlineTracker

torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
started=time.monotonic()
a.out.mkdir(parents=True,exist_ok=False)
source=a.out/'source';source.mkdir()
for path in [Path(__file__), a.project/'experiments/track_mt3_stage2/adapter.py', a.upstream/'b02-stage2.patch']:
    shutil.copy2(path,source/path.name)
train_names=['train_001','train_002','train_004','train_005']
dev_names=['train_015','train_016']
split=json.loads((a.project/'splits/v1.json').read_text())
assert set(train_names)<=set(split['train']) and set(dev_names)<=set(split['dev'])
assert not (set(train_names)|set(dev_names))&set(split['holdout'])
train_specs=[(name,start,4) for name in train_names for start in (100,300)]
eval_specs={'seen':train_specs,
            'same_file_unseen':[(name,start,20) for name in train_names for start in (200,400)],
            'different_file':[(name,start,20) for name in dev_names for start in (200,400)]}
# No input history or target frame overlaps between seen and unseen windows.
def covered(spec):
    name,start,count=spec
    return {(name,circle) for circle in range(max(0,start-HISTORY+1),start+count)}
assert not set.union(*(covered(s) for s in train_specs)) & set.union(*(covered(s) for s in eval_specs['same_file_unseen']))
manifest={'seeds':[20261112,20261113],'max_updates_per_arm_per_seed':a.steps,
          'variants':['pretrained','random'],'train_specs':train_specs,'evaluation':eval_specs,
          'initialization_source':str(a.pretrained),'source_sha256':hashlib.sha256(a.pretrained.read_bytes()).hexdigest(),
          'soft_total_seconds':a.seconds,'batch':1,'clip_windows':4,'learning_rate':5e-5,
          'gradient_clip_norm':10.,'teacher_forcing_probability':.5,
          'detection_threshold':.75,'tracking_threshold':.9,
          'eval_steps':[0,250,a.steps],'holdout_used':False,
          'selection':'Compare final equal-step checkpoints; no tuning or checkpoint selection on dev.',
          'limits':'Small fixed subset and two seeds; same-file unseen may share physical target identities. Not official scores.',
          'random_baseline':'Same 3D architecture, normal constructor initialization, fresh optimizer. Pretrained arm uses existing 2D-to-3D transfer recipe.'}
(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
frames={}
with open_scenes(a.data) as (_,opener):
    for name in train_names+dev_names:
        frames[name]=[]
        with opener(name) as file:
            for frame in _scans(file):
                frames[name].append(frame)
                if frame[0]>=419:break

def windows(spec,supervised):
    name,start,count=spec;history=deque(maxlen=HISTORY);result=[];ages={}
    for circle,points,labels in frames[name][max(0,start-HISTORY+1):start+count]:
        for label in labels.values():
            if label>0:ages[label]=ages.get(label,0)+1
        history.append((circle,points,labels if supervised else None))
        if circle>=start:result.append(make_window(history,circle,labels if supervised else None,ages))
    return result

clips=[windows(spec,True) for spec in train_specs]
eval_cache={key:[(spec,windows(spec,False)) for spec in specs] for key,specs in eval_specs.items()}

@torch.no_grad()
def evaluate(model):
    model.eval();results={}
    for group,entries in eval_cache.items():
        truth=hits=clutter=unknown=raw=identity_tp=0
        per_sequence=[]
        for (name,start,count),items in entries:
            tracker=OnlineTracker(model);pairs=Counter();sh=st=sc=0
            for window,(_,points,labels) in zip(items,frames[name][start:start+count]):
                output=tracker.step(window)
                result=associate(output.positions.numpy(),output.track_ids.numpy(),points)
                _checked_outputs(result,{_identity(point) for point in points})
                nt=sum(v>0 for v in labels.values());nh=sum(labels[r['PntID']]>0 for r in result)
                nc=sum(labels[r['PntID']]==0 for r in result)
                truth+=nt;hits+=nh;clutter+=nc;unknown+=sum(labels[r['PntID']]==-1 for r in result)
                raw+=len(output.track_ids);st+=nt;sh+=nh;sc+=nc
                pairs.update((labels[r['PntID']],r['BatchID']) for r in result if labels[r['PntID']]>0)
            ts=sorted({t for t,i in pairs});ids=sorted({i for t,i in pairs})
            matrix=np.array([[pairs[t,i] for i in ids] for t in ts])
            if matrix.size:
                i,j=linear_sum_assignment(-matrix);identity_tp+=int(matrix[i,j].sum())
            per_sequence.append({'scene':name,'start':start,'truth':st,'hits':sh,'clutter':sc})
        known=hits+clutter
        results[group]={'truth':truth,'hits':hits,'clutter':clutter,'unknown':unknown,'raw_tracks':raw,
                        'recall':hits/truth,'precision':hits/known if known else 0.,
                        'point_f1':2*hits/(truth+known),'identity_consistency_f1':2*identity_tp/(truth+known),
                        'sequences':per_sequence}
    return results

def emit(event):
    event['elapsed_seconds']=time.monotonic()-started
    with (a.out/'events.jsonl').open('a') as file:file.write(json.dumps(event)+'\n')
    print(json.dumps(event),flush=True)

report={'manifest':manifest,'runs':[]}
for seed_index,seed in enumerate(manifest['seeds']):
    arms={}
    seed_start=time.monotonic()
    remaining_seeds=len(manifest['seeds'])-seed_index
    seed_budget=max(0,(a.seconds-(seed_start-started))/remaining_seeds)
    for variant in manifest['variants']:
        torch.manual_seed(seed)
        model=make_model(a.upstream)
        if variant=='pretrained':
            transferred=transfer(model,a.pretrained)
            (a.out/f'transfer-{seed}.json').write_text(json.dumps(transferred,indent=2))
        arms[variant]=(model,torch.optim.Adam(model.parameters(),lr=5e-5),IgnoreUnknownCriterion(model.config.loss))
    order=list(range(len(clips)))
    random.Random(seed).shuffle(order)
    train_seconds={v:0. for v in arms};last_losses={};peak=0.
    for step in range(a.steps+1):
        at_budget=step>0 and time.monotonic()-seed_start>=seed_budget-20
        if step in (0,250,a.steps) or at_budget:
            for variant,(model,_,_) in arms.items():
                metrics=evaluate(model)
                item={'seed':seed,'variant':variant,'step':step,'metrics':metrics,
                      'training_seconds':train_seconds[variant]}
                report['runs'].append(item)
                emit({'event':'evaluation',**item})
            (a.out/'report.json').write_text(json.dumps(report,indent=2))
        if step==a.steps or at_budget:
            for variant,(model,_,_) in arms.items():
                torch.save({'model':model.state_dict(),'seed':seed,'variant':variant,'step':step,
                            'experiment_manifest':manifest},a.out/f'{seed}-{variant}.pt')
            emit({'event':'seed_complete','seed':seed,'step':step,'budget_stop':at_budget,'peak_training_allocated_gib':peak})
            break
        # Same clip order, same per-step RNG seed, same optimization rules.
        # Alternate execution order so neither arm always runs first.
        variants=list(arms) if step%2==0 else list(reversed(arms))
        clip=clips[order[step%len(order)]]
        for variant in variants:
            model,optimizer,criterion=arms[variant]
            torch.manual_seed(seed*10000+step)
            model.train();optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats();before=time.monotonic()
            total,*_=model([clip],criterion=criterion,teacher_forcing_probability=.5)
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite paired-training loss')
            total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            train_seconds[variant]+=time.monotonic()-before;last_losses[variant]=float(total.detach())
            peak=max(peak,torch.cuda.max_memory_allocated()/2**30)
            del total
        if (step+1)%50==0:emit({'event':'train','seed':seed,'step':step+1,'losses':last_losses,'training_seconds':train_seconds})
    del model,optimizer,criterion,arms
    torch.cuda.empty_cache()
report['complete']=True
report['total_seconds']=time.monotonic()-started
report['note']='No long continuation; final steps paired within each seed. Diagnostic checkpoints only.'
(a.out/'report.json').write_text(json.dumps(report,indent=2))
emit({'event':'complete','total_seconds':report['total_seconds']})
