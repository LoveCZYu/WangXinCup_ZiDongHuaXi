"""Nested clip-count probe with matched-total and matched-per-clip budgets."""
import argparse
from collections import deque
import json
from pathlib import Path
import shutil
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--project',type=Path,required=True)
p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--seconds',type=int,default=840)
a=p.parse_args()
sys.path[:0]=[str(a.upstream),str(a.project),str(a.project/'experiments/track_mt3_stage2')]
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from adapter import make_model,make_window,associate,coordinates,IgnoreUnknownCriterion,HISTORY,GATE
from b02.replay import open_scenes,_scans,_checked_outputs,_identity
from track_mt3.tracking import OnlineTracker
from track_mt3.matching.cta import CrossFrameTargetAlignment

torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
started=time.monotonic();a.out.mkdir(parents=True,exist_ok=False)
source=a.out/'source';source.mkdir()
for path in [Path(__file__),a.project/'experiments/track_mt3_stage2/adapter.py',a.upstream/'b02-stage2.patch']:
    shutil.copy2(path,source/path.name)
names=['train_001','train_002','train_004','train_005']
dev=['train_015','train_016']
split=json.loads((a.project/'splits/v1.json').read_text())
assert set(names)<=set(split['train']) and set(dev)<=set(split['dev'])
specs=[(name,start,4) for name in names for start in (100,300)]
unseen=[(name,start,20) for name in names for start in (200,400)]
external=[(name,start,20) for name in dev for start in (200,400)]
manifest={'seed':20261112,'initialization':'random','nested_clips':specs,
          'counts':[1,2,4,8],'equal_total_updates':300,'equal_exposures_per_clip':150,
          'max_steps_by_count':{str(n):max(300,n*150) for n in [1,2,4,8]},
          'same_file_unseen':unseen,'different_file':external,'holdout_used':False,
          'soft_seconds':a.seconds,'batch':1,'clip_windows':4,'learning_rate':5e-5,
          'teacher_forcing':.5,'thresholds':[.75,.9],
          'note':'One seed diagnostic; clip count also expands diversity. Oracle uses labels only as diagnostic, never a deployable score.'}
(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
frames={}
with open_scenes(a.data) as (_,opener):
    for name in names+dev:
        with opener(name) as file:
            frames[name]=[]
            for record in _scans(file):
                frames[name].append(record)
                if record[0]>=419:break

def make_windows(spec,supervised):
    name,start,count=spec;history=deque(maxlen=HISTORY);ages={};result=[]
    for circle,points,labels in frames[name][max(0,start-HISTORY+1):start+count]:
        for label in labels.values():
            if label>0:ages[label]=ages.get(label,0)+1
        history.append((circle,points,labels if supervised else None))
        if circle>=start:result.append(make_window(history,circle,labels if supervised else None,ages))
    return result

training=[make_windows(spec,True) for spec in specs]
cache={(spec,teacher):make_windows(spec,teacher) for spec in specs+unseen+external for teacher in [False,True]}

@torch.no_grad()
def evaluate(model,chosen,teacher=False):
    model.eval();alignment=CrossFrameTargetAlignment(model.config.loss)
    total={'truth':0,'hits':0,'clutter':0,'unknown':0,'raw_tracks':0,'geometry_matches':0}
    distances=[];probabilities=[];per_clip=[]
    for spec in chosen:
        name,start,count=spec;online=OnlineTracker(model);tracks=model.empty_tracks()
        ct=ch=0
        for window,(_,points,labels) in zip(cache[spec,teacher],frames[name][start:start+count]):
            if teacher:
                prediction=model.forward_window(window,tracks)
                assignment=alignment(prediction,tracks,window.target_positions,window.target_ids)
                tracks,selected=model.qtm(prediction,tracks,query_target_ids=assignment.query_target_ids,
                                         query_track_ids=assignment.query_target_ids,teacher_force=True)
                positions=prediction.positions[selected].cpu().numpy()
                ids=np.arange(len(positions))
            else:
                output=online.step(window);prediction=output.prediction
                positions=output.positions.numpy();ids=output.track_ids.numpy()
            result=associate(positions,ids,points)
            _checked_outputs(result,{_identity(point) for point in points})
            truth_points=[point for point in points if labels[point['PntID']]>0]
            nt=len(truth_points);nh=sum(labels[r['PntID']]>0 for r in result)
            total['truth']+=nt;total['hits']+=nh;total['raw_tracks']+=len(positions)
            total['clutter']+=sum(labels[r['PntID']]==0 for r in result)
            total['unknown']+=sum(labels[r['PntID']]==-1 for r in result);ct+=nt;ch+=nh
            if nt:
                residual=np.abs(prediction.positions.cpu().numpy()[:,None,:]-coordinates(truth_points)[None,:,:])
                d=(residual/np.array(GATE)).max(axis=2)
                rows,cols=linear_sum_assignment(np.concatenate([np.where(d<=1,d,1e6),np.full((len(d),len(d)),10.)],axis=1))
                total['geometry_matches']+=int(sum(j<nt and d[i,j]<=1 for i,j in zip(rows,cols)))
                ri,ci=linear_sum_assignment(np.linalg.norm(residual,axis=2))
                distances.extend(residual[ri,ci].tolist())
                probabilities.extend(prediction.probabilities.cpu().numpy().ravel()[ri].tolist())
        per_clip.append({'scene':name,'start':start,'truth':ct,'hits':ch,'recall':ch/ct if ct else None})
    known=total['hits']+total['clutter'];truth=total['truth']
    return {**total,'recall':total['hits']/truth,'precision':total['hits']/known if known else 0,
            'point_f1':2*total['hits']/(truth+known),'geometry_recall':total['geometry_matches']/truth,
            'median_assignment_error':np.median(distances,axis=0).tolist(),
            'median_assigned_probability':float(np.median(probabilities)),'per_clip':per_clip}

def emit(event):
    event['elapsed']=time.monotonic()-started
    with (a.out/'events.jsonl').open('a') as file:file.write(json.dumps(event)+'\n')
    compact={**event}
    if 'metrics' in compact:
        compact['metrics']={k:{m:v[m] for m in ['recall','precision','point_f1','geometry_recall','median_assigned_probability']} for k,v in compact['metrics'].items()}
    print(json.dumps(compact),flush=True)

report={'manifest':manifest,'evaluations':[]}
for n in manifest['counts']:
    torch.manual_seed(manifest['seed']);model=make_model(a.upstream)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-5);criterion=IgnoreUnknownCriterion(model.config.loss)
    maximum=max(300,n*150);losses=[];peak=0
    milestones={0,150,300,n*150,maximum}
    for step in range(maximum+1):
        deadline=step>0 and time.monotonic()-started>=a.seconds-20
        if step in milestones or deadline:
            chosen_names={spec[0] for spec in specs[:n]}
            same=[spec for spec in unseen if spec[0] in chosen_names]
            metrics={'seen_free':evaluate(model,specs[:n]),'seen_oracle':evaluate(model,specs[:n],True)}
            if step in (300,n*150,maximum) or deadline:
                metrics['same_file_unseen']=evaluate(model,same)
                metrics['different_file']=evaluate(model,external)
            row={'count':n,'step':step,'exposures_per_clip':step/n,'metrics':metrics,
                 'peak_allocated_gib':peak,'budget_stop':deadline}
            report['evaluations'].append(row);emit({'event':'evaluation',**row})
            (a.out/'report.json').write_text(json.dumps(report,indent=2))
            if step in (300,n*150,maximum) or deadline:
                torch.save({'model':model.state_dict(),'count':n,'step':step,'manifest':manifest},a.out/f'clips-{n}-step-{step}.pt')
        if step==maximum or deadline:break
        torch.manual_seed(manifest['seed']*10000+step)
        model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        total,*_=model([training[step%n]],criterion=criterion,teacher_forcing_probability=.5)
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite coverage loss')
        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
        optimizer.step();losses.append(float(total.detach()));peak=max(peak,torch.cuda.max_memory_allocated()/2**30)
        del total
        if (step+1)%100==0:emit({'event':'train','count':n,'step':step+1,'loss_ma50':sum(losses[-50:])/len(losses[-50:]),'peak_gib':peak})
    del model,optimizer,criterion;torch.cuda.empty_cache()
    if deadline:break
report['complete']=len(report['evaluations'])>0 and report['evaluations'][-1]['count']==8 and report['evaluations'][-1]['step']==1200
report['elapsed_seconds']=time.monotonic()-started
(a.out/'report.json').write_text(json.dumps(report,indent=2))
emit({'event':'complete','all_planned_updates_completed':report['complete']})
