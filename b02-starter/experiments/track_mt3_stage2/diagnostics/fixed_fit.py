"""Equal-budget fixed-clip fits with only the propagation assistance rule changed."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--project',type=Path,required=True)
p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--steps',type=int,default=300)
a=p.parse_args()
sys.path[:0]=[str(a.upstream),str(a.project),str(a.project/'experiments/track_mt3_stage2')]
import numpy as np
import torch
from adapter import make_model,make_window,associate,IgnoreUnknownCriterion,HISTORY
from b02.replay import open_scenes,_scans
from track_mt3.tracking import OnlineTracker
from track_mt3.matching.cta import CrossFrameTargetAlignment

torch.set_num_threads(4)
scenes={}
with open_scenes(a.data) as (_,opener):
    for name in ['train_001','train_015']:
        with opener(name) as source:
            scenes[name]=[]
            for frame in _scans(source):
                scenes[name].append(frame)
                if frame[0]>=119:break


def windows(name,start,count,supervised):
    history=deque(maxlen=HISTORY);result=[];ages={}
    for circle,points,labels in scenes[name][max(0,start-HISTORY+1):start+count]:
        for label in labels.values():
            if label>0:ages[label]=ages.get(label,0)+1
        history.append((circle,points,labels if supervised else None))
        if circle>=start:result.append(make_window(history,circle,labels if supervised else None,ages))
    return result


@torch.no_grad()
def evaluate(model,name,start,count,teacher=False):
    model.eval();online=OnlineTracker(model);tracks=model.empty_tracks()
    align=CrossFrameTargetAlignment(model.config.loss)
    truth=hits=outputs=clutter=raw=0
    for window,(_,points,labels) in zip(windows(name,start,count,teacher),scenes[name][start:start+count]):
        if teacher:
            pred=model.forward_window(window,tracks)
            assigned=align(pred,tracks,window.target_positions,window.target_ids)
            tracks,selected=model.qtm(pred,tracks,query_target_ids=assigned.query_target_ids,
                                      query_track_ids=assigned.query_target_ids,teacher_force=True)
            positions=pred.positions[selected].cpu().numpy()
        else:positions=online.step(window).positions.numpy()
        result=associate(positions,np.arange(len(positions)),points)
        truth+=sum(v>0 for v in labels.values());hits+=sum(labels[r['PntID']]>0 for r in result)
        clutter+=sum(labels[r['PntID']]==0 for r in result);outputs+=len(result);raw+=len(positions)
    return {'truth':truth,'hits':hits,'outputs':outputs,'raw':raw,'clutter':clutter,
            'recall':hits/truth,'precision':hits/(hits+clutter) if hits+clutter else 0}


a.out.mkdir(parents=True,exist_ok=False)
start_time=time.monotonic()
clip=windows('train_001',100,4,True)
report={}
for variant in ('legacy_forced_first_two','respect_probability_05','fully_free'):
    torch.manual_seed(20261111)
    model=make_model(a.upstream)
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu')['model'])
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-5)
    criterion=IgnoreUnknownCriterion(model.config.loss)
    history=[]
    for step in range(a.steps+1):
        if step in (0,100,a.steps):
            row={'step':step,'fit_free':evaluate(model,'train_001',100,4),
                 'fit_oracle':evaluate(model,'train_001',100,4,True),
                 'dev_free':evaluate(model,'train_015',100,20)}
            history.append(row)
            print(json.dumps({'variant':variant,**row}),flush=True)
        if step==a.steps:break
        calls=[0]
        def change_assistance(module,args,kwargs):
            index=calls[0];calls[0]+=1
            if variant=='legacy_forced_first_two' and index<2:kwargs['teacher_force']=True
            if variant=='fully_free':kwargs['teacher_force']=False
            return args,kwargs
        hook=model.qtm.register_forward_pre_hook(change_assistance,with_kwargs=True)
        model.train();optimizer.zero_grad(set_to_none=True)
        total,*components=model([clip],criterion=criterion,teacher_forcing_probability=.5)
        hook.remove()
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite fixed-clip loss')
        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
        optimizer.step()
        if (step+1)%50==0:
            print(json.dumps({'variant':variant,'step':step+1,'loss':float(total),
                              'components':[float(v) for v in components],
                              'elapsed':time.monotonic()-start_time}),flush=True)
    report[variant]=history
    torch.save({'model':model.state_dict(),'variant':variant,'step':a.steps},a.out/(variant+'.pt'))
    del optimizer,model;torch.cuda.empty_cache()
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'event':'complete','elapsed':time.monotonic()-start_time}),flush=True)
