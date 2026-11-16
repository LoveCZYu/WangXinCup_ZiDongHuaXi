"""Collect wide-policy fit rollouts and learn prefix/identity continuation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from b02.replay import _scans,open_scenes
from experiments.continuation.tracker import ContinuationTracker,DIMENSION,MAX_OLD
from experiments.id_recovery.affinity import load_affinity
from experiments.v2_association.run_stage1 import save,sha
from experiments.v3_tracking.runtime import load_scorer


class ContinuationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.zeros(DIMENSION))
        self.register_buffer('std',torch.ones(DIMENSION))
        self.net=nn.Sequential(nn.Linear(DIMENSION,64),nn.ReLU(),nn.Dropout(.10),
                               nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,x):
        return self.net(((x-self.mean)/self.std).clamp(-8,8)).squeeze(-1)

    @torch.inference_mode()
    def score(self,features):
        return self(torch.from_numpy(features)).numpy()


def load_model(path,prior,affinity):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    assert checkpoint['prior_sha256']==sha(prior) and checkpoint['affinity_sha256']==sha(affinity)
    assert checkpoint['learner_sha256']==sha(__file__)
    assert checkpoint['tracker_sha256']==sha(Path(__file__).with_name('tracker.py'))
    model=ContinuationModel().eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    return model,checkpoint


def historical_target(truth,history):
    """Return None for an unobservable history; zero for clutter/impure owners."""
    if truth<0:
        return None
    known={identity:n for identity,n in history.items() if identity>=0 and n}
    if not known:
        return None
    owner=max(known,key=known.get)
    purity=known[owner]/sum(known.values())
    positive=truth>0 and owner==truth and purity>=.80
    return float(positive),float(np.log1p(known.get(truth,0))) if positive else 1.


def collect(args,plan):
    manifest=json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['data_sha256']==sha(args.data) and manifest['fit']==plan['fit64']
    assert not set(plan['fit64'])&set(plan['calibration']+plan['confirmation16']+plan['sealed_excluded'])
    scorer,memory,_,_=load_scorer(args.prior)
    assert memory and 0<=args.index<args.shards
    affinity=load_affinity(args.affinity,args.prior)
    args.out.mkdir(parents=True,exist_ok=True)
    with open_scenes(args.data) as (_,opener):
        for scene in plan['fit64'][args.index::args.shards]:
            with opener(scene) as source:frames=list(_scans(source))
            with np.load(args.cache/'fit'/f'{scene}.npz') as cache:
                scores,offsets=cache['point_scores'],cache['point_offsets']
            assert len(offsets)==len(frames)+1
            tracker=ContinuationTracker(scorer,affinity)
            histories=defaultdict(Counter)
            inputs,valids,labels,priorities=[],[],[],[]
            for frame,(_,points,truth) in enumerate(frames):
                outputs=tracker(points,scores[offsets[frame]:offsets[frame+1]].tolist())
                groups=defaultdict(list)
                for row in tracker.samples:groups[row['new']].append(row)
                for rows in groups.values():
                    label=truth[points[rows[0]['index']]['PntID']]
                    if label<0:continue
                    selected=sorted(rows,key=lambda row:row['strength'],reverse=True)[:MAX_OLD]
                    x=np.zeros((MAX_OLD,DIMENSION),dtype=np.float32)
                    valid=np.zeros(MAX_OLD,dtype=bool)
                    target=np.zeros(MAX_OLD,dtype=np.float32)
                    priority=np.zeros(MAX_OLD,dtype=np.float32)
                    for col,row in enumerate(selected):
                        result=historical_target(label,histories[row['old']])
                        x[col]=row['features']
                        if result is not None:
                            valid[col]=True;target[col],priority[col]=result
                    if valid.any():
                        inputs.append(x);valids.append(valid);labels.append(target);priorities.append(priority)
                # Only past outputs label subsequent groups. Truth never enters the tracker.
                for point in outputs:histories[point['BatchID']][truth[point['PntID']]]+=1
            destination=args.out/f'{scene}.npz'
            assert not destination.exists()
            np.savez_compressed(destination,features=np.asarray(inputs,dtype=np.float32).reshape(-1,MAX_OLD,DIMENSION),
                valid=np.asarray(valids,dtype=bool).reshape(-1,MAX_OLD),
                labels=np.asarray(labels,dtype=np.float32).reshape(-1,MAX_OLD),
                priority=np.asarray(priorities,dtype=np.float32).reshape(-1,MAX_OLD),
                prior_sha256=sha(args.prior),affinity_sha256=sha(args.affinity),
                tracker_sha256=sha(Path(__file__).with_name('tracker.py')),learner_sha256=sha(__file__))
            print(scene,'groups',len(inputs),'positive_edges',int(sum(y.sum() for y in labels)),flush=True)


def read_samples(root,names,args):
    values=defaultdict(list)
    for scene in names:
        with np.load(root/f'{scene}.npz') as data:
            for key,path in (('prior_sha256',args.prior),('affinity_sha256',args.affinity),
                             ('tracker_sha256',Path(__file__).with_name('tracker.py')),('learner_sha256',__file__)):
                assert data[key].item()==sha(path),(scene,key)
            for name in ('features','valid','labels','priority'):values[name].append(data[name])
    result={name:torch.from_numpy(np.concatenate(rows)) for name,rows in values.items()}
    assert len(result['features'])>0
    return result


def objective(model,data):
    logits=model(data['features'])
    valid=data['valid']
    labels=data['labels']
    binary=F.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    binary=(binary*(1+(1-labels)))[valid].mean()
    targets=labels*data['priority']*valid
    support=targets.sum(1,keepdim=True)
    targets=targets/support.clamp_min(1e-8)
    targets=torch.cat((targets,(support==0).float()),dim=1)
    grouped=torch.cat((logits.masked_fill(~valid,-1e4),logits.new_zeros(len(logits),1)),dim=1)
    ranked=-(targets*F.log_softmax(grouped,dim=1)).sum(1).mean()
    return binary+.5*ranked


@torch.inference_mode()
def validate(model,data):
    logits=model(data['features']).masked_fill(~data['valid'],-1e4)
    top,columns=logits.max(1)
    second=logits.topk(2,dim=1).values[:,1]
    correct=data['labels'].gather(1,columns[:,None]).squeeze(1)
    possible=data['labels'].bool().any(1)
    rows=[]
    for threshold in (.50,.60,.70,.80,.85,.90,.95,.98,.99,.995):
        selected=(top.sigmoid()>=threshold)&(top-second>=.25)
        hits=float(correct[selected].sum())
        rows.append({'threshold':threshold,'selected':int(selected.sum()),'correct':int(hits),
                     'precision':hits/max(1,int(selected.sum())),
                     'recall':hits/max(1,int(possible.sum()))})
    return rows


def choose_threshold(rows,precision):
    feasible=[row for row in rows if row['precision']>=precision and row['selected']>=20]
    if not feasible:return {'threshold':.999999,'feasible':False,'target_precision':precision}
    best=max(feasible,key=lambda row:(row['recall'],row['precision'],row['threshold']))
    return {**best,'feasible':True,'target_precision':precision}


def train(args,plan):
    validation=plan['fit64'][::4]
    names=[name for name in plan['fit64'] if name not in validation]
    assert len(names)==48 and len(validation)==16
    fit=read_samples(args.samples,names,args)
    val=read_samples(args.samples,validation,args)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    args.out.mkdir(parents=True,exist_ok=False)
    fit={k:v.to(device) for k,v in fit.items()};val={k:v.to(device) for k,v in val.items()}
    reports={}
    for seed in (20261116,20261117):
        torch.manual_seed(seed)
        model=ContinuationModel()
        observed=fit['features'][fit['valid']].cpu()
        model.mean.copy_(observed.mean(0));model.std.copy_(observed.std(0,unbiased=False).clamp_min(.03))
        model.to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.03)
        best=float('inf');history=[]
        for step in range(1601):
            if step in (0,200,600,1000,1600):
                model.eval()
                with torch.no_grad():loss=float(objective(model,val))
                rows=validate(model,val)
                selection={name:choose_threshold(rows,p) for name,p in (('learned',.97),('learned_safe',.99))}
                row={'seed':seed,'step':step,'loss':loss,'selection':selection,'threshold_table':rows}
                history.append(row);print(json.dumps({k:v for k,v in row.items() if k!='threshold_table'}),flush=True)
                if loss<best:
                    best=loss
                    checkpoint={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        'seed':seed,'step':step,'prior_sha256':sha(args.prior),'affinity_sha256':sha(args.affinity),
                        'tracker_sha256':sha(Path(__file__).with_name('tracker.py')),'learner_sha256':sha(__file__),
                        'thresholds':{name:item['threshold'] for name,item in selection.items()},
                        'threshold_selection':selection,'fit':names,'head_validation16':validation}
                    torch.save(checkpoint,args.out/f'{seed}.pt')
            if step==1600:break
            model.train()
            index=torch.randint(len(fit['features']),(64,),device=device)
            loss=objective(model,{name:value[index] for name,value in fit.items()})
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        reports[str(seed)]={'history':history}
    save(args.out/'report.json',{'device':device,'fit_groups':len(fit['features']),'validation_groups':len(val['features']),
        'fit':names,'head_validation16':validation,'calibration_used':False,'seeds':reports})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('collect','train'))
    for name in ('data','cache','prior','affinity','out','samples'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--index',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    (collect if args.command=='collect' else train)(args,plan)


if __name__=='__main__':main()
