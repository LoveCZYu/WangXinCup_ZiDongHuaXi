"""Fit-validation-only fusion of frozen evidence and the learned head."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.continuation.learn import load_model,read_samples
from experiments.continuation.memory_guard import MemoryGuardTracker
from experiments.continuation.tracker import ContinuationTracker
from experiments.id_recovery.tracker import cutoff
from experiments.v2_association.run_stage1 import save,sha

EXTRA=186


class FusionTracker(ContinuationTracker):
    def __init__(self,scorer,affinity,head,config):
        super().__init__(scorer,affinity)
        self.head=head
        self.fusion=config

    def intercept(self,matched,used,scan,positions,points,scores):
        rows=self.candidates(matched,used,scan,positions,points,scores)
        if not rows:return
        self.vectorize(rows,scan,positions,points,scores)
        self.samples.extend(rows)
        logits=self.head.score(np.stack([row['features'] for row in rows]))
        amount=self.fusion['amount']
        options,competing=defaultdict(list),defaultdict(list)
        for row,logit in zip(rows,logits,strict=True):
            corrected=(1-amount)*float(np.clip(row['logit'],-6,6))+amount*float(np.clip(logit,-8,8))
            row['strength']=corrected-1.2*row['temporal']-.5*row['velocity']
            row['eligible']=bool(corrected>=cutoff(self.fusion['threshold']) and row['temporal']<=.85 and row['velocity']<=1.)
            self.proposals.append(row)
            self.counters['prebirth_eligible']+=row['eligible']
            options[row['old']].append(row);competing[row['new']].append(row)
        ranked=[]
        for old,choices in options.items():
            choices.sort(key=lambda row:row['strength'],reverse=True)
            best=choices[0]
            if not best['eligible']:continue
            if len(choices)>1 and best['strength']-choices[1]['strength']<.25:continue
            if any(row['old']!=old and best['strength']-row['strength']<.25 for row in competing[best['new']]):continue
            ranked.append(best)
        for row in sorted(ranked,key=lambda row:row['strength'],reverse=True):
            old,new,index=row['old'],row['new'],row['index']
            if old in used or matched.get(new)!=index:continue
            if row['retired']:self.restore(old)
            del matched[new];matched[old]=index;used.add(old);row['selected']=True
            self.decisions.append(('fusion_retired' if row['retired'] else 'fusion_live',old,index,new))


class MemoryFusionTracker(MemoryGuardTracker,FusionTracker):
    pass


def evaluate(features,valid,labels,head,amount,threshold):
    prior=np.clip(features[:,:,EXTRA+28],-6,6)
    corrected=(1-amount)*prior+amount*np.clip(head,-8,8)
    temporal=features[:,:,EXTRA+24]/1.75**2
    velocity=features[:,:,EXTRA+27]
    strength=corrected-1.2*temporal-.5*velocity
    strength=np.where(valid,strength,-1e4)
    winner=strength.argmax(1)
    top=strength[np.arange(len(winner)),winner]
    second=np.partition(strength,-2,axis=1)[:,-2]
    eligible=(corrected>=cutoff(threshold))&(temporal<=.85)&(velocity<=1.)&valid
    chosen=eligible[np.arange(len(winner)),winner]&(top-second>=.25)
    correct=labels[np.arange(len(winner)),winner][chosen].sum()
    selected=int(chosen.sum());wrong=selected-int(correct)
    return {'amount':amount,'threshold':threshold,'selected':selected,'correct':int(correct),'wrong':wrong,
            'precision':float(correct)/max(1,selected),'recall':float(correct)/max(1,int(labels.any(1).sum())),
            'utility':int(correct)-5*wrong}


def calibrate(args):
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    validation=plan['fit64'][::4]
    data=read_samples(args.samples,validation,args)
    features,valid,labels=(data[k].numpy() for k in ('features','valid','labels'))
    args.out.mkdir(parents=True,exist_ok=False)
    report={}
    for seed in (20261116,20261117):
        path=args.training/f'{seed}.pt'
        model,metadata=load_model(path,args.prior,args.affinity)
        head=model.score(features.reshape(-1,features.shape[-1])).reshape(valid.shape)
        baseline=evaluate(features,valid,labels,head,0.,.80)
        table=[evaluate(features,valid,labels,head,amount,threshold)
               for amount in (0.,.25,.50,.75,1.) for threshold in (.70,.80,.90,.95)]
        feasible=[row for row in table if row['selected']>=20 and row['precision']>=baseline['precision'] and row['utility']>=baseline['utility']]
        selected=max(feasible,key=lambda row:(row['utility'],row['recall'],row['precision'],-row['amount'])) if feasible else baseline
        result={**selected,'baseline':baseline,'validation_table':table,'seed':seed,
                'weights_sha256':sha(path),'source_sha256':sha(__file__),
                'head_validation16':validation,'calibration_used':False}
        save(args.out/f'{seed}.json',result);report[str(seed)]=result
        print(json.dumps({'seed':seed,'selection':selected,'baseline':baseline}),flush=True)
    save(args.out/'report.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('samples','training','prior','affinity','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);calibrate(args)


if __name__=='__main__':main()
