"""Controlled data/prefix/memory training with fixed seeds and reserved scene roles."""
import argparse
import json
from pathlib import Path
import sys
from time import monotonic

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import save,sha


def load_groups(folder,names,memory=False,labels='pure'):
    import numpy as np
    groups={k:[] for k in ('track','edges','base','valid','target')};counts={'all_rows':0,'known':0,'rank_rows':0}
    for name in names:
        with np.load(folder/f'{name}.npz') as d:
            track=d['memory_track' if memory else 'track'];edges=d['memory_edges' if memory else 'edges']
            target=d['anchor_target' if labels=='anchor' else 'target'];valid=d['valid']
            known=d['anchor_known'] if labels=='anchor' else d['known'] if 'known' in d else np.ones(len(track),dtype=bool)
            keep=known&~target[:,-1]&(valid[:,:-1].sum(1)>1)
            counts['all_rows']+=len(track);counts['known']+=int(known.sum());counts['rank_rows']+=int(keep.sum())
            values={'track':track,'edges':edges,'base':d['base'],'valid':valid,'target':target}
            for key,value in values.items():groups[key].append(value[keep])
    result={k:np.concatenate(v) for k,v in groups.items()};result['valid'][:,-1]=False
    assert len(result['track']) and result['target'].any(1).all() and (result['target']<=result['valid']).all()
    return result,counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--memory',action='store_true');p.add_argument('--labels',choices=('pure','anchor'),default='pure')
    p.add_argument('--seed',type=int,default=20261116);p.add_argument('--steps',type=int,default=8000)
    p.add_argument('--batch',type=int,default=1024);p.add_argument('--seconds',type=int,default=3500)
    a=p.parse_args()
    import numpy as np
    import torch
    from experiments.v2_association.model import AssociationModel,association_loss
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    manifest=json.loads((a.pairs/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['fit']==plan['fit64'] and manifest['calibration']==plan['calibration']
    assert json.loads((a.pairs/'report.json').read_text(encoding='utf-8'))['complete']
    if a.memory or a.labels=='anchor':assert manifest['memory_available']
    torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(2)
    start=monotonic();a.out.mkdir(parents=True,exist_ok=False)
    fit,fit_counts=load_groups(a.pairs/'fit',manifest['fit'],a.memory,a.labels)
    cal,cal_counts=load_groups(a.pairs/'calibration',manifest['calibration'],a.memory,a.labels)
    assert all(np.isfinite(fit[k]).all() for k in ('track','edges','base'))
    device='cuda';assert torch.cuda.is_available()
    model=AssociationModel(fit['track'].shape[1],fit['edges'].shape[2]).to(device)
    real=fit['edges'][fit['valid'][:,:-1]]
    with torch.no_grad():
        for prefix,values in (('track',fit['track']),('edge',real)):
            getattr(model,prefix+'_mean').copy_(torch.as_tensor(values.mean(0),device=device))
            getattr(model,prefix+'_std').copy_(torch.as_tensor(np.maximum(values.std(0),.01),device=device))
    arrays={k:torch.as_tensor(v,device=device) for k,v in fit.items()};validation={k:torch.as_tensor(v,device=device) for k,v in cal.items()}
    category=((arrays['edges'][:,:,11]>=.5)&arrays['valid'][:,:-1]&~arrays['target'][:,:-1]).any(1)
    hard=torch.nonzero(category).flatten();easy=torch.nonzero(~category).flatten();fraction=len(hard)/len(category)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    metadata={'format':'b02-v3-association-training-v1','seed':a.seed,'memory':a.memory,'labels':a.labels,
        'feature_version':'b02-association-memory-v3' if a.memory else 'b02-association-context-v2',
        'track_dim':fit['track'].shape[1],'edge_dim':fit['edges'].shape[2],'steps':a.steps,'batch':a.batch,'lr':.001,
        'fit_counts':fit_counts,'calibration_counts':cal_counts,'fit_hard_fraction':fraction,
        'parameters':sum(p.numel() for p in model.parameters()),'torch':str(torch.__version__),
        'pairs_manifest_sha256':sha(a.pairs/'manifest.json'),'data_sha256':manifest['data_sha256'],
        'split_sha256':sha(ROOT/'splits/v3_tracking.json'),'pairs':manifest,'strategy':'rank_only','scale':1.,
        'normalization':'fit rows only, std floor .01, standardized clip +/-10',
        'sampling':'half hard competing-point rows with observed prior correction; uniform fallback if only one category',
        'runtime_sha256':{p.name:sha(p) for p in (Path(__file__).parent/'memory.py',Path(__file__).parent/'tracker.py')},
        'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
    save(a.out/'manifest.json',metadata)
    def validation_metrics():
        model.eval();loss=correct=n=0
        with torch.no_grad():
            for i in range(0,len(cal['track']),2048):
                b={k:v[i:i+2048] for k,v in validation.items()};size=len(b['track'])
                loss+=float(association_loss(model,**b))*size
                logits=(-b['base']/.25-model(b['track'],b['edges'])).masked_fill(~b['valid'],-1e9)
                correct+=int(b['target'].gather(1,logits.argmax(1)[:,None]).sum());n+=size
        return {'calibration_loss':loss/n,'calibration_row_accuracy':correct/n}
    completed=False;step=0;history=[];recent=[]
    try:
        for step in range(a.steps+1):
            if monotonic()-start>a.seconds:raise TimeoutError('Training budget')
            if step in {0,250,1000,3000,a.steps}:
                row={'step':step,'seconds':round(monotonic()-start,2),**validation_metrics()}
                if recent:row['fit_loss_recent']=sum(recent)/len(recent)
                recent=[];history.append(row)
                torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'step':step,'metadata':metadata},a.out/f'step-{step:05d}.pt')
                save(a.out/'history.json',history);print(json.dumps(row),flush=True)
            if step==a.steps:completed=True;break
            model.train()
            if len(hard) and len(easy):
                n=a.batch//2;indices=torch.cat((hard[torch.randint(len(hard),(n,),device=device)],easy[torch.randint(len(easy),(a.batch-n,),device=device)]))
                weights=torch.where(category[indices],2*fraction,2*(1-fraction))
            else:indices=torch.randint(len(category),(a.batch,),device=device);weights=None
            batch={k:v[indices] for k,v in arrays.items()};loss=association_loss(model,**batch,weights=weights)
            if not torch.isfinite(loss):raise ValueError('Nonfinite association loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            recent.append(float(loss.detach()))
        best=min(history,key=lambda row:row['calibration_loss'])
        save(a.out/'selected.json',{'rule':'minimum held-out row loss over 0/250/1000/3000/8000; full rollout still required','selected':best})
    finally:save(a.out/'report.json',{'complete':completed,'step':step,'wall_seconds':monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated()})


if __name__=='__main__':main()
