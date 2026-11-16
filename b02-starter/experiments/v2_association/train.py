"""Train a residual association model on fit scenes, monitor separate calibration."""
import argparse
import json
from pathlib import Path
import sys
from time import monotonic

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def load_groups(folder,names):
    import numpy as np
    keys=('track','edges','base','valid','target');parts={k:[] for k in keys}
    for name in names:
        with np.load(folder/f'{name}.npz') as data:
            for key in keys:parts[key].append(data[key])
    return {k:np.concatenate(v) for k,v in parts.items()}


def select_objective(groups,objective):
    if objective=='residual':return groups
    # Learn identity choice conditional on a real continuation being available.
    # Birth/null decisions remain the frozen v1 rule in the rank-only adapter.
    keep=(~groups['target'][:,-1])&(groups['valid'][:,:-1].sum(1)>1)
    result={k:v[keep].copy() for k,v in groups.items()}
    result['valid'][:,-1]=False
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261116);p.add_argument('--steps',type=int,default=8000)
    p.add_argument('--batch',type=int,default=1024);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seconds',type=int,default=3300);p.add_argument('--device',default='cuda')
    p.add_argument('--checkpoints',nargs='+',type=int,default=[0,1000,3000,8000])
    p.add_argument('--objective',choices=('residual','pair_ranking'),default='residual')
    a=p.parse_args()
    import numpy as np
    import torch
    from experiments.v2_association.features import FEATURE_VERSION
    from experiments.v2_association.model import AssociationModel,association_loss
    from experiments.v2_association.run_stage1 import save,sha
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    start=monotonic();a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.pairs/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['feature_version'] in (FEATURE_VERSION,'b02-association-context-v2')
    assert not set(manifest['fit'])&set(manifest['calibration'])
    report=json.loads((a.pairs/'report.json').read_text(encoding='utf-8'));assert report['complete']
    fit=select_objective(load_groups(a.pairs/'fit',manifest['fit']),a.objective)
    cal=select_objective(load_groups(a.pairs/'calibration',manifest['calibration']),a.objective)
    assert len(fit['track']) and len(cal['track'])
    assert all(np.isfinite(fit[k]).all() for k in ('track','edges','base'))
    assert fit['target'].any(1).all() and (fit['target']<=fit['valid']).all()
    model=AssociationModel(manifest['track_dim'],manifest['edge_dim']).to(a.device)
    real=fit['edges'][fit['valid'][:,:-1]]
    with torch.no_grad():
        for prefix,values in (('track',fit['track']),('edge',real)):
            getattr(model,prefix+'_mean').copy_(torch.as_tensor(values.mean(0),device=a.device))
            getattr(model,prefix+'_std').copy_(torch.as_tensor(np.maximum(values.std(0),.01),device=a.device))
    arrays={k:torch.as_tensor(v,device=a.device) for k,v in fit.items()}
    category=~arrays['target'][:,-1]
    if a.objective=='pair_ranking':
        category=((arrays['edges'][:,:,11]>=.5)&arrays['valid'][:,:-1]&~arrays['target'][:,:-1]).any(1)
    positives=torch.nonzero(category).flatten();negatives=torch.nonzero(~category).flatten()
    assert len(positives) and len(negatives)
    positive_fraction=len(positives)/len(fit['track'])
    # Balanced minibatches with importance correction preserve the observed fit prior.
    validation={k:torch.as_tensor(v,device=a.device) for k,v in cal.items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
    metadata={'format':'b02-residual-association-v1','feature_version':manifest['feature_version'],
              'seed':a.seed,'batch':a.batch,'lr':a.lr,'steps':a.steps,
              'objective':a.objective,
              'fit_groups':len(fit['track']),'calibration_groups':len(cal['track']),
              'fit_positive_fraction':positive_fraction,'parameters':sum(v.numel() for v in model.parameters()),
              'sampling':'half positive, half null, prior corrected' if a.objective=='residual' else 'half hard competing high-score edge rows, half other continuation rows, prior corrected',
              'normalization':'fit means/std only; std floor .01; standardized features clipped to +/-10',
              'pairs_manifest_sha256':sha(a.pairs/'manifest.json'),'pairs':manifest,
              'sources':{s.name:sha(s) for s in Path(__file__).parent.glob('*.py')},
              'device':a.device,'torch':str(torch.__version__),'seconds_limit':a.seconds}
    save(a.out/'manifest.json',metadata)
    def metrics():
        model.eval();loss=correct=total=0
        with torch.no_grad():
            for i in range(0,len(cal['track']),2048):
                b={k:v[i:i+2048] for k,v in validation.items()};n=len(b['track'])
                loss+=float(association_loss(model,**b))*n
                logits=(-b['base']/.25-model(b['track'],b['edges'])).masked_fill(~b['valid'],-1e9)
                chosen=logits.argmax(1);correct+=int(b['target'].gather(1,chosen[:,None]).sum());total+=n
        return {'calibration_loss':loss/total,'calibration_row_accuracy':correct/total}
    losses=[];completed=False;step=0
    try:
        for step in range(a.steps+1):
            if monotonic()-start>a.seconds:raise TimeoutError('Training wall time limit')
            if step in set(a.checkpoints)|{a.steps}:
                row={'step':step,'seconds':round(monotonic()-start,3),**metrics()}
                if losses:row['fit_loss_recent']=sum(losses)/len(losses)
                losses=[]
                torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                            'step':step,'metadata':metadata},a.out/f'step-{step:05d}.pt')
                with (a.out/'history.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            if step==a.steps:completed=True;break
            model.train();n=a.batch//2
            ix=torch.cat((positives[torch.randint(len(positives),(n,),device=a.device)],
                          negatives[torch.randint(len(negatives),(a.batch-n,),device=a.device)]))
            weight=torch.where(category[ix],2*positive_fraction,2*(1-positive_fraction))
            batch={k:v[ix] for k,v in arrays.items()}
            loss=association_loss(model,**batch,weights=weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();losses.append(float(loss.detach()))
    finally:
        save(a.out/'report.json',{'complete':completed,'step':step,'wall_seconds':monotonic()-start,
             'peak_cuda_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else 0})


if __name__=='__main__':main()
