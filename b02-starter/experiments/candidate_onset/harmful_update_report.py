"""Fit the rollout-supervised detector and report the five paired scores."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from experiments.candidate_onset.harmful_update import FEATURES
from experiments.candidate_onset.joint_cost_report import scene_cutoff
from experiments.candidate_onset.propagation import sha
from experiments.candidate_onset.propagation_work_summary import read
from experiments.candidate_onset.update_stream_report import paired
from experiments.candidate_onset.visible_onset_hold import score


def load(root,role,plan,digest):
    results={}
    for name in plan[role]['scenes']:
        row=read(root/f'harmful-update-{role}'/(name+'.json'))
        assert row['scene']==name and row['protocol_sha256']==digest
        results[name]=row
    return results


def fit(rows,l2):
    data=np.asarray([r['x'] for r in rows])
    positive=np.asarray([r['beneficial'] for r in rows])
    assert positive.any() and (~positive).any(), 'The audit did not produce both replay outcome classes.'
    mean,scale=data.mean(0),np.maximum(data.std(0),.05)
    x=np.column_stack(((data-mean)/scale,np.ones(len(data))))
    weights=np.asarray([2. if r['correct_gain']<0 or r['wrong_reduction']<0 else 1. for r in rows])
    for mask in (positive,~positive):
        weights[mask]*=.5/weights[mask].sum()

    def objective(a):
        z=x@a
        penalty=np.r_[a[:-1],0.]
        return (np.dot(weights,np.logaddexp(0.,z)-positive*z)+l2*np.dot(penalty,penalty),
                x.T@(weights*(expit(z)-positive))+2*l2*penalty)

    result=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
    if not result.success:
        raise ValueError(result.message)
    return dict(mean=mean.tolist(),scale=scale.tolist(),coefficients=result.x.tolist(),feature_names=list(FEATURES))


def sample_counts(dataset,model=None):
    rows=[row for scene in dataset.values() for row in scene['samples']]
    summary=dict(samples=len(rows),positive=sum(r['beneficial'] for r in rows),
                 harmful=sum(r['correct_gain']<0 or r['wrong_reduction']<0 for r in rows),
                 kinds=dict(Counter(r['kind'] for r in rows)),
                 scenes_with_positive=sum(any(r['beneficial'] for r in scene['samples']) for scene in dataset.values()))
    if model is not None and rows:
        v=score(model,np.asarray([row['x'] for row in rows]))
        flagged=[row for row,prediction in zip(rows,v,strict=True) if prediction>model['cutoff']]
        summary.update(flagged=len(flagged),flagged_positive=sum(row['beneficial'] for row in flagged),
                       flagged_harmful=sum(r['correct_gain']<0 or r['wrong_reduction']<0 for r in flagged))
    return summary


def train(args,plan,digest):
    training=load(args.root,'fit',plan,digest)
    validation=load(args.root,'validation',plan,digest)
    model=fit([row for scene in training.values() for row in scene['samples']],plan['l2'])
    cutoffs={}
    for name,scene in validation.items():
        ref=read(args.root/'stream-census-development'/(name+'.json'))
        assert ref['protocol_sha256']==plan['census_sha256']
        denominator=ref['variants']['census']['operations']['eligible_correct']
        budget=int(denominator*plan['false_budget_per_10000']/10000)
        protected=[r['x'] for r in scene['census'] if r['kind'] in ('correct','clutter','absent','unsupported')]
        if protected:
            cutoffs[name]=scene_cutoff(score(model,np.asarray(protected)),budget)
    model.update(cutoff=max(cutoffs.values()),protocol_sha256=digest,fit_scenes=plan['fit']['scenes'],
                 validation_scenes=plan['validation']['scenes'])
    args.policy.write_text(json.dumps(model,indent=2)+'\n',encoding='utf8')
    report=dict(protocol_sha256=digest,policy_sha256=sha(args.policy),training=sample_counts(training),
                validation=sample_counts(validation,model),scene_cutoffs=cutoffs)
    print(report,flush=True)
    return report


def work(args,plan,digest):
    results=load(args.root,'work',plan,digest)
    control, old = {}, {}
    previous=args.root/'visible-onset-hold-summary.json'
    assert sha(previous)==plan['previous_auto_sha256']
    old_policy=read(previous)['policy_sha256']
    for name,row in results.items():
        assert row['policy_sha256']==sha(args.policy)
        original=read(args.root/'stream-census-work'/(name+'.json'))
        assert original['protocol_sha256']==plan['census_sha256']
        previous_run=read(args.root/'visible-onset-hold-work'/(name+'.json'))
        assert previous_run['policy_sha256']==old_policy
        labels=read(args.root/'mix-attribution'/(name+'.json'))
        assert labels['protocol_sha256']==plan['mix_sha256']
        control[name]=dict(original['variants']['census'],continued=labels['rollups']['3']['decomposition']['continued'])
        old[name]=previous_run['variants']['automatic']
    all_names=plan['work']['scenes']
    summary={}
    for label,names in (('all16',all_names),('other14',[n for n in all_names if n not in plan['subset_excludes']])):
        reference={n:control[n] for n in names}
        cells={}
        for mode, batch in (('combined',control),('previous',old),('rollout',results)):
            subset={n:batch[n] for n in names}
            output=paired(names,reference,subset,plan['guards'])
            output['continued']=sum(row['continued'] for row in subset.values())
            output['short_ids']=sum(row['birth_audit']['extra_ids_1or2'] for row in subset.values())
            cells[mode]=output
        operations=cells['rollout']['operations']
        protection=sum(operations.get('hold_'+name,0) for name in ('correct','clutter','absent','unsupported'))
        allowance=int(cells['combined']['operations']['eligible_correct']*plan['false_budget_per_10000']/10000)
        cells['rollout']['checks'].update(continued=cells['rollout']['continued']<cells['combined']['continued'],
                                         budget=protection<=allowance, not_absent=operations.get('hold_absent',0)==0)
        cells['rollout']['budget']=dict(known_non_target=protection,allowance=allowance)
        print(label,{name:{k:row[k] for k in ('mixed','continued','metrics','checks','operations')} for name,row in cells.items()},flush=True)
        summary[label]=cells
    return dict(protocol_sha256=digest,policy_sha256=sha(args.policy),scopes=summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','protocol','policy','out'):
        p.add_argument(name,type=Path)
    p.add_argument('--work',action='store_true')
    args=p.parse_args()
    plan,digest=read(args.protocol),sha(args.protocol)
    result=work(args,plan,digest) if args.work else train(args,plan,digest)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
