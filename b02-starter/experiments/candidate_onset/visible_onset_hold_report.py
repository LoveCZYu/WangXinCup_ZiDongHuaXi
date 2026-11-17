"""Calibrate the first-error screen and compare complete tracking runs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from experiments.candidate_onset.joint_cost_report import scene_cutoff
from experiments.candidate_onset.propagation import sha
from experiments.candidate_onset.propagation_work_summary import read
from experiments.candidate_onset.update_stream_report import TIMINGS, paired
from experiments.candidate_onset.visible_onset_hold import FEATURES, score

POSITIVE = ('visible_supported', 'visible_other_supported')


def fit(rows, regularization):
    candidates = [(name,r) for name, records in rows.items() for r in records if r['label'] != 'unresolved']
    grouped = Counter((name,r['batch']) for name,r in candidates if r['label'] in POSITIVE)
    if len(grouped) < 8:
        return dict(mean=[0.]*len(FEATURES), scale=[1.]*len(FEATURES),
                    coefficients=[-1.,1.,1.,-1.,*([0.]*(len(FEATURES)-3))],
                    method='fixed physical and affinity margins, sparse positives')
    matrix = np.asarray([r['x'] for _,r in candidates])
    positive = np.asarray([r['label'] in POSITIVE for _,r in candidates])
    mean, scale = matrix.mean(0), np.maximum(matrix.std(0), .01)
    design = np.column_stack(((matrix-mean)/scale, np.ones(len(matrix))))
    w = np.asarray([1/grouped[name,r['batch']] if flag else 1. for flag,(name,r) in zip(positive,candidates,strict=True)])
    w[positive] *= .5/w[positive].sum()
    w[~positive] *= .5/w[~positive].sum()

    def objective(coef):
        pred = design@coef
        penalty = np.r_[coef[:-1],0.]
        value = np.dot(w,np.logaddexp(0.,pred)-positive*pred)+regularization*np.dot(penalty,penalty)
        return value, design.T@(w*(expit(pred)-positive))+2*regularization*penalty

    result = minimize(objective,np.zeros(design.shape[1]),jac=True,method='L-BFGS-B')
    if not result.success:
        raise ValueError(result.message)
    return dict(mean=mean.tolist(), scale=scale.tolist(), coefficients=result.x.tolist(), method='balanced logistic')


def calibration(rows, denominators, plan):
    model = fit(rows,plan['regularization'])
    bounds = {}
    for name, cases in rows.items():
        x = np.asarray([r['x'] for r in cases if r['label'] not in (*POSITIVE,'unresolved')]).reshape(-1,len(FEATURES))
        allowance = int(denominators[name]*plan['false_budget_per_10000']/10000)
        bound = scene_cutoff(score(model,x),allowance)
        if bound is not None:
            bounds[name] = bound
    if not bounds:
        raise ValueError('No adjudicated calibration controls')
    return dict(model, cutoff=max(bounds.values()))


def count_flags(model, rows):
    scores = score(model,np.asarray([row['x'] for row in rows]).reshape(-1,len(FEATURES)))
    eligible = Counter(row['label'] for row in rows)
    flags = Counter(row['label'] for pred,row in zip(scores,rows,strict=True) if pred>model['cutoff'])
    return dict(eligible=dict(eligible), flags=dict(flags))


def development(args, plan):
    data, denominators = {}, {}
    for name in plan['development']['scenes']:
        row=read(args.root/'visible-onset-hold-development'/(name+'.json'))
        assert row['scene']==name and row['protocol_sha256']==sha(args.protocol)
        control=read(args.root/'stream-census-development'/(name+'.json'))
        assert control['protocol_sha256']==plan['census_protocol_sha256']
        actual, ref = row['variants']['census'], control['variants']['census']
        assert actual['identity']==ref['identity']
        assert all(v==ref['summary'][key] for key,v in actual['summary'].items() if key not in TIMINGS)
        data[name]=actual['rows']
        denominators[name]=ref['operations']['eligible_correct']
    model = dict(calibration(data,denominators,plan), features=list(FEATURES),
                 protocol_sha256=sha(args.protocol), fit_scenes=list(data))
    args.policy.write_text(json.dumps(model,indent=2)+'\n',encoding='utf8')
    folds = {}
    for name, cases in data.items():
        inner = calibration({k:v for k,v in data.items() if k!=name},denominators,plan)
        folds[name] = count_flags(inner,cases)
    totals = {key:Counter() for key in ('eligible','flags')}
    for row in folds.values():
        for key in totals:
            totals[key].update(row[key])
    report = dict(protocol_sha256=sha(args.protocol), policy_sha256=sha(args.policy), development_loso=totals, scenes=folds)
    print(model['method'], model['cutoff'], dict(totals),flush=True)
    return report


def work(args,plan):
    assert read(args.policy)['protocol_sha256']==sha(args.protocol)
    names=plan['work']['scenes']
    controls, methods = {}, {key:{} for key in ('automatic','oracle')}
    for name in names:
        actual=read(args.root/'visible-onset-hold-work'/(name+'.json'))
        assert actual['scene']==name and actual['protocol_sha256']==sha(args.protocol) and actual['policy_sha256']==sha(args.policy)
        control=read(args.root/'stream-census-work'/(name+'.json'))
        attribution=read(args.root/'mix-attribution'/(name+'.json'))
        assert control['protocol_sha256']==plan['census_protocol_sha256'] and attribution['protocol_sha256']==plan['mix_protocol_sha256']
        controls[name]=dict(control['variants']['census'],continued=attribution['rollups']['3']['decomposition']['continued'])
        for mode in methods:
            methods[mode][name]=actual['variants'][mode]
    arms = {}
    denom=sum(row['operations']['eligible_correct'] for row in controls.values())
    limit=int(denom*plan['false_budget_per_10000']/10000)
    short_gap=read(args.root/'mix-absence-cases.json')['cases']
    for mode, table in (('combined',controls),*methods.items()):
        result=paired(names,controls,table,plan['guards'])
        result['continued']=sum(row['continued'] for row in table.values())
        result['short_ids']=sum(row['birth_audit']['extra_ids_1or2'] for row in table.values())
        outcome=result['operations']
        incorrect=sum(outcome.get('hold_'+key,0) for key in ('correct','clutter','visible_unsupported','original_absent'))
        result['checks']['false_budget']=incorrect<=limit
        result['checks']['no_known_absent_hold']=outcome.get('hold_original_absent',0)==0
        result['budget']=dict(protected_holds=incorrect, allowed=limit, denominator=denom)
        result['short_gap_onset_actions']=sum(any(action['batch']==event['batch'] and action['circle']==event['onset']['circle']
                                                    for action in table[event['scene']].get('actions',())) for event in short_gap)
        result['checks']['no_previous_short_gap_onset_hold']=result['short_gap_onset_actions']==0
        result['checks']['continued_reduced']=result['continued']<sum(row['continued'] for row in controls.values())
        arms[mode]=result
    report=dict(protocol_sha256=sha(args.protocol),policy_sha256=sha(args.policy),arms=arms)
    for mode,result in arms.items():
        print(mode,{k:result[k] for k in ('mixed','continued','metrics','operations','checks','short_gap_onset_actions')},flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','protocol','policy','out'):
        p.add_argument(name,type=Path)
    p.add_argument('--work',action='store_true')
    args=p.parse_args()
    plan=read(args.protocol)
    result=work(args,plan) if args.work else development(args,plan)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
