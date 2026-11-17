"""Keep model verification separate from truth-assisted diagnostic references."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from b02.scorecard import build_scorecard,write_scorecard
from experiments.robust_validation.frozen import MODES
from experiments.robust_validation.uncertainty import sufficient,metrics,paired_bootstrap
from experiments.v2_association.run_stage1 import save


def diagnostic_report(root):
    paths=sorted(root.glob('diagnosis-*/train_*.json'));assert len(paths)==8
    variants={};misses=Counter();fragments=Counter();short=Counter();all_causes=Counter();operations={};shadow={}
    for path in paths:
        row=json.loads(path.read_text(encoding='utf-8'));assert row['read_only_shadow_exact']
        for name,value in row['variants'].items():variants.setdefault(name,{})[row['scene']]=value
        misses.update(row['misses']);fragments.update(row['fragment_counts']);short.update(row['short_first_causes']);all_causes.update(row['all_first_causes'])
        for name,value in row['operations'].items():
            operations.setdefault(name,Counter()).update({k:v for k,v in value.items() if isinstance(v,int) and not isinstance(v,bool)})
        for name,value in row['shadow'].items():
            shadow.setdefault(name,Counter()).update({k:v for k,v in value.items() if isinstance(v,int)})
    reference={'kind':'local_diagnostics','scenes':variants['frozen_candidate']};results={}
    for name,scenes in variants.items():
        summary={'kind':'local_diagnostics','scenes':scenes,'config':{'diagnostic':name,'deployable':name=='frozen_candidate'}}
        folder=root/'diagnosis'/name;folder.mkdir(parents=True,exist_ok=True)
        save(folder/'summary.json',summary)
        card=build_scorecard(summary,name,reference,'frozen_candidate');write_scorecard(folder,card)
        s=shadow[name]
        results[name]={'metrics':card['overall']['values'],'delta':card['overall']['delta'],
                       'strict_one_to_one_recall':100*s['one_to_one_correct_points']/s['truth_points'],
                       'mixed_minority_points':s['mixed_minority_points']}
    result={'scenes':sorted(reference['scenes']),'variants':results,'misses':dict(misses),'fragment_counts':dict(fragments),
            'short_first_causes':dict(short),'all_first_causes':dict(all_causes),'operations':{k:dict(v) for k,v in operations.items()},
            'limitations':['Truth-assisted references are not deployable scores.',
                           'Candidate repairs depend on the frozen current state trajectory and retrospective main IDs; they are not global upper bounds.',
                           'Perfect identity holds current selected target points and first-hit times fixed; it is not an architecture ceiling.']}
    save(root/'diagnostic-report.json',result)
    print('diagnostics',json.dumps({'variants':{k:{'total':v['metrics']['weighted_total'],'frag':v['metrics']['fragmentation_score'],'continuous':v['metrics']['continuous_score']} for k,v in results.items()},'fragments':dict(fragments),'misses':dict(misses)},ensure_ascii=False))
    return result


def validation_report(root,freeze):
    paths=sorted(root.glob('verify-*/train_*.json'))
    assert len(paths)==len(freeze['work_validation16'])==16
    scenes={mode:{} for mode in MODES};identities={mode:Counter() for mode in MODES};shadows={mode:Counter() for mode in MODES}
    audits={mode:{} for mode in MODES};by_name={}
    for path in paths:
        row=json.loads(path.read_text(encoding='utf-8'));scene=row['scene']
        assert scene in freeze['work_validation16'] and scene not in by_name;by_name[scene]=row
        for mode in MODES:
            scenes[mode][scene]=row['variants'][mode];identities[mode].update(row['identity'][mode]);audits[mode][scene]=row['birth_audit'][mode]
            shadows[mode].update({k:v for k,v in row['shadow'][mode].items() if isinstance(v,int)})
    names=freeze['work_validation16'];reference={'kind':'local_diagnostics','scenes':scenes['wide']};reports={}
    arrays={mode:np.stack([sufficient(scenes[mode][scene],by_name[scene]['identity'][mode]) for scene in names]) for mode in MODES}
    for mode in MODES:
        folder=root/'validation'/mode;folder.mkdir(parents=True,exist_ok=True)
        summary={'kind':'local_diagnostics','scenes':{name:scenes[mode][name] for name in names},'config':{'frozen_mode':mode}}
        save(folder/'summary.json',summary);card=build_scorecard(summary,mode,reference,'wide');write_scorecard(folder,card)
        computed=metrics(arrays[mode].sum(0))
        for key in computed:
            if key in card['overall']['values']:assert abs(float(computed[key])-card['overall']['values'][key])<1e-9,(mode,key)
        for shard in root.glob('verify-*'):
            trace=shard/mode/f'{names[0]}.jsonl'
            if trace.exists():(folder/trace.name).write_bytes(trace.read_bytes())
        short=sum(row['extra_ids_1or2'] for row in audits[mode].values());delays=Counter()
        for name in names:
            prior=audits['wide'][name]['first_truth_delays'];current=audits[mode][name]['first_truth_delays']
            delta=[current[t]-d for t,d in prior.items() if t in current]
            delays.update({'later':sum(d>0 for d in delta),'later_over_one':sum(d>1 for d in delta),'earlier':sum(d<0 for d in delta),'lost':len(set(prior)-set(current))})
        s=shadows[mode]
        reports[mode]={'metrics':card['overall']['values'],'delta':card['overall']['delta'],'identity':dict(identities[mode]),
                       'mixed_rate_percent':float(computed['mixed_rate_percent']),'strict_one_to_one_recall':100*s['one_to_one_correct_points']/s['truth_points'],
                       'short_ids_1or2':short,'first_hit_changes':dict(delays),'scenes':{name:scope['delta'] for name,scope in card['scenes'].items()}}
    boot=freeze['bootstrap'];ci=paired_bootstrap(arrays['wide'],arrays['candidate'],boot['resamples'],boot['seed'])
    base=reports['wide'];candidate=reports['candidate'];b=base['metrics'];c=candidate['metrics'];rule=freeze['adoption']
    ratio=(candidate['mixed_rate_percent']/base['mixed_rate_percent'] if base['mixed_rate_percent'] else
           1. if not candidate['mixed_rate_percent'] else None)
    guards={'point_recall':c['point_recall']>=b['point_recall']-rule['point_recall_drop_pp_max'],
            'track_recall':c['track_recall']>=b['track_recall']-rule['track_recall_drop_pp_max'],
            'continuous':c['continuous_score']>=b['continuous_score']-rule['continuous_drop_max'],
            'timeliness':c['timeliness_score']>=b['timeliness_score']-rule['timeliness_drop_max'],
            'clutter':c['clutter_outputs']<=rule['clutter_ratio_max']*b['clutter_outputs'],
            'worst_scene_total':min(row['weighted_total'] for row in candidate['scenes'].values())>=-rule['any_scene_total_drop_max'],
            'worst_scene_point_recall':min(row['point_recall'] for row in candidate['scenes'].values())>=-rule['any_scene_point_drop_pp_max']}
    primary={'mixed_rate_reduction':ratio is not None and ratio<=rule['mixed_rate_ratio_max'],
             'mixed_rate_ci':ci['mixed_rate_percent']['upper']<=rule['mixed_rate_difference_ci_upper_max'],
             'weighted_noninferiority_ci':ci['weighted_total']['lower']>=rule['weighted_delta_ci_lower_min']}
    if all(guards.values()) and all(primary.values()):decision='support'
    elif not all(guards.values()) or ci['mixed_rate_percent']['lower']>0 or ci['weighted_total']['upper']<rule['weighted_delta_ci_lower_min']:decision='regression'
    else:decision='insufficient_evidence'
    result={'format':'frozen-work-validation16-v1','models':reports,'paired_scene_bootstrap_95':ci,'mixed_rate_ratio':ratio,
            'guards':guards,'primary_checks':primary,'decision':decision,'protocol_frozen_before_results':True,
            'assumption':'Scene-paired resampling; no guarantee that simulated scenes are independent families.',
            'new_validation_role':'Evaluated work validation, not a fresh blind set for future tuning.',
            'sealed_test_used':False,'no_training_or_threshold_changes':True}
    save(root/'validation-report.json',result)
    with (root/'validation-comparison.csv').open('w',encoding='utf-8-sig',newline='') as target:
        writer=csv.writer(target);writer.writerow(['mode','local_total','fragmentation','continuous','point_recall','track_recall','timeliness','clutter','mixed_minority','mixed_rate_percent','strict_one_to_one_recall','short_ids_1or2'])
        for mode,row in reports.items():
            m=row['metrics'];writer.writerow([mode,m['weighted_total'],m['fragmentation_score'],m['continuous_score'],m['point_recall'],m['track_recall'],m['timeliness_score'],m['clutter_outputs'],row['identity']['mixed_minority_truth_points'],row['mixed_rate_percent'],row['strict_one_to_one_recall'],row['short_ids_1or2']])
    print('new work validation',json.dumps({'decision':decision,'guards':guards,'primary':primary,'models':{k:{'total':v['metrics']['weighted_total'],'frag':v['metrics']['fragmentation_score'],'mixed':v['identity']['mixed_minority_truth_points']} for k,v in reports.items()},'ci':ci},ensure_ascii=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--diagnostic-only',action='store_true')
    args=parser.parse_args();freeze=json.loads((ROOT/'experiments/robust_validation/freeze.json').read_text(encoding='utf-8'))
    diagnostic_report(args.root)
    if not args.diagnostic_only:validation_report(args.root,freeze)


if __name__=='__main__':main()
