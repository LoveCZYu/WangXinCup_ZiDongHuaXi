"""Paired exposed-work16 summary and uncertainty; no winner selection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from b02.scorecard import build_scorecard,write_scorecard
from experiments.v2_association.run_stage1 import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--protocol',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.protocol.read_text(encoding='utf-8'));names=plan['work16'];rows={}
    for path in sorted(args.root.glob('work16-*/train_*.json')):
        row=json.loads(path.read_text(encoding='utf-8'))
        assert row['protocol_sha256']==sha(args.protocol) and row['scene'] not in rows
        rows[row['scene']]=row
    assert set(rows)==set(names)
    baseline={'scenes':{name:rows[name]['variants']['baseline'] for name in names}};result={}
    for mode in ('baseline','combined'):
        folder=args.root/'work16'/mode
        summary=dict(kind='local_diagnostics',config=dict(mode=mode,role='previously_exposed_work16'),scenes={n:rows[n]['variants'][mode] for n in names})
        save(folder/'summary.json',summary)
        card=build_scorecard(summary,mode,baseline,'baseline');write_scorecard(folder,card)
        identity,shadow,ops=Counter(),Counter(),Counter()
        for row in rows.values():
            identity.update(row['identity'][mode]);ops.update(row['operations'][mode])
            shadow.update({k:v for k,v in row['shadow'][mode].items() if isinstance(v,int)})
        result[mode]=dict(metrics=card['overall']['values'],delta=card['overall']['delta'],identity=dict(identity),
            short_ids_1or2=sum(row['birth_audit'][mode]['extra_ids_1or2'] for row in rows.values()),
            strict_one_to_one_recall=100*shadow['one_to_one_correct_points']/shadow['truth_points'],operations=dict(ops),
            scenes={n:c['delta'] for n,c in card['scenes'].items()})
        for shard in args.root.glob('work16-*'):
            path=shard/mode/f'{names[0]}.jsonl'
            if path.exists():shutil.copyfile(path,folder/path.name)
    a,b=result['baseline']['metrics'],result['combined']['metrics'];r=plan['guards']
    checks=dict(fragmentation=b['fragmentation_score']>=a['fragmentation_score']+r['fragmentation_gain_min'],
        weighted=b['weighted_total']>=a['weighted_total']+r['weighted_gain_min'],point=b['point_recall']>=a['point_recall']-r['point_drop_pp_max'],
        track=b['track_recall']>=a['track_recall']-r['track_drop_pp_max'],continuous=b['continuous_score']>=a['continuous_score']-r['continuous_drop_max'],
        timeliness=b['timeliness_score']>=a['timeliness_score']-r['timeliness_drop_max'],clutter=b['clutter_outputs']<=a['clutter_outputs']*r['clutter_ratio_max'],
        mixed=result['combined']['identity']['mixed_minority_truth_points']<=result['baseline']['identity']['mixed_minority_truth_points']*r['mixed_ratio_max'],
        strict_recall=result['combined']['strict_one_to_one_recall']>=result['baseline']['strict_one_to_one_recall']-r['strict_drop_pp_max'])
    rng=np.random.default_rng(plan['bootstrap']['seed'])
    draw=rng.integers(0,len(names),size=(plan['bootstrap']['resamples'],len(names)))
    intervals={}
    for metric in ('weighted_total','fragmentation_score','continuous_score','point_recall'):
        delta=np.asarray([result['combined']['scenes'][n][metric] for n in names])
        samples=delta[draw].mean(1)
        intervals[metric]=dict(scene_mean=float(delta.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),
            positive_scenes=int((delta>0).sum()),negative_scenes=int((delta<0).sum()),worst=float(delta.min()),best=float(delta.max()))
    checks['weighted_interval_positive']=intervals['weighted_total']['ci95'][0]>0
    report=dict(models=result,primary_checks=checks,primary_passed=all(checks.values()),intervals=intervals,
        protocol_sha256=sha(args.protocol),independent_validation=False,version_promoted=False,
        interval_scope='Paired-scene mean delta, not CI of pooled official score; scene independence not guaranteed.')
    save(args.root/'work16-comparison.json',report)
    print(json.dumps(dict(primary_passed=report['primary_passed'],checks=checks,intervals=intervals,
        models={k:dict(total=v['metrics']['weighted_total'],fragmentation=v['metrics']['fragmentation_score'],continuous=v['metrics']['continuous_score'],point=v['metrics']['point_recall'],mixed=v['identity']['mixed_minority_truth_points']) for k,v in result.items()}),indent=2))


if __name__=='__main__':main()
