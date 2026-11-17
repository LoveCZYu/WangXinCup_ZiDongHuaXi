"""Fixed primary comparison and all secondary physical-state controls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from b02.scorecard import build_scorecard,write_scorecard
from experiments.candidate_onset.mixed import MODES
from experiments.v2_association.run_stage1 import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--phase',choices=('training4','development8'),required=True)
    args=p.parse_args();protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    names=protocol[args.phase];rows={}
    for path in sorted(args.root.glob(f'mixed-{args.phase}-*/train_*.json')):
        r=json.loads(path.read_text(encoding='utf-8'))
        assert r['phase']==args.phase and r['protocol_sha256']==sha(args.protocol) and r['scene'] not in rows
        rows[r['scene']]=r
    assert set(rows)==set(names)
    baseline={'scenes':{name:rows[name]['variants']['baseline'] for name in names}}
    result={}
    for mode in MODES:
        folder=args.root/('mixed-'+args.phase)/mode
        summary={'kind':'local_diagnostics','config':dict(mode=mode,phase=args.phase),
            'scenes':{name:rows[name]['variants'][mode] for name in names}}
        save(folder/'summary.json',summary)
        card=build_scorecard(summary,mode,baseline,'baseline');write_scorecard(folder,card)
        identity,shadow,operations=Counter(),Counter(),Counter()
        for row in rows.values():
            identity.update(row['identity'][mode]);operations.update(row['operations'][mode])
            shadow.update({k:v for k,v in row['shadow'][mode].items() if isinstance(v,int)})
        result[mode]=dict(metrics=card['overall']['values'],delta=card['overall']['delta'],identity=dict(identity),
            short_ids_1or2=sum(row['birth_audit'][mode]['extra_ids_1or2'] for row in rows.values()),
            strict_one_to_one_recall=100*shadow['one_to_one_correct_points']/shadow['truth_points'],operations=dict(operations),
            scenes={name:scope['delta'] for name,scope in card['scenes'].items()})
        for shard in args.root.glob(f'mixed-{args.phase}-*'):
            trace=shard/mode/f'{names[0]}.jsonl'
            if trace.exists():shutil.copyfile(trace,folder/trace.name)
    a,b=result['baseline']['metrics'],result['combined']['metrics'];rule=protocol['guards']
    checks=dict(fragmentation=b['fragmentation_score']>=a['fragmentation_score']+rule['fragmentation_gain_min'],
        weighted=b['weighted_total']>=a['weighted_total']+rule['weighted_gain_min'],
        point=b['point_recall']>=a['point_recall']-rule['point_drop_pp_max'],
        track=b['track_recall']>=a['track_recall']-rule['track_drop_pp_max'],
        continuous=b['continuous_score']>=a['continuous_score']-rule['continuous_drop_max'],
        timeliness=b['timeliness_score']>=a['timeliness_score']-rule['timeliness_drop_max'],
        clutter=b['clutter_outputs']<=a['clutter_outputs']*rule['clutter_ratio_max'],
        mixed=result['combined']['identity']['mixed_minority_truth_points']<=result['baseline']['identity']['mixed_minority_truth_points']*rule['mixed_ratio_max'],
        strict_recall=result['combined']['strict_one_to_one_recall']>=result['baseline']['strict_one_to_one_recall']-rule['strict_drop_pp_max'])
    report=dict(phase=args.phase,models=result,primary_checks=checks,primary_passed=all(checks.values()),
        protocol_sha256=sha(args.protocol),independent_validation=False,version_promoted=False)
    save(args.root/f'mixed-{args.phase}-comparison.json',report)
    print(json.dumps(dict(primary_passed=report['primary_passed'],checks=checks,
        models={name:dict(total=r['metrics']['weighted_total'],fragmentation=r['metrics']['fragmentation_score'],
            continuous=r['metrics']['continuous_score'],point=r['metrics']['point_recall'],
            clutter=r['metrics']['clutter_outputs'],mixed=r['identity']['mixed_minority_truth_points'],short=r['short_ids_1or2']) for name,r in result.items()}),indent=2))


if __name__=='__main__':main()
