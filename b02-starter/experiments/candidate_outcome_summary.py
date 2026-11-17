"""Summarize fixed training4 episode consequences, not cumulative loss counts."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import hashlib


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--protocol',type=Path,required=True);args=p.parse_args()
    plan=json.loads(args.protocol.read_text(encoding='utf-8'))
    digest=hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    by_reason=defaultdict(Counter);scenes={};totals=Counter()
    paths=list(args.root.glob('outcome-training4-*/train_*.json'))
    assert len(paths)==len(plan['training_sample4'])
    for path in sorted(paths):
        row=json.loads(path.read_text(encoding='utf-8'))
        assert row['protocol_sha256']==digest and row['scene'] not in scenes
        scenes[row['scene']]=row['counts'];totals.update(row['counts'])
        for e in row['events']:
            c=by_reason[e['reason']];c['first_visible']+=1
            c['any_other_id']+=e['any_other_id_before_recovery']
            c['replacement_three_plus']+=e['replacement_three_plus']
            c['recovered_same_id']+=e['recovered']
            c['recovered_without_other_id']+=e['recovered'] and not e['any_other_id_before_recovery']
            c['unrecovered_with_later_truth']+=not e['recovered'] and e['had_later_truth_observation']
            c['fragment_event_sampling_eligible']+=e['any_other_id_before_recovery'] and e['prior_sampling_rule_eligible']
    assert set(scenes)==set(plan['training_sample4'])
    result=dict(complete=True,protocol_sha256=digest,totals=dict(totals),by_reason={k:dict(v) for k,v in by_reason.items()},scenes=scenes,
        unit='ID episodes; other ID appears before original ID recovery. Three-plus is descriptive stability, not an official matching threshold.',
        scope='same four actual training scenes, externally label-dependent future outcomes; no labels/future enter inference',
        new_training=False,independent_validation=False)
    (args.root/'outcome-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'totals':dict(totals),'by_reason':result['by_reason']},indent=2))


if __name__=='__main__':main()
