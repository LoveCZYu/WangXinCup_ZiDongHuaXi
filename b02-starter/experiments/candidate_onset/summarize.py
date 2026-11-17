"""Report all fixed-scene episode and training-rule audits separately."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--protocol',type=Path,required=True)
    args=p.parse_args();protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    names=protocol['training_sample4']+protocol['head_validation8']
    rows={}
    for path in sorted(args.root.glob('audit-*/train_*.json')):
        row=json.loads(path.read_text(encoding='utf-8'))
        assert row['protocol_sha256']==sha(args.protocol) and row['scene'] not in rows
        rows[row['scene']]=row
    assert set(rows)==set(names),'All fixed scenes must complete'
    result={}
    for role in ('training_sample4','head_validation8'):
        counts=Counter();eligible=defaultdict(Counter);prior_wrong=defaultdict(Counter);gaps=Counter();fixtures=[]
        for name,row in rows.items():
            if row['role']!=role:continue
            counts.update(row['counts']);fixtures.extend(row['fixtures'])
            for event in row['events']:
                reason=event['reason']
                eligible[reason]['eligible' if event['prior_sampling_rule_eligible'] else 'excluded']+=1
                recent=event['intervening_internal_updates']
                prior_wrong[reason]['other_target']+=recent.get('internal_other_target',0)>0
                prior_wrong[reason]['clutter']+=recent.get('internal_clutter',0)>0
                gaps['gap' if event['episode_started_with_observation_gap'] else 'no_gap']+=1
        result[role]=dict(counts=dict(counts),sampling_by_reason={k:dict(v) for k,v in eligible.items()},
            preceding_wrong_by_reason={k:dict(v) for k,v in prior_wrong.items()},first_visible_start=gaps,fixtures=fixtures)
    report=dict(complete=True,protocol_sha256=sha(args.protocol),groups=result,
        scenes={n:dict(counts=r['counts'],circles=r['circles'],fixtures=r['fixtures'],shadow=r['shadow_all_outputs_equal']) for n,r in rows.items()},
        new_training=False,independent_validation=False,
        unit='ID episodes; first visible failure once until recovery, not cumulative missed points or distinct physical causes')
    save(args.root/'audit-report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
