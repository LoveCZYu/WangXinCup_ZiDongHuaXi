"""Strictly merge a single preregistered confirmation experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True);p.add_argument('--shards',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    from b02.scorecard import build_scorecard,write_scorecard
    from experiments.v3_tracking.freeze_candidate import feasibility
    freeze=json.loads(a.frozen.read_text(encoding='utf-8'));plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    names=freeze['confirmation_scenes'];summaries={n:None for n in ('v2','candidate')};diags={n:{} for n in summaries};seen=set();indices=set();times=[]
    for folder in a.shards:
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'));report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        assert manifest['freeze_sha256']==sha(a.frozen) and report['complete'] and not report['holdout_used']
        i,count=manifest['shard_index'],manifest['shard_count'];assert count==len(a.shards) and i not in indices;indices.add(i)
        assert manifest['scenes']==names[i::count] and not seen&set(manifest['scenes']);seen.update(manifest['scenes']);times.append(report['wall_seconds'])
        for n in summaries:
            summary=json.loads((folder/n/'summary.json').read_text(encoding='utf-8'));assert set(summary['scenes'])==set(manifest['scenes'])
            if summaries[n] is None:summaries[n]={**summary,'scenes':{}}
            summaries[n]['scenes'].update(summary['scenes']);diags[n].update(json.loads((folder/n/'diagnostics.json').read_text(encoding='utf-8')))
    assert seen==set(names) and indices==set(range(len(a.shards)))
    a.out.mkdir(parents=True,exist_ok=False);results={}
    for n in summaries:
        folder=a.out/n;folder.mkdir();summaries[n]['scenes']={scene:summaries[n]['scenes'][scene] for scene in names}
        save(folder/'summary.json',summaries[n]);save(folder/'diagnostics.json',diags[n])
        card=build_scorecard(summaries[n],n,summaries['v2'] if n!='v2' else None,'v2' if n!='v2' else None);write_scorecard(folder,card)
        counts=Counter()
        for d in diags[n].values():counts.update(d['counts'])
        results[n]={'metrics':card['overall']['values'],'identity':dict(counts)}
        for shard in a.shards:
            trace=shard/n/f'{names[0]}.jsonl'
            if trace.exists():shutil.copyfile(trace,folder/trace.name)
    checks=feasibility(results['candidate'],results['v2'],plan['calibration_feasible'])
    report={'complete':True,'results':results,'checks':checks,'confirmation_passed':all(checks.values()),
        'scenes':names,'confirmation_used':True,'selection_or_tuning_from_confirmation':False,'holdout_used':False,
        'shard_wall_seconds':times,'freeze_sha256':sha(a.frozen),'official_submission_ready':False}
    save(a.out/'manifest.json',{'freeze':freeze,'freeze_sha256':sha(a.frozen),'shards':[str(p) for p in a.shards]})
    save(a.out/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
