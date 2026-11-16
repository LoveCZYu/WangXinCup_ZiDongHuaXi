"""Merge disjoint scene shards, preserving the original frozen-candidate scope."""
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
    p.add_argument('--shards',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frozen',type=Path,required=True);p.add_argument('--partial-reference',type=Path)
    a=p.parse_args()
    from b02.scorecard import build_scorecard,write_scorecard
    freeze=json.loads(a.frozen.read_text(encoding='utf-8'));assert freeze['checks_passed']
    names=['v1',*[str(m['seed']) for m in freeze['models']]];expected=freeze['dev_scenes']
    summaries={n:None for n in names};diagnostics={n:{} for n in names};seen=set();shard_ids=set();wall=[]
    for folder in a.shards:
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'));report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        assert manifest['freeze_sha256']==sha(a.frozen) and report['complete'] and not report['holdout_used']
        index,count=manifest['shard_index'],manifest['shard_count']
        assert count==len(a.shards) and index not in shard_ids;shard_ids.add(index)
        assert manifest['scenes']==expected[index::count] and not seen&set(manifest['scenes'])
        seen.update(manifest['scenes']);wall.append(report['wall_seconds'])
        for n in names:
            s=json.loads((folder/n/'summary.json').read_text(encoding='utf-8'));assert set(s['scenes'])==set(manifest['scenes'])
            if summaries[n] is None:summaries[n]={**s,'scenes':{}}
            summaries[n]['scenes'].update(s['scenes'])
            diagnostics[n].update(json.loads((folder/n/'diagnostics.json').read_text(encoding='utf-8')))
    assert seen==set(expected) and shard_ids==set(range(len(a.shards)))
    a.out.mkdir(parents=True,exist_ok=False);results={};reference_checks={}
    for n in names:
        summary=summaries[n];summary['scenes']={s:summary['scenes'][s] for s in expected}
        folder=a.out/n;folder.mkdir()
        if a.partial_reference and (a.partial_reference/n/'summary.json').exists():
            old=json.loads((a.partial_reference/n/'summary.json').read_text(encoding='utf-8'))
            for scene,stats in old['scenes'].items():
                fresh=summary['scenes'][scene]
                equal={k:v for k,v in stats.items() if not k.endswith('handler_ms')}=={k:v for k,v in fresh.items() if not k.endswith('handler_ms')}
                reference_checks[n+'/'+scene]=equal;assert equal,(n,scene)
        save(folder/'summary.json',summary);save(folder/'diagnostics.json',diagnostics[n])
        card=build_scorecard(summary,n,summaries['v1'] if n!='v1' else None,'v1' if n!='v1' else None)
        write_scorecard(folder,card);counts=Counter()
        for item in diagnostics[n].values():counts.update(item['counts'])
        results[n]={'metrics':card['overall']['values'],'identity':dict(counts)}
        for shard in a.shards:
            trace=shard/n/'train_015.jsonl'
            if trace.exists():shutil.copyfile(trace,folder/trace.name)
    report={'complete':True,'holdout_used':False,'frozen_v1_promoted':False,'scenes':expected,
            'results':results,'partial_reference_equal':reference_checks,'shard_wall_seconds':wall}
    save(a.out/'manifest.json',{'freeze':freeze,'freeze_sha256':sha(a.frozen),
         'shards':[str(s) for s in a.shards],'merge_source_sha256':sha(__file__)})
    save(a.out/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
