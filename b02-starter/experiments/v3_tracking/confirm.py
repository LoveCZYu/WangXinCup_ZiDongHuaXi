"""One fixed candidate versus frozen v2 on reserved confirmation scenes only."""
import argparse
from collections import Counter,deque
import json
from pathlib import Path
import sys
from time import monotonic,perf_counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import Diagnostic,Discard,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--shard-count',type=int,default=1);p.add_argument('--seconds',type=int,default=3500);a=p.parse_args()
    import torch
    from b02.replay import open_scenes,_scans,replay_frames
    from b02.scorecard import build_scorecard,write_scorecard
    from experiments.v2_association.candidate import load_candidate
    from experiments.v3_tracking.runtime import load_scorer,Pipeline
    freeze=json.loads(a.frozen.read_text(encoding='utf-8'));assert freeze['checks_passed'] and not freeze['confirmation_used']
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    assert freeze['split_sha256']==sha(ROOT/'splits/v3_tracking.json')
    assert freeze['confirmation_scenes']==plan['confirmation16'] and len(plan['confirmation16'])==16
    for name,digest in freeze['runtime_sha256'].items():assert sha(Path(__file__).parent/name)==digest
    for path,key in ((ROOT/'profiles/current.json','point_profile_sha256'),(ROOT/'assets/point-scorer-8000.pt','point_weights_sha256'),
                     (ROOT/'profiles/v2-association-candidate.json','baseline_profile_sha256'),(ROOT/'assets/v2-association-candidate.pt','baseline_head_sha256')):assert sha(path)==freeze[key]
    selected=next(x for x in freeze['models'] if x['seed']==freeze['selected_seed'])
    scorer,memory,meta,step=load_scorer(selected['checkpoint'],selected['checkpoint_sha256'])
    assert selected['seed']==plan['output_seed'] and step==selected['step'] and memory==selected['memory']
    assert not set(meta['pairs']['fit'])&set(plan['confirmation16'])
    baseline,base_spec=load_candidate(ROOT/'profiles/v2-association-candidate.json','cpu');torch.set_num_threads(2)
    assert 0<=a.shard_index<a.shard_count<=16
    names=plan['confirmation16'][a.shard_index::a.shard_count]
    a.out.mkdir(parents=True,exist_ok=False);start=monotonic();summaries={n:{'kind':'local_diagnostics',
        'config':{'confirmation':True,'freeze_sha256':sha(a.frozen),'variant':n,'identity':freeze['identity']},'scenes':{}} for n in ('v2','candidate')}
    diagnostics={n:{} for n in summaries};report={'complete':False,'holdout_used':False,'confirmation_used':True,'scenes':names,
        'shard_index':a.shard_index,'shard_count':a.shard_count}
    save(a.out/'manifest.json',{'frozen':freeze,'freeze_sha256':sha(a.frozen),'data_sha256':sha(a.data),'scenes':names,
        'shard_index':a.shard_index,'shard_count':a.shard_count,'preview_scene':plan['confirmation16'][0],
        'selection_or_tuning_from_confirmation':False})
    try:
        with open_scenes(a.data) as (available,opener):
            assert set(available)==set(plan['confirmation16'])
            for scene in names:
                with opener(scene) as file:frames=list(_scans(file))
                pipelines={'v2':Pipeline(baseline.scorer),'candidate':Pipeline(scorer,memory,freeze['identity'])}
                history=deque(maxlen=5);diags={n:Diagnostic() for n in pipelines};cache={n:[] for n in pipelines};times={n:[] for n in pipelines}
                for circle,points,labels in frames:
                    if monotonic()-start>a.seconds:raise TimeoutError('Confirmation validation budget')
                    tick=perf_counter()
                    with torch.no_grad():scores=baseline.point.model(torch.as_tensor(baseline.point.features(points,list(history)))).sigmoid().tolist()
                    feature_ms=(perf_counter()-tick)*1000;history.append(points)
                    for n,pipeline in pipelines.items():
                        before=set(pipeline.engine.state.get('tracks',{}));tick=perf_counter();outputs=pipeline(points,scores)
                        times[n].append(feature_ms+(perf_counter()-tick)*1000);cache[n].append(outputs)
                        diags[n].observe(points,labels,outputs,pipeline.engine,before,circle)
                for n in pipelines:
                    folder=a.out/n;folder.mkdir(exist_ok=True)
                    stream=(folder/f'{scene}.jsonl').open('w',encoding='utf-8') if scene==plan['confirmation16'][0] else Discard()
                    try:
                        iterator=iter(cache[n]);stats=replay_frames(frames,stream,lambda p,s:next(iterator))
                    finally:
                        if hasattr(stream,'close'):stream.close()
                    measured=sorted(times[n]);stats.update(p95_handler_ms=measured[(95*len(measured)-1)//100],max_handler_ms=measured[-1])
                    summaries[n]['scenes'][scene]=stats;diagnostics[n][scene]=diags[n].result()
                    save(folder/'summary.json',summaries[n]);save(folder/'diagnostics.json',diagnostics[n])
                print(json.dumps({'scene':scene,'seconds':round(monotonic()-start,2),
                    'switches':{n:diags[n].counts.get('switches',0) for n in pipelines}}),flush=True)
        results={}
        for n in pipelines:
            card=build_scorecard(summaries[n],n,summaries['v2'] if n!='v2' else None,'v2' if n!='v2' else None)
            write_scorecard(a.out/n,card);counts=Counter()
            for d in diagnostics[n].values():counts.update(d['counts'])
            results[n]={'metrics':card['overall']['values'],'identity':dict(counts)}
        report.update(complete=True,results=results)
    finally:
        report['wall_seconds']=monotonic()-start;save(a.out/'report.json',report)


if __name__=='__main__':main()
