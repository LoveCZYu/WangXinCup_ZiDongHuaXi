"""Frozen-candidate development validation; point scores shared across trackers."""
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
    p.add_argument('--data',type=Path,required=True);p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=int,default=3500)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1)
    a=p.parse_args()
    import torch
    from b02.point_tracker import load_tracker
    from b02.replay import open_scenes,_scans,replay_frames
    from b02.scorecard import build_scorecard,write_scorecard
    from experiments.v2_association.model import AssociationModel,Scorer
    from experiments.v2_association.tracker import Tracker
    freeze=json.loads(a.frozen.read_text(encoding='utf-8'));assert freeze['checks_passed']
    assert sha(a.data)==freeze['data_sha256']
    for name,digest in freeze['runtime_feature_sha256'].items():assert sha(Path(__file__).parent/name)==digest
    assert sha(ROOT/'assets/point-scorer-8000.pt')==freeze['point_model_sha256']
    assert sha(ROOT/'profiles/current.json')==freeze['profile_sha256']
    pairs=Path(freeze['pairs']);assert sha(pairs/'manifest.json')==freeze['pairs_manifest_sha256']
    training=json.loads((pairs/'manifest.json').read_text(encoding='utf-8'))
    split=json.loads((ROOT/'splits/v1.json').read_text(encoding='utf-8'));public=json.loads((ROOT/'splits/a_board.json').read_text(encoding='utf-8'))
    scenes=freeze['dev_scenes'];assert scenes==split['dev'] and len(scenes)==30
    assert not set(scenes)&set(training['fit']+training['calibration']+split['holdout']+public['scenes'])
    assert 0<=a.shard_index<a.shard_count<=30
    scenes=scenes[a.shard_index::a.shard_count]
    scorers={}
    for item in freeze['models']:
        checkpoint=Path(item['checkpoint']);assert sha(checkpoint)==item['sha256']
        loaded=torch.load(checkpoint,map_location='cpu');metadata=loaded['metadata']
        assert loaded['step']==item['step'] and metadata['seed']==item['seed']
        assert metadata['pairs_manifest_sha256']==freeze['pairs_manifest_sha256']
        model=AssociationModel(training['track_dim'],training['edge_dim']);model.load_state_dict(loaded['model'])
        scorers[str(item['seed'])]=Scorer(model,'cpu',freeze['scale'],freeze['strategy'])
    handler,profile=load_tracker(device='cpu');a.out.mkdir(parents=True,exist_ok=False);start=monotonic()
    names=['v1',*scorers];summaries={n:{'kind':'local_diagnostics','config':{'frozen_sha256':sha(a.frozen),
        'tracker':n,'profile':profile,'scope':'30-file development; no tuning'},'scenes':{}} for n in names}
    diagnostics={n:{} for n in names};report={'complete':False,'holdout_used':False,'frozen_v1_promoted':False,
        'scenes':scenes,'shard_index':a.shard_index,'shard_count':a.shard_count}
    save(a.out/'manifest.json',{'format':'b02-v2-frozen-development-v1','freeze':freeze,'freeze_sha256':sha(a.frozen),
         'preview_scene':'train_015','scenes':scenes,'shard_index':a.shard_index,'shard_count':a.shard_count,
         'sources':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}})
    def budget():
        if monotonic()-start>a.seconds:raise TimeoutError('Development validation wall time limit')
    try:
        with open_scenes(a.data) as (_,opener):
            for scene in scenes:
                budget()
                with opener(scene) as file:frames=list(_scans(file))
                trackers={'v1':Tracker('v1',capture=True)}
                trackers.update({n:Tracker('v1',capture=True,contextual=True,scorer=s) for n,s in scorers.items()})
                history=deque(maxlen=5);diags={n:Diagnostic() for n in names};cache={n:[] for n in names};times={n:[] for n in names}
                for circle,points,labels in frames:
                    budget();tick=perf_counter()
                    with torch.no_grad():scores=handler.model(torch.as_tensor(handler.features(points,list(history)))).sigmoid().tolist()
                    feature_ms=(perf_counter()-tick)*1000;history.append(points)
                    for n,tracker in trackers.items():
                        before=set(tracker.state.get('tracks',{}));tick=perf_counter();output=tracker(points,scores)
                        times[n].append(feature_ms+(perf_counter()-tick)*1000);cache[n].append(output)
                        diags[n].observe(points,labels,output,tracker,before,circle)
                for n in names:
                    folder=a.out/n;folder.mkdir(exist_ok=True)
                    stream=(folder/f'{scene}.jsonl').open('w',encoding='utf-8') if scene=='train_015' else Discard()
                    try:
                        iterator=iter(cache[n]);stats=replay_frames(frames,stream,lambda p,s:next(iterator))
                    finally:
                        if hasattr(stream,'close'):stream.close()
                    measured=sorted(times[n]);stats.update(p95_handler_ms=measured[(95*len(measured)-1)//100],max_handler_ms=measured[-1])
                    summaries[n]['scenes'][scene]=stats;diagnostics[n][scene]=diags[n].result()
                    save(folder/'summary.json',summaries[n]);save(folder/'diagnostics.json',diagnostics[n])
                print(json.dumps({'scene':scene,'seconds':round(monotonic()-start,2),
                    'switches':{n:diags[n].counts.get('switches',0) for n in names}}),flush=True)
        results={}
        for n in names:
            card=build_scorecard(summaries[n],n,summaries['v1'] if n!='v1' else None,'v1' if n!='v1' else None)
            write_scorecard(a.out/n,card);counts=Counter()
            for d in diagnostics[n].values():counts.update(d['counts'])
            results[n]={'metrics':card['overall']['values'],'identity':dict(counts)}
        report.update(complete=True,results=results)
    finally:
        report['wall_seconds']=monotonic()-start;save(a.out/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
