"""Full causal calibration rollout of a fixed candidate against frozen v2."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import monotonic,perf_counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import Diagnostic,Discard,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--scenes',nargs='+')
    p.add_argument('--seconds',type=int,default=3500);p.add_argument('--preview',action='store_true');a=p.parse_args()
    import numpy as np
    import torch
    from b02.replay import open_scenes,_scans,replay_frames
    from b02.scorecard import build_scorecard,write_scorecard
    from experiments.v3_tracking.runtime import load_scorer, Pipeline
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'));scenes=a.scenes or plan['calibration']
    assert scenes and set(scenes)<=set(plan['calibration'])
    cache_manifest=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8'));assert cache_manifest['data_sha256']==sha(a.data)
    scorer,memory,metadata,step=load_scorer(a.checkpoint);torch.set_num_threads(2)
    assert metadata['pairs']['fit']==plan['fit64'] and metadata['pairs']['calibration']==plan['calibration']
    reference=json.loads(a.reference.read_text(encoding='utf-8'))
    a.out.mkdir(parents=True,exist_ok=False);start=monotonic();diagnostics={}
    summary={'kind':'local_diagnostics','config':{'checkpoint_sha256':sha(a.checkpoint),'memory':memory,'labels':metadata.get('labels','column ownership'),
        'seed':metadata['seed'],'step':step,'strategy':'rank_only','timing':'association only; frozen point scores cached'},'scenes':{}}
    save(a.out/'manifest.json',{'training':metadata,'checkpoint_sha256':sha(a.checkpoint),'scenes':scenes,'source_sha256':sha(__file__)})
    with open_scenes(a.data) as (_,opener):
        for scene in scenes:
            with opener(scene) as file:frames=list(_scans(file))
            with np.load(a.cache/'calibration'/f'{scene}.npz') as data:scores_all=data['point_scores'];offsets=data['point_offsets']
            tracker=Pipeline(scorer,memory).physical;diag=Diagnostic();predictions=[];times=[]
            for i,(circle,points,labels) in enumerate(frames):
                if monotonic()-start>a.seconds:raise TimeoutError('Calibration rollout budget')
                before=set(tracker.state.get('tracks',{}));scores=scores_all[offsets[i]:offsets[i+1]].tolist();tick=perf_counter()
                outputs=tracker(points,scores);times.append((perf_counter()-tick)*1000);predictions.append(outputs)
                diag.observe(points,labels,outputs,tracker,before,circle)
            stream=(a.out/f'{scene}.jsonl').open('w',encoding='utf-8') if a.preview and scene=='train_032' else Discard()
            try:
                iterator=iter(predictions);stats=replay_frames(frames,stream,lambda p,s:next(iterator))
            finally:
                if hasattr(stream,'close'):stream.close()
            times.sort();stats.update(p95_handler_ms=times[(95*len(times)-1)//100],max_handler_ms=times[-1])
            summary['scenes'][scene]=stats;diagnostics[scene]=diag.result();save(a.out/'summary.json',summary);save(a.out/'diagnostics.json',diagnostics)
            print(json.dumps({'scene':scene,'seconds':round(monotonic()-start,2),'switches':diag.counts.get('switches',0)}),flush=True)
    card=build_scorecard(summary,a.out.name,reference,'frozen-v2');write_scorecard(a.out,card);counts=Counter()
    for d in diagnostics.values():counts.update(d['counts'])
    save(a.out/'report.json',{'complete':True,'metrics':card['overall']['values'],'baseline':card['overall']['baseline'],'delta':card['overall']['delta'],
        'identity':dict(counts),'scenes':scenes,'wall_seconds':monotonic()-start,'confirmation_used':False,'holdout_used':False})


if __name__=='__main__':main()
