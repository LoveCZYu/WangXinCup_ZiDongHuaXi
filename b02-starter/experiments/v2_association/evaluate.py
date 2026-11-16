"""Causal calibration rollout using frozen point scores; never reads holdout truth."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import monotonic,perf_counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from experiments.v2_association.run_stage1 import Diagnostic,Discard,save,sha


def evaluate(model,data,pairs,out,scenes,*,scale=1.,strategy='residual',reference=None,preview=False):
    import numpy as np
    from b02.replay import open_scenes,_scans,replay_frames
    from b02.scorecard import build_scorecard,write_scorecard
    from experiments.v2_association.model import Scorer
    from experiments.v2_association.tracker import Tracker
    manifest=json.loads((pairs/'manifest.json').read_text(encoding='utf-8'))
    assert scenes and set(scenes)<=set(manifest['calibration'])
    assert not set(scenes)&set(manifest['fit'])
    assert manifest['data_sha256']==sha(data)
    out.mkdir(parents=True,exist_ok=False)
    start=monotonic();summary={'kind':'local_diagnostics','config':{'association_scale':scale,'strategy':strategy,
        'timing':'cached frozen point scores; association only, not end-to-end'},'scenes':{}}
    diagnostics={};scorer=Scorer(model,'cpu',scale,strategy)
    with open_scenes(data) as (_,opener):
        for scene in scenes:
            with opener(scene) as file:frames=list(_scans(file))
            with np.load(pairs/'calibration'/f'{scene}.npz') as cache:
                scores=cache['point_scores'];offsets=cache['point_offsets']
            assert len(offsets)==len(frames)+1 and offsets[0]==0 and offsets[-1]==len(scores)
            tracker=Tracker('v1',scorer=scorer,capture=True,contextual=manifest.get('context',False));diag=Diagnostic();outputs=[];times=[]
            for i,(circle,points,labels) in enumerate(frames):
                assert offsets[i+1]-offsets[i]==len(points)
                before=set(tracker.state.get('tracks',{}));tick=perf_counter()
                predictions=tracker(points,scores[offsets[i]:offsets[i+1]].tolist())
                times.append((perf_counter()-tick)*1000);outputs.append(predictions)
                diag.observe(points,labels,predictions,tracker,before,circle)
            stream=(out/f'{scene}.jsonl').open('w',encoding='utf-8') if preview and scene=='train_032' else Discard()
            try:
                iterator=iter(outputs);stats=replay_frames(frames,stream,lambda points,state:next(iterator))
            finally:
                if hasattr(stream,'close'):stream.close()
            times.sort();stats.update(p95_handler_ms=times[(95*len(times)-1)//100],max_handler_ms=times[-1])
            summary['scenes'][scene]=stats;diagnostics[scene]=diag.result()
            save(out/'summary.json',summary);save(out/'diagnostics.json',diagnostics)
            print(json.dumps({'evaluation':out.name,'scene':scene,'seconds':round(monotonic()-start,2),
                              'switches':diag.counts.get('switches',0)}),flush=True)
    card=build_scorecard(summary,out.name,reference,'frozen-v1' if reference else None)
    write_scorecard(out,card)
    identity=Counter()
    for item in diagnostics.values():identity.update(item['counts'])
    result={'complete':True,'metrics':card['overall']['values'],'identity':dict(identity),
            'scenes':scenes,'scale':scale,'strategy':strategy,'wall_seconds':monotonic()-start,
            'timing':'association only; frozen point model scores cached',
            'frozen_v1_promoted':False}
    if reference:
        ref_subset={**reference,'scenes':{s:reference['scenes'][s] for s in scenes}}
        baseline=build_scorecard(ref_subset,'frozen-v1')['overall']['values']
        result['baseline']=baseline
        result['delta']={k:v-baseline[k] for k,v in result['metrics'].items() if v is not None and baseline[k] is not None}
    save(out/'report.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--reference',type=Path);p.add_argument('--scenes',nargs='+')
    p.add_argument('--scale',type=float,default=1.);p.add_argument('--preview',action='store_true')
    p.add_argument('--strategy',choices=('residual','rank_only','trusted_rank'),default='residual')
    a=p.parse_args()
    import torch
    from experiments.v2_association.model import AssociationModel
    manifest=json.loads((a.pairs/'manifest.json').read_text(encoding='utf-8'))
    model=AssociationModel(manifest['track_dim'],manifest['edge_dim'])
    if a.checkpoint:
        checkpoint=torch.load(a.checkpoint,map_location='cpu')
        assert checkpoint['metadata']['feature_version']==manifest['feature_version']
        assert checkpoint['metadata']['pairs_manifest_sha256']==sha(a.pairs/'manifest.json')
        model.load_state_dict(checkpoint['model'])
    reference=json.loads(a.reference.read_text(encoding='utf-8')) if a.reference else None
    result=evaluate(model,a.data,a.pairs,a.out,a.scenes or manifest['calibration'],
                    scale=a.scale,strategy=a.strategy,reference=reference,preview=a.preview)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
