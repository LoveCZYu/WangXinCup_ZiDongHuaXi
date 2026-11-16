"""Compare ID management alone while strictly preserving every selected point."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import monotonic

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import Diagnostic,Discard,save,sha
from experiments.v3_tracking.diagnose import reference_equal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--scenes',nargs='+');p.add_argument('--checkpoint',type=Path)
    p.add_argument('--seconds',type=int,default=3500);a=p.parse_args()
    import numpy as np
    import torch
    from b02.replay import open_scenes,_scans,replay_frames
    from b02.scorecard import build_scorecard,write_scorecard
    from experiments.v2_association.candidate import load_candidate
    from experiments.v2_association.model import AssociationModel,Scorer
    from experiments.v3_tracking.tracker import Tracker
    from experiments.v3_tracking.identity import IdentityManager
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'));scenes=a.scenes or plan['calibration']
    assert scenes and set(scenes)<=set(plan['calibration'])
    cache_manifest=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8'));assert cache_manifest['data_sha256']==sha(a.data)
    reference=json.loads(a.reference.read_text(encoding='utf-8'))
    if a.checkpoint:
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);meta=ck['metadata']
        if meta['format']=='b02-v3-association-training-v1':
            for name,digest in meta['runtime_sha256'].items():assert sha(Path(__file__).parent/name)==digest
            track_dim,edge_dim,memory=meta['track_dim'],meta['edge_dim'],meta['memory']
        else:
            assert meta['format']=='b02-residual-association-v1' and meta['feature_version']=='b02-association-context-v2'
            for name in ('features.py','context.py','model.py','tracker.py'):
                assert sha(ROOT/'experiments/v2_association'/name)==meta['sources'][name]
            track_dim,edge_dim,memory=60,50,False
        model=AssociationModel(track_dim,edge_dim);model.load_state_dict(ck['model'])
        scorer=Scorer(model,'cpu',1.,'rank_only')
    else:
        handler,spec=load_candidate(ROOT/'profiles/v2-association-candidate.json','cpu');scorer=handler.scorer;memory=False
    torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False);start=monotonic()
    names=['physical','geo4','learned4','learned8'];summaries={n:{'kind':'local_diagnostics','config':{'identity_variant':n,'point_set':'exactly upstream selections'},'scenes':{}} for n in names}
    diagnostics={n:{} for n in names};checks={}
    save(a.out/'manifest.json',{'scenes':scenes,'data_sha256':cache_manifest['data_sha256'],'split_sha256':sha(ROOT/'splits/v3_tracking.json'),
        'checkpoint_sha256':sha(a.checkpoint) if a.checkpoint else spec['checkpoint_sha256'],'scorer_memory':memory,
        'sources':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'confirmation_used':False,'holdout_used':False})
    with open_scenes(a.data) as (_,opener):
        for scene in scenes:
            with opener(scene) as file:frames=list(_scans(file))
            with np.load(a.cache/'calibration'/f'{scene}.npz') as data:all_scores=data['point_scores'];offsets=data['point_offsets']
            physical=Tracker(scorer,memory=memory);managers={'geo4':IdentityManager(None,4),
                'learned4':IdentityManager(scorer,4,memory),'learned8':IdentityManager(scorer,8,memory)}
            diags={n:Diagnostic() for n in names};outputs={n:[] for n in names}
            for i,(circle,points,labels) in enumerate(frames):
                if monotonic()-start>a.seconds:raise TimeoutError('Identity-only replay budget')
                scores=all_scores[offsets[i]:offsets[i+1]].tolist();before=set(physical.state.get('tracks',{}));chosen=physical(points,scores)
                outputs['physical'].append(chosen);diags['physical'].observe(points,labels,chosen,physical,before,circle)
                for name,manager in managers.items():
                    before=set(manager.engine.state.get('tracks',{}));selected=manager(points,scores,chosen,physical.current_context)
                    assert [{k:v for k,v in p.items() if k!='BatchID'} for p in selected]==[{k:v for k,v in p.items() if k!='BatchID'} for p in chosen]
                    outputs[name].append(selected);diags[name].observe(points,labels,selected,manager.engine,before,circle)
            for name in names:
                folder=a.out/name;folder.mkdir(exist_ok=True)
                stream=(folder/f'{scene}.jsonl').open('w',encoding='utf-8') if scene=='train_032' else Discard()
                try:
                    iterator=iter(outputs[name]);stats=replay_frames(frames,stream,lambda p,s:next(iterator))
                finally:
                    if hasattr(stream,'close'):stream.close()
                if name=='physical':
                    checks[scene]=reference_equal(stats,reference['scenes'][scene])
                    if not a.checkpoint:assert checks[scene],scene
                for key in ('points','outputs','truth_points','matched_target_points','clutter_outputs','unknown_outputs'):
                    if name!='physical':assert stats[key]==summaries['physical']['scenes'][scene][key]
                summaries[name]['scenes'][scene]=stats;diagnostics[name][scene]=diags[name].result()
                save(folder/'summary.json',summaries[name]);save(folder/'diagnostics.json',diagnostics[name])
            print(json.dumps({'scene':scene,'seconds':round(monotonic()-start,2),'switches':{n:diags[n].counts.get('switches',0) for n in names}}),flush=True)
    results={}
    for name in names:
        card=build_scorecard(summaries[name],name,summaries['physical'] if name!='physical' else reference,'physical' if name!='physical' else 'frozen-v2')
        write_scorecard(a.out/name,card);counts=Counter()
        for d in diagnostics[name].values():counts.update(d['counts'])
        results[name]={'metrics':card['overall']['values'],'identity':dict(counts)}
    save(a.out/'report.json',{'complete':True,'results':results,'scenes':scenes,'physical_reference_equal':checks,
        'point_sets_identical':True,'wall_seconds':monotonic()-start,'confirmation_used':False,'holdout_used':False})


if __name__=='__main__':main()
