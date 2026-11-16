"""Paired short/long memory supervision on fixed, causally replayed prefixes."""
import argparse
from collections import Counter,deque
import json
from pathlib import Path
import sys
from time import monotonic

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import save,sha,historical_identity,Discard
from experiments.v3_tracking.diagnose import reference_equal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--prefix',choices=('v1','v2'),required=True)
    p.add_argument('--reuse-v1-pairs',type=Path)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--seconds',type=int,default=3500);a=p.parse_args()
    import numpy as np
    import torch
    from b02.replay import open_scenes,_scans,replay_frames
    from experiments.v2_association.candidate import load_candidate
    from experiments.v3_tracking.tracker import Tracker
    from experiments.v2_association.features import MAX_EDGES
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    cache_manifest=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache_manifest['complete'] and cache_manifest['fit']==plan['fit64'] and cache_manifest['calibration']==plan['calibration']
    assert cache_manifest['data_sha256']==sha(a.data)
    names=plan['fit64']+plan['calibration'];assert not set(names)&set(plan['confirmation16']+plan['sealed_excluded']+plan['development_regression'])
    assert 0<=a.shard_index<a.shard_count<=len(names)
    names=names[a.shard_index::a.shard_count]
    frozen=None
    if a.prefix=='v2':frozen,_=load_candidate(ROOT/'profiles/v2-association-candidate.json','cpu')
    torch.set_num_threads(2)
    a.out.mkdir(parents=True,exist_ok=True);start=monotonic();report={'complete':False,'scenes':{},'shard_index':a.shard_index,'shard_count':a.shard_count,
        'prefix':a.prefix,'data_sha256':cache_manifest['data_sha256'],'cache_manifest_sha256':sha(a.cache/'manifest.json'),
        'split_sha256':sha(ROOT/'splits/v3_tracking.json'),'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
    reused=json.loads((a.reuse_v1_pairs/'manifest.json').read_text(encoding='utf-8')) if a.reuse_v1_pairs else None
    if reused:assert reused['feature_version']=='b02-association-context-v2' and reused.get('label_mode','physical')=='physical'
    with open_scenes(a.data) as (_,opener):
        for name in names:
            if monotonic()-start>a.seconds:raise TimeoutError('Pair preparation budget')
            role='fit' if name in plan['fit64'] else 'calibration';folder=a.out/role;folder.mkdir(exist_ok=True);path=folder/f'{name}.npz'
            assert not path.exists()
            if a.prefix=='v1' and reused and name in reused['fit']+reused['calibration']:
                old_role='fit' if name in reused['fit'] else 'calibration';old=a.reuse_v1_pairs/old_role/f'{name}.npz'
                with np.load(old) as data:arrays={k:data[k] for k in ('track','edges','base','valid','target')}
                np.savez_compressed(path,**arrays)
                report['scenes'][name]={'role':role,'reused':True,'groups':len(arrays['track']),'source_sha256':sha(old)}
            else:
                with opener(name) as file:frames=list(_scans(file))
                with np.load(a.cache/role/f'{name}.npz') as cache:scores_all=cache['point_scores'];offsets=cache['point_offsets']
                assert len(offsets)==len(frames)+1
                starts=np.linspace(0,len(frames)-8,8,dtype=int);selected={i for s in starts for i in range(int(s),int(s)+8)}
                tracker=Tracker(frozen.scorer if frozen else None,memory=a.prefix=='v2')
                truths={};anchors={};counts=Counter();parts={k:[] for k in ('track','edges','base','valid','target','known')}
                if a.prefix=='v2':parts.update({k:[] for k in ('memory_track','memory_edges','anchor_target','anchor_known')})
                for frame_index,(circle,points,labels) in enumerate(frames):
                    if monotonic()-start>a.seconds:raise TimeoutError('Pair preparation budget')
                    scores=scores_all[offsets[frame_index]:offsets[frame_index+1]].tolist()
                    assert len(scores)==len(points)
                    tracker.feature_capture=frame_index in selected
                    tracker(points,scores)
                    if frame_index in selected:
                        for event in tracker.events:
                            counts['rows_seen']+=1
                            if not event['indices']:counts['empty_candidates']+=1;continue
                            batch=event['batch'];pure=historical_identity(truths.get(batch,()))
                            anchor=anchors.get(batch,pure)
                            if anchor is not None and anchor<0:anchor=None
                            if pure is None and (a.prefix=='v1' or anchor is None):counts['unknown_identity']+=1;continue
                            n=len(event['indices']);valid=np.zeros(MAX_EDGES+1,dtype=bool);valid[-1]=True
                            point_truth=[labels[points[i]['PntID']] for i in event['indices']]
                            valid[:n]=np.asarray(point_truth)>=0
                            if not valid[:-1].any():continue
                            def target(identity):
                                mask=np.zeros(MAX_EDGES+1,dtype=bool)
                                if identity is not None and identity>0:mask[:n]=np.asarray(point_truth)==identity
                                if not mask.any():mask[-1]=True
                                return mask
                            y=target(pure);tx=event.get('short_track_features',event['track_features']);ex=event.get('short_edge_features',event['edge_features'])
                            padded=np.zeros((MAX_EDGES,50),dtype=np.float32);padded[:n]=ex
                            base=np.zeros(MAX_EDGES+1,dtype=np.float32);base[:n]=event['base_costs'];base[-1]=event['dummy_cost']
                            for key,value in (('track',tx),('edges',padded),('base',base),('valid',valid),('target',y),('known',pure is not None)):
                                parts[key].append(value)
                            counts['pure_positive']+=pure is not None and not y[-1]
                            if a.prefix=='v2':
                                long=np.zeros((MAX_EDGES,85),dtype=np.float32);long[:n]=event['edge_features']
                                parts['memory_track'].append(event['track_features']);parts['memory_edges'].append(long)
                                ay=target(anchor);parts['anchor_target'].append(ay);parts['anchor_known'].append(anchor is not None)
                                counts['anchor_positive']+=anchor is not None and not ay[-1]
                                counts['anchor_recovery_rows']+=anchor is not None and (pure is None or anchor!=pure)
                    # The row labels above see only the prefix BEFORE this scan.
                    for batch,index in tracker.last_matches.items():truths.setdefault(batch,deque(maxlen=8)).append(labels[points[index]['PntID']])
                    for batch,index in tracker.last_births.items():truths[batch]=deque([labels[points[index]['PntID']]],maxlen=8)
                    if a.prefix=='v2':
                        for batch in tracker.new_anchors:anchors[batch]=labels[points[tracker.trusted_updates[batch]]['PntID']]
                    for mapping in (truths,anchors):
                        for batch in list(mapping):
                            if batch not in tracker.state['tracks']:del mapping[batch]
                if not parts['track']:raise ValueError('No supervised rows')
                np.savez_compressed(path,**{k:np.asarray(v) for k,v in parts.items()})
                report['scenes'][name]={'role':role,'reused':False,'groups':len(parts['track']),'counts':dict(counts),'frames':len(frames),'supervised_frames':len(selected)}
            report['scenes'][name]['sha256']=sha(path);save(a.out/f'part-{a.shard_index}.json',report)
            print(json.dumps({'scene':name,'prefix':a.prefix,'groups':report['scenes'][name]['groups'],'seconds':round(monotonic()-start,2)}),flush=True)
    report['complete']=True;report['wall_seconds']=monotonic()-start;save(a.out/f'part-{a.shard_index}.json',report)


if __name__=='__main__':main()
