"""One predeclared work-validation pass, with frozen point scores and models."""
import argparse
from collections import Counter,deque
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from b02.replay import _scans,open_scenes,replay_frames
from experiments.id_recovery.replay import classify_recovery
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components,MODES
from experiments.robust_validation.oracles import shadow_identity
from experiments.v2_association.run_stage1 import Diagnostic,Discard,save,sha


def cache_points(args,freeze):
    from b02.point_tracker import load_tracker
    assert sha(ROOT/'assets/point-scorer-8000.pt')==freeze['artifacts']['point']
    for name,digest in freeze['sources'].items():assert sha(ROOT/name)==digest,name
    handler,_=load_tracker(device='cpu');torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    selected=freeze['work_validation16'][args.index::args.shards]
    report={'complete':False,'index':args.index,'shards':args.shards,'scenes':{},
            'data_sha256':sha(args.data),'point_weights_sha256':freeze['artifacts']['point'],
            'freeze_sha256':sha(args.freeze),'labels_used_for_point_inference':False}
    with open_scenes(args.data) as (available,opener):
        assert set(available)==set(freeze['work_validation16'])
        for scene in selected:
            scores=[];offsets=[0];history=deque(maxlen=handler.history-1)
            with opener(scene) as source:
                for _,points,_labels in _scans(source):
                    with torch.inference_mode():
                        values=handler.model(torch.as_tensor(handler.features(points,list(history)))).sigmoid().tolist()
                    history.append(points);scores.extend(values);offsets.append(len(scores))
            path=args.out/f'{scene}.npz';assert not path.exists()
            np.savez_compressed(path,point_scores=np.asarray(scores,dtype=np.float32),
                point_offsets=np.asarray(offsets,dtype=np.int64),data_sha256=report['data_sha256'],
                point_weights_sha256=freeze['artifacts']['point'])
            report['scenes'][scene]={'frames':len(offsets)-1,'points':len(scores),'cache_sha256':sha(path)}
            save(args.out/f'part-{args.index}.json',report)
            print(scene,'point cache',len(scores),flush=True)
    report['complete']=True;save(args.out/f'part-{args.index}.json',report)


def merge_cache(args,freeze):
    scenes={}
    for index in range(args.shards):
        part=json.loads((args.cache/f'part-{index}.json').read_text(encoding='utf-8'))
        assert part['complete'] and part['index']==index and part['shards']==args.shards
        assert part['freeze_sha256']==sha(args.freeze) and part['data_sha256']==freeze['data_sha256']
        assert not set(scenes)&set(part['scenes']);scenes.update(part['scenes'])
    assert set(scenes)==set(freeze['work_validation16'])
    path=args.cache/'manifest.json';assert not path.exists()
    save(path,{'complete':True,'scenes':scenes,'data_sha256':freeze['data_sha256'],
               'point_weights_sha256':freeze['artifacts']['point'],'freeze_sha256':sha(args.freeze)})


def evaluate(args,freeze):
    cache=json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['complete'] and cache['freeze_sha256']==sha(args.freeze)
    assert cache['point_weights_sha256']==freeze['artifacts']['point']
    components=Components(args.prior,args.affinity,args.head,args.fusion,args.freeze)
    args.out.mkdir(parents=True,exist_ok=False);start=monotonic()
    preview=freeze['work_validation16'][0]
    with open_scenes(args.data) as (available,opener):
        assert set(available)==set(freeze['work_validation16'])
        for scene in freeze['work_validation16'][args.index::args.shards]:
            with opener(scene) as source:frames=list(_scans(source))
            path=args.cache/f'{scene}.npz';assert sha(path)==cache['scenes'][scene]['cache_sha256']
            with np.load(path) as data:
                scores,offsets=data['point_scores'],data['point_offsets']
                assert data['data_sha256'].item()==freeze['data_sha256']
            assert len(offsets)==len(frames)+1
            row={'scene':scene,'variants':{},'identity':{},'birth_audit':{},'shadow':{},'plans':{}}
            for mode in MODES:
                tracker=components.build(mode)
                diagnostic,audit,decisions=Diagnostic(),BirthAudit(),Counter();index=0
                def run(points,_state):
                    nonlocal index
                    circle,_,labels=frames[index]
                    before=set(tracker.state.get('tracks',{}))
                    prediction=tracker(points,scores[offsets[index]:offsets[index+1]].tolist());index+=1
                    classify_recovery(tracker,points,labels,diagnostic.owners,prediction,decisions)
                    audit.observe(points,labels,prediction,tracker,circle)
                    diagnostic.observe(points,labels,prediction,tracker,before,circle)
                    return prediction
                folder=args.out/mode;folder.mkdir(exist_ok=True)
                stream=(folder/f'{scene}.jsonl').open('w',encoding='utf-8') if scene==preview else Discard()
                try:summary=replay_frames(frames,stream,run)
                finally:
                    if hasattr(stream,'close'):stream.close()
                row['variants'][mode]=summary;row['identity'][mode]=diagnostic.result()['counts']
                row['birth_audit'][mode]=audit.result();row['shadow'][mode]=shadow_identity(summary)
                row['plans'][mode]=dict(decisions+tracker.counters)
            save(args.out/f'{scene}.json',row)
            print(scene,'all three frozen modes complete',round(monotonic()-start,1),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('cache','merge-cache','evaluate'))
    for name in ('data','cache','prior','affinity','head','fusion','freeze','out'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--index',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args();torch.set_num_threads(1)
    freeze=json.loads(args.freeze.read_text(encoding='utf-8'))
    if args.data:assert sha(args.data)==freeze['data_sha256']
    assert 0<=args.index<args.shards
    {'cache':cache_points,'merge-cache':merge_cache,'evaluate':evaluate}[args.command](args,freeze)


if __name__=='__main__':main()
