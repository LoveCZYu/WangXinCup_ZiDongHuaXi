"""Fixed eight-file calibration replay; truth is external to inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from b02.replay import _scans, open_scenes, replay_frames
from experiments.continuation.tracker import MODES, ContinuationTracker
from experiments.id_recovery.affinity import load_affinity
from experiments.id_recovery.replay import classify_recovery
from experiments.prebirth.replay import BirthAudit
from experiments.v2_association.run_stage1 import Diagnostic, Discard, save, sha
from experiments.v3_tracking.diagnose import reference_equal
from experiments.v3_tracking.runtime import load_scorer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','cache','prior','affinity','reference','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--narrow-reference',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--method',choices=sorted(MODES),required=True)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--memory-guard',type=float)
    parser.add_argument('--blend-config',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    manifest=json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['data_sha256']==sha(args.data) and manifest['calibration']==plan['calibration']
    assert not set(plan['calibration'])&set(plan['sealed_excluded']+plan['confirmation16'])
    assert 0<=args.index<args.shards
    scorer,memory,_,_=load_scorer(args.prior)
    assert memory
    affinity=load_affinity(args.affinity,args.prior)
    model,threshold=None,None
    if MODES[args.method].learned or args.blend_config:
        from experiments.continuation.learn import load_model
        model,metadata=load_model(args.checkpoint,args.prior,args.affinity)
        threshold=metadata['thresholds'].get(args.method)
    blend=None
    if args.blend_config:
        from experiments.continuation import fusion
        blend=json.loads(args.blend_config.read_text(encoding='utf-8'))
        assert args.method=='wide_shadow' and blend['weights_sha256']==sha(args.checkpoint)
        assert blend['source_sha256']==sha(fusion.__file__)
    reference=json.loads(args.reference.read_text(encoding='utf-8'))['scenes']
    narrow=json.loads(args.narrow_reference.read_text(encoding='utf-8'))['scenes'] if args.narrow_reference else None
    args.out.mkdir(parents=True,exist_ok=False)
    started=monotonic()
    with open_scenes(args.data) as (_,opener):
        for scene in plan['calibration'][args.index::args.shards]:
            with opener(scene) as source:frames=list(_scans(source))
            with np.load(args.cache/'calibration'/f'{scene}.npz') as cache:
                scores,offsets=cache['point_scores'],cache['point_offsets']
            assert len(offsets)==len(frames)+1
            if blend is not None:
                from experiments.continuation.fusion import FusionTracker,MemoryFusionTracker
                tracker=FusionTracker(scorer,affinity,model,blend) if args.memory_guard is None else MemoryFusionTracker(scorer,affinity,model,blend,guard=args.memory_guard)
            elif args.memory_guard is None:
                tracker=ContinuationTracker(scorer,affinity,args.method,model,threshold)
            else:
                from experiments.continuation.memory_guard import MemoryGuardTracker
                tracker=MemoryGuardTracker(scorer,affinity,args.method,model,threshold,guard=args.memory_guard)
            diagnostic,audit,decisions=Diagnostic(),BirthAudit(),Counter()
            index=0
            def run(points,_state):
                nonlocal index
                circle,_,labels=frames[index]
                before=set(tracker.state.get('tracks',{}))
                predictions=tracker(points,scores[offsets[index]:offsets[index+1]].tolist())
                index+=1
                classify_recovery(tracker,points,labels,diagnostic.owners,predictions,decisions)
                for batch,p,accepted in getattr(tracker,'memory_decisions',[]):
                    if accepted:continue
                    owners=diagnostic.owners.get(batch,{})
                    owner=max(owners,key=owners.get) if owners else None
                    truth=labels[points[p]['PntID']]
                    decisions['memory_blocked_same_prior_truth' if truth>0 and truth==owner else 'memory_blocked_other_or_unlabelled']+=1
                audit.observe(points,labels,predictions,tracker,circle)
                diagnostic.observe(points,labels,predictions,tracker,before,circle)
                return predictions
            stream=(args.out/f'{scene}.jsonl').open('w',encoding='utf-8') if scene=='train_032' else Discard()
            try:summary=replay_frames(frames,stream,run)
            finally:
                if hasattr(stream,'close'):stream.close()
            if args.method=='wide_shadow' and args.memory_guard is None and blend is None:assert reference_equal(summary,reference[scene]),scene
            if args.method=='narrow_shadow':assert narrow is not None and reference_equal(summary,narrow[scene]),scene
            save(args.out/f'{scene}.json',{'scene':scene,'variants':{args.method:summary},
                'identity':diagnostic.result()['counts'],'plans':dict(decisions+tracker.counters),
                'birth_audit':audit.result(),'threshold':threshold,'memory_guard':args.memory_guard})
            print(scene,args.method,dict(decisions),round(monotonic()-started,1),flush=True)


if __name__=='__main__':main()
