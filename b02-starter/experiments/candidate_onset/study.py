"""Fixed four-way whole-scene physical-state correction experiment."""
import argparse
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from b02.replay import _scans,open_scenes,replay_frames
from experiments.candidate_onset.policy import MODES,install
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.robust_validation.oracles import shadow_identity
from experiments.v2_association.run_stage1 import Diagnostic,Discard,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','cache','prior','affinity','head','fusion','freeze','protocol','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=('training4','development8'),required=True)
    p.add_argument('--index',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(1)
    plan=json.loads(args.protocol.read_text(encoding='utf-8'))
    for name,digest in plan['sources'].items():assert sha(ROOT/name)==digest,name
    assert sha(args.freeze)==plan['parent_freeze_sha256'] and sha(args.data)==plan['data_sha256']
    split=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    names=plan['training4' if args.phase=='training4' else 'development8']
    assert set(names)<=set(split['fit64']) and not set(names)&set(split['calibration']+split['confirmation16']+split['sealed_excluded'])
    cache=json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['complete'] and cache['data_sha256']==sha(args.data)
    components=Components(args.prior,args.affinity,args.head,args.fusion,args.freeze)
    if args.phase=='development8':
        stage=json.loads((args.out.parent/'training4-comparison.json').read_text(encoding='utf-8'))
        assert stage['primary_passed'] and stage['protocol_sha256']==sha(args.protocol)
    args.out.mkdir(parents=True,exist_ok=False);began=monotonic()
    with open_scenes(args.data) as (_,opener):
        for scene in names[args.index::args.shards]:
            with opener(scene) as source:frames=list(_scans(source))
            with np.load(args.cache/'fit'/f'{scene}.npz') as cached:scores,offsets=cached['point_scores'],cached['point_offsets']
            assert len(offsets)==len(frames)+1
            row=dict(scene=scene,variants={},identity={},birth_audit={},shadow={},operations={},phase=args.phase,protocol_sha256=sha(args.protocol))
            for mode in MODES:
                tracker=install(components.build('candidate'),mode)
                birth,diagnostic=BirthAudit(),Diagnostic();index=0
                folder=args.out/mode;folder.mkdir(exist_ok=True)
                stream=(folder/f'{scene}.jsonl').open('w',encoding='utf-8') if scene==names[0] else Discard()
                def run(points,_state):
                    nonlocal index
                    circle,_,labels=frames[index]
                    before=set(tracker.state.get('tracks',{}))
                    outputs=tracker(points,scores[offsets[index]:offsets[index+1]].tolist());index+=1
                    birth.observe(points,labels,outputs,tracker,circle)
                    diagnostic.observe(points,labels,outputs,tracker,before,circle)
                    return outputs
                try:summary=replay_frames(frames,stream,run)
                finally:
                    if hasattr(stream,'close'):stream.close()
                row['variants'][mode]=summary;row['identity'][mode]=diagnostic.result()['counts']
                row['birth_audit'][mode]=birth.result();row['shadow'][mode]=shadow_identity(summary)
                row['operations'][mode]=dict(tracker.motion_counts)
            save(args.out/f'{scene}.json',row)
            print(scene,args.phase,'four-way complete',round(monotonic()-began,1),flush=True)


if __name__=='__main__':main()
