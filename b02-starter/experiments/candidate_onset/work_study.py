"""Frozen baseline/combined check on the previously exposed work16 only."""
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
from experiments.candidate_onset.mixed import install
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.robust_validation.oracles import shadow_identity
from experiments.v2_association.run_stage1 import Diagnostic,Discard,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','cache','prior','affinity','head','fusion','freeze','protocol','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--index',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(1)
    plan=json.loads(args.protocol.read_text(encoding='utf-8'))
    for name,digest in plan['sources'].items():assert sha(ROOT/name)==digest,name
    assert sha(args.data)==plan['data_sha256'] and sha(args.freeze)==plan['parent_freeze_sha256']
    stage_path=args.out.parent/'mixed-development8-comparison.json'
    assert sha(stage_path)==plan['development_report_sha256']
    stage=json.loads(stage_path.read_text(encoding='utf-8'))
    assert stage['primary_passed'] and stage['protocol_sha256']==plan['previous_study_protocol_sha256']
    components=Components(args.prior,args.affinity,args.head,args.fusion,args.freeze)
    assert plan['work16']==components.freeze['work_validation16']
    split=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    assert not set(plan['work16'])&set(split['fit64']+split['calibration']+split['confirmation16']+split['sealed_excluded'])
    cache=json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['complete'] and cache['freeze_sha256']==sha(args.freeze) and cache['data_sha256']==sha(args.data)
    assert cache['point_weights_sha256']==components.freeze['artifacts']['point']
    args.out.mkdir(parents=True,exist_ok=False);began=monotonic()
    with open_scenes(args.data) as (available,opener):
        assert set(available)==set(plan['work16'])
        for scene in plan['work16'][args.index::args.shards]:
            with opener(scene) as source:frames=list(_scans(source))
            path=args.cache/f'{scene}.npz';assert sha(path)==cache['scenes'][scene]['cache_sha256']
            with np.load(path) as stored:scores,offsets=stored['point_scores'],stored['point_offsets']
            assert len(offsets)==len(frames)+1
            row=dict(scene=scene,variants={},identity={},birth_audit={},shadow={},operations={},phase='work16',protocol_sha256=sha(args.protocol))
            for mode in ('baseline','combined'):
                tracker=install(components.build('candidate'),mode)
                birth,diagnostic=BirthAudit(),Diagnostic();index=0
                folder=args.out/mode;folder.mkdir(exist_ok=True)
                stream=(folder/f'{scene}.jsonl').open('w',encoding='utf-8') if scene==plan['work16'][0] else Discard()
                def run(points,_state):
                    nonlocal index
                    circle,_,labels=frames[index];before=set(tracker.state.get('tracks',{}))
                    output=tracker(points,scores[offsets[index]:offsets[index+1]].tolist());index+=1
                    birth.observe(points,labels,output,tracker,circle);diagnostic.observe(points,labels,output,tracker,before,circle)
                    return output
                try:summary=replay_frames(frames,stream,run)
                finally:
                    if hasattr(stream,'close'):stream.close()
                row['variants'][mode]=summary;row['identity'][mode]=diagnostic.result()['counts']
                row['birth_audit'][mode]=birth.result();row['shadow'][mode]=shadow_identity(summary)
                row['operations'][mode]=dict(tracker.motion_counts)
            save(args.out/f'{scene}.json',row)
            print(scene,'work16 fixed pair complete',round(monotonic()-began,1),flush=True)


if __name__=='__main__':main()
