"""Capture first visible old-ID failures on fixed fit-only scenes."""
import argparse
import json
from pathlib import Path
import pickle
import sys
from time import monotonic

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from b02.replay import _scans,open_scenes
from experiments.candidate_onset.outcomes import OutcomeAudit as Audit
from experiments.candidate_onset.observe import build,snapshot
from experiments.robust_validation.frozen import Components
from experiments.v2_association.run_stage1 import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','cache','prior','affinity','head','fusion','freeze','protocol','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--index',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(1)
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    for name,digest in protocol['sources'].items():assert sha(ROOT/name)==digest,name
    assert sha(args.freeze)==protocol['parent_freeze_sha256'] and sha(args.data)==protocol['data_sha256']
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    names=protocol['training_sample4']+protocol['head_validation8']
    assert set(names)<=set(plan['fit64']) and not set(names)&set(plan['calibration']+plan['confirmation16']+plan['sealed_excluded'])
    cache=json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['complete'] and cache['data_sha256']==sha(args.data)
    components=Components(args.prior,args.affinity,args.head,args.fusion,args.freeze)
    assert cache['point_weights_sha256']==components.freeze['artifacts']['point']
    args.out.mkdir(parents=True,exist_ok=False);began=monotonic()
    with open_scenes(args.data) as (_,opener):
        for scene in names[args.index::args.shards]:
            with opener(scene) as source:frames=list(_scans(source))
            last={}
            for frame,(_,pts,labels) in enumerate(frames):
                for truth in labels.values():
                    if truth>0:last[truth]=frame
            with np.load(args.cache/'fit'/f'{scene}.npz') as cached:scores,offsets=cached['point_scores'],cached['point_offsets']
            assert len(offsets)==len(frames)+1
            tracker=build(components);audit=Audit(last);fixtures=[]
            shadow=components.build('candidate') if scene==names[0] else None
            for frame,(circle,points,labels) in enumerate(frames):
                before=snapshot(tracker)
                raw_before=pickle.dumps(tracker,protocol=5) if len(fixtures)<protocol['fixtures_per_scene'] else None
                values=scores[offsets[frame]:offsets[frame+1]].tolist()
                outputs=tracker(points,values)
                if shadow is not None:assert outputs==shadow(points,values),(scene,circle)
                events=audit.observe(frame,circle,points,labels,outputs,tracker,before,values)
                for event in events:
                    if raw_before is None or len(fixtures)>=protocol['fixtures_per_scene']:break
                    if event['candidate_sources']['main'] or event['reason']=='output_gate':continue
                    folder=args.out/'fixtures';folder.mkdir(exist_ok=True)
                    path=folder/f'{scene}-{len(fixtures):02d}.pkl'
                    raw=pickle.dumps(dict(before=raw_before,points=points,scores=values,labels=labels,observed_outputs=outputs),protocol=5)
                    path.write_bytes(raw)
                    manifest=dict(event,scene=scene,fixture_sha256=sha(path),protocol_sha256=sha(args.protocol),
                        expected_previous_id=event['batch'],fitting_pool_only=True,private_project_created_fixture=True)
                    save(path.with_suffix('.json'),manifest);fixtures.append(path.stem)
            audit.finalize()
            save(args.out/f'{scene}.json',dict(scene=scene,role='training_sample4' if scene in protocol['training_sample4'] else 'head_validation8',
                counts=dict(audit.counts),events=audit.events,fixtures=fixtures,protocol_sha256=sha(args.protocol),
                shadow_all_outputs_equal=shadow is not None,circles=len(frames),elapsed_seconds=monotonic()-began,
                future_usage='Only last target presence for external end-of-scene exclusion; never passed to tracker.'))
            print(scene,'onset audit complete',len(audit.events),'events',len(fixtures),'fixtures',round(monotonic()-began,1),flush=True)


if __name__=='__main__':main()
