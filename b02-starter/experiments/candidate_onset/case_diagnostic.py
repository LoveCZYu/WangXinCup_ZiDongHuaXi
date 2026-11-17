"""Preregistered one-variable tests on real/minimal first-loss captures."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from b02.replay import _scans,open_scenes
from experiments.candidate_onset.repro import load,run
from experiments.identity_competition.motion import memory_motion
from experiments.v2_association.features import normalized
from experiments.v2_association.run_stage1 import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','protocol','data','out'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    plan=json.loads(args.protocol.read_text(encoding='utf-8'))
    assert sha(Path(__file__))==plan['source_sha256'] and sha(args.data)==plan['data_sha256']
    findings={}
    labels_by_scene={};memory_by_scene={}
    with open_scenes(args.data) as (_,opener):
        for scene in plan['scenes']:
            with opener(scene) as source:frames=list(_scans(source))
            labels_by_scene[scene]={(pt['CircleID'],pt['PntID']):labels[pt['PntID']]
                for _,pts,labels in frames for pt in pts}
            memory_by_scene[scene]={(frame+1,normalized(pt).tobytes()):labels[pt['PntID']]
                for frame,(_,pts,labels) in enumerate(frames) for pt in pts}
    for relative in plan['fixtures']:
        path=args.root/relative;manifest,packet=load(path)
        assert sha(path)==plan['fixture_sha256'][relative]
        original=pickle.loads(packet['before']);batch=manifest['batch'];truth=manifest['truth']
        labels=labels_by_scene[manifest['scene']]
        history=original.history.get(batch,[])
        memory=original.memories.get(batch)
        when=original.state['scan']+1
        previous=original.state['tracks'].get(batch)
        motion=memory_motion(memory,previous.last_scan) if memory is not None and previous is not None else None
        tests={}
        for mode in plan['modes']:
            tracker=pickle.loads(packet['before'])
            track=tracker.state['tracks'].get(batch)
            if mode=='zero_neural_correction':tracker.model_scorer.scale=0.
            if mode in ('memory_position','memory_velocity','memory_both') and motion is not None:
                if mode in ('memory_position','memory_both'):track.position=tuple(motion[0])
                if mode in ('memory_velocity','memory_both'):track.velocity=tuple(motion[1])
            correct,reason,outputs,_=run(tracker,packet['points'],packet['scores'],packet['labels'],batch,truth)
            tests[mode]=dict(correct_same_id=correct,reason=reason,output_count=len(outputs),
                selected_points_changed={p['PntID'] for p in outputs}!={p['PntID'] for p in packet['observed_outputs']})
        raw_current=[normalized(pt) for pt in packet['points'] if packet['labels'][pt['PntID']]==truth]
        before_residual=[];backup_residual=[]
        if previous is not None:
            predicted=np.asarray(previous.position)+np.asarray(previous.velocity)*(when-previous.last_scan)
            before_residual=[float(np.linalg.norm(x-predicted)) for x in raw_current]
            if motion is not None:backup_residual=[float(np.linalg.norm(x-motion[0]-motion[1]*(when-previous.last_scan))) for x in raw_current]
        trusted_labels=[]
        if memory is not None:
            for xyz,_,scan,_,_ in memory.entries:
                # Resolve memory labels only in this external diagnostic by exact past normalized point bytes.
                assert scan<when
                trusted_labels.append(memory_by_scene[manifest['scene']].get((scan,xyz.tobytes()),'not_found'))
        findings[relative]=dict(scene=manifest['scene'],circle=manifest['circle'],reason=manifest['reason'],tests=tests,
            short_history_labels=dict(Counter(str(labels.get((pt['CircleID'],pt['PntID']))) for pt,_,_ in history)),
            trusted_labels=dict(Counter(map(str,trusted_labels))),trusted_count=len(memory.entries) if memory else 0,
            last_trusted_scan=memory.entries[-1][2] if memory and memory.entries else None,
            last_physical_scan=previous.last_scan if previous is not None else None,
            physical_residual_norm=before_residual,backup_residual_norm=backup_residual,
            caveat='Targeted past-only position/velocity tests are diagnostic sufficient interventions; not a deployable policy for selecting which ID to change.')
    save(args.out,dict(protocol_sha256=sha(args.protocol),findings=findings,training=False,independent_validation=False))
    print(json.dumps(findings,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
