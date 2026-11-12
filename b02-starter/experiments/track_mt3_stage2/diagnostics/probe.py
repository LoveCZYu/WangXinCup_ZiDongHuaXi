"""Deterministic diagnostic probes; no tuning on the locked holdout set."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--scene',default='train_015')
    p.add_argument('--start',type=int,default=100)
    p.add_argument('--frames',type=int,default=8)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cold',action='store_true')
    p.add_argument('--mode',choices=['repro'],default='repro')
    return p.parse_args()


def main():
    a=arguments()
    sys.path[:0]=[str(a.upstream.resolve()),str(a.project.resolve()),
                  str(a.project/'experiments/track_mt3_stage2')]
    import torch
    from adapter import make_model,Handler
    from b02.replay import open_scenes,_scans
    torch.set_num_threads(4);torch.manual_seed(20261111)
    model=make_model(a.upstream)
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu')['model'],strict=True)
    model.eval()
    with open_scenes(a.data) as (_,opener):
        with opener(a.scene) as source:
            frames=[]
            for record in _scans(source):
                frames.append(record)
                if record[0]>=a.start+a.frames-1:break
    handler=Handler(model);state={};truth=hits=outputs=0
    for circle,points,labels in frames:
        if a.cold and circle<a.start:continue
        result=handler(points,state)
        if circle>=a.start:
            truth+=sum(label>0 for label in labels.values())
            hits+=sum(labels[p['PntID']]>0 for p in result)
            outputs+=len(result)
    report={'scene':a.scene,'start':a.start,'frames':a.frames,'cold':a.cold,
            'truth':truth,'hits':hits,'outputs':outputs,'recall':hits/truth if truth else None,
            'minimum_recall':.5,'pass':bool(truth and hits/truth>=.5)}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    assert report['pass'], 'REPRODUCED: B02 model misses more than half of labelled target points'


if __name__=='__main__':main()
