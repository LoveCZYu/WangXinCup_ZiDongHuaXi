"""Real-capture regression for the conservative backup branch."""
import argparse
import json
from pathlib import Path
import pickle
import sys
from time import monotonic

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.candidate_onset.fallback import install,MODES
from experiments.candidate_onset.repro import load,run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fixture',type=Path)
    p.add_argument('--mode',choices=MODES,default='backup_guarded');args=p.parse_args()
    import torch
    torch.set_num_threads(1)
    manifest,packet=load(args.fixture);began=monotonic();results=[];operations=[]
    baseline=run(pickle.loads(packet['before']),packet['points'],packet['scores'],packet['labels'],manifest['batch'],manifest['truth'])
    assert baseline[2]==packet['observed_outputs'] and not baseline[0]
    for _ in range(2):
        tracker=install(pickle.loads(packet['before']),args.mode)
        results.append(run(tracker,packet['points'],packet['scores'],packet['labels'],manifest['batch'],manifest['truth']))
        operations.append(dict(tracker.motion_counts))
    assert results[0]==results[1] and operations[0]==operations[1]
    print(json.dumps(dict(verdict='GREEN' if results[0][0] else 'RED',mode=args.mode,scene=manifest['scene'],
        circle=manifest['circle'],operations=operations[0],repeat_exact=True,seconds=monotonic()-began),ensure_ascii=False))
    raise SystemExit(0 if results[0][0] else 1)


if __name__=='__main__':main()
