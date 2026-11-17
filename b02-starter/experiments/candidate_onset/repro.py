"""Seconds-scale red/green loop for a project-created first-loss fixture."""
import argparse
import json
from pathlib import Path
import pickle
import sys
from time import monotonic

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.candidate_onset.audit import classify
from experiments.candidate_onset.observe import snapshot
from experiments.v2_association.run_stage1 import sha


def load(path):
    manifest=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert manifest['private_project_created_fixture'] and sha(path)==manifest['fixture_sha256']
    # Only load private fixtures produced by this project's collector, never third-party pickle files.
    return manifest,pickle.loads(path.read_bytes())


def run(tracker,points,scores,labels,batch,target):
    before=snapshot(tracker)
    outputs=tracker(points,scores)
    right=[i for i,p in enumerate(points) if labels[p['PntID']]==target]
    correct=any(p['BatchID']==batch and labels[p['PntID']]==target for p in outputs)
    reason,source=('correct_output',{}) if correct else classify(batch,right,before,tracker,scores)
    return correct,reason,outputs,source


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fixture',type=Path)
    args=p.parse_args()
    import torch
    torch.set_num_threads(1)
    manifest,packet=load(args.fixture)
    started=monotonic()
    first=run(pickle.loads(packet['before']),packet['points'],packet['scores'],packet['labels'],manifest['batch'],manifest['truth'])
    second=run(pickle.loads(packet['before']),packet['points'],packet['scores'],packet['labels'],manifest['batch'],manifest['truth'])
    assert first==second and first[2]==packet['observed_outputs'],'Capture must reproduce exactly'
    assert first[1]==manifest['reason'],'Do not substitute a different failure'
    print(json.dumps(dict(verdict='GREEN' if first[0] else 'RED',symptom='established ID fails to publish its visible target',
        scene=manifest['scene'],circle=manifest['circle'],reason=first[1],expected_id=manifest['batch'],
        target_truth_diagnostic_only=manifest['truth'],repeat_exact=True,seconds=monotonic()-started),ensure_ascii=False))
    raise SystemExit(0 if first[0] else 1)


if __name__=='__main__':main()
