"""Minimize current points/competitors while preserving the actual history."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys
from time import monotonic

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.candidate_onset.repro import load,run
from experiments.v2_association.run_stage1 import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fixture',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    import torch
    torch.set_num_threads(1)
    manifest,packet=load(args.fixture)
    original=pickle.loads(packet['before']);baseline=deepcopy(original)
    pairs=list(zip(packet['points'],packet['scores']));started=monotonic();trials=0
    def check(t,values):
        nonlocal trials
        trials+=1
        if monotonic()-started>120 or trials>256:raise TimeoutError('minimizer budget exhausted')
        points,scores=map(list,zip(*values))
        if not any(packet['labels'][pt['PntID']]==manifest['truth'] for pt in points):return False,None
        result=run(deepcopy(t),points,scores,packet['labels'],manifest['batch'],manifest['truth'])
        return not result[0] and result[1]==manifest['reason'],result
    assert check(baseline,pairs)[0]
    # A real historical prefix is the fixed unit; do not invent a simpler synthetic history.
    keep={manifest['batch']}|set(manifest['raw_target_outputs'])
    reduced=deepcopy(baseline)
    def prune(t,ids):
        t.state['tracks']={b:v for b,v in t.state['tracks'].items() if b in ids}
        for name in ('history','context_history','birth_scan','quality_hits','strong_hits','memories','retired','short_retired',
            'emissions','last_emission'):
            value=getattr(t,name)
            for b in list(value):
                if b not in ids:del value[b]
    prune(reduced,keep)
    if check(reduced,pairs)[0]:baseline=reduced
    # Remove each remaining unrelated track when the exact reason survives.
    for batch in list(baseline.state['tracks']):
        if batch==manifest['batch']:continue
        reduced=deepcopy(baseline);prune(reduced,(set(baseline.state['tracks'])-{batch})|{manifest['batch']})
        if check(reduced,pairs)[0]:baseline=reduced
    width=max(1,len(pairs)//2)
    while width:
        changed=False
        for start in range(0,len(pairs),width):
            candidate=pairs[:start]+pairs[start+width:]
            if not candidate:continue
            if check(baseline,candidate)[0]:pairs=candidate;changed=True;break
        if not changed:width//=2
    for i in range(len(pairs)):
        if len(pairs)>1:assert not check(baseline,pairs[:i]+pairs[i+1:])[0]
    ok,result=check(baseline,pairs);assert ok
    points,scores=map(list,zip(*pairs))
    args.out.parent.mkdir(parents=True,exist_ok=True);assert not args.out.exists()
    args.out.write_bytes(pickle.dumps(dict(before=pickle.dumps(baseline,protocol=5),points=points,scores=scores,
        labels={p['PntID']:packet['labels'][p['PntID']] for p in points},observed_outputs=result[2]),protocol=5))
    manifest.update(fixture_sha256=sha(args.out),minimized_from_sha256=sha(args.fixture),
        original_tracks=len(original.state['tracks']),remaining_tracks=len(baseline.state['tracks']),
        original_points=len(packet['points']),remaining_points=len(points),trials=trials,
        point_irreducible_given_fixed_real_history=True,minimization_seconds=monotonic()-started)
    save(args.out.with_suffix('.json'),manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
