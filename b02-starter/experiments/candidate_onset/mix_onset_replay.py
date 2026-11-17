"""Test whether one selected onset update causes later mixed outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from b02.replay import _scans, open_scenes
from experiments.candidate_onset.joint_event import clone, trace_keys
from experiments.candidate_onset.mixed import install
from experiments.candidate_onset.propagation import restore, sha, veto_this_scan
from experiments.candidate_onset.propagation_work import capture_prefix
from experiments.candidate_onset.propagation_work_summary import read
from experiments.robust_validation.frozen import Components
from experiments.v2_association.run_stage1 import save


def measure(keys, labels, batch, truth):
    actual = [labels[p] for b,p in keys if b == batch]
    if truth in actual:
        return 'correct'
    return 'other_target' if any(t > 0 for t in actual) else 'silent_or_other'


def run_case(tracker, frames, cached, event, reference, horizon):
    start, batch, truth = event['start'], event['batch'], event['expected']
    for i, (_, points, _) in enumerate(frames[:start]):
        assert trace_keys(tracker(points, cached(i))) == reference[i], i
    _, points, _ = frames[start]
    index = next(i for i,p in enumerate(points) if p['PntID'] == event['onset']['point'])
    variants, trajectories = {}, {}
    for mode in ('original', 'veto_onset', 'retain_prior_state'):
        trial = clone(tracker)
        if mode == 'veto_onset':
            veto_this_scan(trial, batch, index)
        traces = []
        for i in range(start, start+horizon+1):
            circle, points, labels = frames[i]
            output = trace_keys(trial(points, cached(i)))
            if i == start:
                if mode == 'veto_onset':
                    del trial.assign
                    assert (batch,event['onset']['point']) not in output
                else:
                    assert output == reference[i]
                if mode == 'retain_prior_state':
                    old = capture_prefix(tracker, (batch,))
                    assert restore(trial, old, batch, full=True)
            if mode == 'original':
                assert output == reference[i], i
            traces.append(dict(circle=circle, outcome=measure(output,labels,batch,truth),
                               true_observed=truth in labels.values()))
        follow = Counter(row['outcome'] for row in traces[1:])
        variants[mode] = dict(onset=traces[0]['outcome'], follow=dict(follow))
        trajectories[mode] = traces
    return dict(event=event, variants=variants, trajectories=trajectories)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data','cache','prior','affinity','head','fusion','freeze','protocol','cases','audit','out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    plan, manifest = read(args.protocol), read(args.cases)
    assert manifest['enabled'] and manifest['protocol_sha256'] == sha(args.protocol)
    assert sha(args.data) == plan['data_sha256'] and sha(args.freeze) == plan['parent_freeze_sha256']
    components = Components(args.prior,args.affinity,args.head,args.fusion,args.freeze)
    event = manifest['cases'][args.index]
    scene = event['scene']
    assert scene in plan['scenes'] and scene in components.freeze['work_validation16']
    baseline = read(args.audit/(scene+'.json'))
    assert baseline['scene'] == scene and baseline['protocol_sha256'] == sha(args.protocol)
    cursor = args.cache/(scene+'.npz')
    assert sha(cursor) == read(args.cache/'manifest.json')['scenes'][scene]['cache_sha256']
    with open_scenes(args.data) as (_, opener), opener(scene) as source:
        frames = list(_scans(source))
    with np.load(cursor) as cache:
        scores, offsets = cache['point_scores'], cache['point_offsets']
    assert len(offsets) == len(frames)+1

    def cached(i):
        return scores[offsets[i]:offsets[i+1]].tolist()

    expected = [tuple(tuple(key) for key in row) for row in baseline['frozen_outputs']]
    report = run_case(install(components.build('candidate'),'combined'), frames, cached, event,
                      expected, manifest['horizon'])
    report.update(scene=scene, protocol_sha256=sha(args.protocol), cases_sha256=sha(args.cases))
    save(args.out/(scene+'.json'), report)
    print(scene,event['onset']['circle'],report['variants'],flush=True)


if __name__ == '__main__':
    main()
