"""Preserve pre-transfer local context for an otherwise unmatched receiver."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from b02.replay import _scans, open_scenes
from experiments.candidate_onset.joint_cost_replay import JointCostTracker, run
from experiments.candidate_onset.mixed import install
from experiments.candidate_onset.propagation import sha
from experiments.robust_validation.frozen import Components
from experiments.v2_association.run_stage1 import save


class ContextHoldTracker(JointCostTracker):
    total_context_holds = 0

    def __call__(self, points, scores):
        self._prior_context = {}
        outputs = super().__call__(points, scores)
        for batch, previous in self._prior_context.items():
            self.context_history[batch] = previous
        self.total_context_holds += len(self._prior_context)
        return outputs

    def assign(self, track_ids, scan, positions, grid, occupied, points, scores):
        stage = self.assignment_stage
        matched = super().assign(track_ids, scan, positions, grid, occupied, points, scores)
        if stage == 0:
            self._prior_context = {p.rival: self.context_history[p.rival].copy()
                                   for p in self.joint_applied if p.other is None}
        return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'calibration', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.protocol.read_bytes())
    root = Path(__file__).resolve().parents[2]
    assert all(sha(root / name) == digest for name, digest in plan['sources'].items())
    assert sha(args.data) == plan['data_sha256'] and sha(args.freeze) == plan['parent_freeze_sha256']
    assert sha(args.calibration) == plan['calibration_sha256']
    calibration = json.loads(args.calibration.read_bytes())
    assert calibration['protocol_sha256'] == plan['joint_cost_protocol_sha256']
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    split = json.loads((root / 'splits/v3_tracking.json').read_bytes())
    names = plan['scenes']
    assert set(names) <= set(components.freeze['work_validation16']) and not set(names) & set(split['sealed_excluded'])
    scene = names[args.index]
    manifest = json.loads((args.cache / 'manifest.json').read_bytes())
    cache_path = args.cache / f'{scene}.npz'
    assert sha(cache_path) == manifest['scenes'][scene]['cache_sha256']
    with open_scenes(args.data) as (_, opener), opener(scene) as source:
        frames = list(_scans(source))
    with np.load(cache_path) as cache:
        scores, offsets = cache['point_scores'], cache['point_offsets']
    assert len(offsets) == len(frames) + 1
    tracker = install(components.build('candidate'), 'combined')
    tracker.__class__ = ContextHoldTracker
    tracker.joint_cutoff = calibration['cutoff']
    args.out.mkdir(parents=True, exist_ok=True)
    report = run(frames, scores, offsets, tracker, args.out, scene, shadow=False)
    report['operations']['context_holds'] = tracker.total_context_holds
    save(args.out / f'{scene}.json', dict(report, scene=scene, protocol_sha256=sha(args.protocol)))
    print(scene, report['identity']['mixed_minority_truth_points'], report['operations'], flush=True)


if __name__ == '__main__':
    main()
