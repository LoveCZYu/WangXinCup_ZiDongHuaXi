"""Research-only free-running diagnostic; does not override a failed gate."""
import argparse
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.replay import _scans, open_scenes, replay_frames
from experiments.competition_learning.model import Model
from experiments.competition_learning.runtime import LearnedScorer
from experiments.continuation.fusion import MemoryFusionTracker
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.robust_validation.oracles import shadow_identity
from experiments.v2_association.run_stage1 import Diagnostic, Discard, save, sha

MODES = ('baseline', 'binary_absolute', 'joint_raw', 'joint_calibrated')


class OffsetScorer(LearnedScorer):
    def __init__(self, prior, learned, tracker, offset):
        super().__init__(prior, learned, tracker, 'learned_joint')
        self.offset = offset

    def __call__(self, events):
        result = super().__call__(events)
        for row, event in enumerate(events):
            if self.tracker.emissions.get(event['batch'], 0) >= 3:
                edge, null = result[row]
                result[row] = (edge, null-self.offset)
        return result


def checkpoint(path, kind, protocol, trained, seed):
    assert sha(path) == trained['models'][f'{kind}-{seed}']['selected_weights_sha256']
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['protocol_sha256'] == sha(protocol) and saved['kind'] == kind and saved['seed'] == seed
    assert saved['model_source_sha256'] == sha(Path(__file__).with_name('model.py'))
    assert saved['feature_source_sha256'] == sha(Path(__file__).with_name('features.py'))
    model = Model().eval()
    model.load_state_dict(saved['model'], strict=True)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'pilot',
        'binary_weights', 'joint_weights', 'training_report', 'calibration_report', 'out'):
        p.add_argument('--'+name.replace('_','-'), dest=name, type=Path, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    pilot = json.loads(args.pilot.read_text(encoding='utf-8'))
    assert pilot['research_only'] and pilot['does_not_override_original_gate']
    assert pilot['learning_protocol_sha256'] == sha(args.protocol)
    for name, digest in pilot['sources'].items():assert sha(ROOT/name) == digest, name
    trained = json.loads(args.training_report.read_text(encoding='utf-8'))
    assert trained['complete'] and sha(args.training_report) == pilot['training_report_sha256']
    calibration = json.loads(args.calibration_report.read_text(encoding='utf-8'))
    assert sha(args.calibration_report) == pilot['calibration_report_sha256']
    seed = protocol['default_seed']
    assert seed == pilot['default_seed']
    binary = checkpoint(args.binary_weights, 'binary', args.protocol, trained, seed)
    joint = checkpoint(args.joint_weights, 'joint', args.protocol, trained, seed)
    offset = calibration['seeds'][str(seed)]['offset_fitted_on_training24_only']
    assert offset == pilot['offset']
    cache = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['complete'] and cache['data_sha256'] == sha(args.data) == protocol['data_sha256']
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    names = pilot['head_validation8']
    assert names == protocol['validation8'] and set(names).isdisjoint(protocol['training24'])
    assert set(names) <= set(plan['fit64']) and not set(names) & set(plan['calibration']+plan['confirmation16']+plan['sealed_excluded'])
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    assert list(MODES) == pilot['modes']
    args.out.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    with open_scenes(args.data) as (_, opener):
        for scene in names[args.index::args.shards]:
            with opener(scene) as source:frames = list(_scans(source))
            with np.load(args.cache/'fit'/f'{scene}.npz') as cached:scores, offsets = cached['point_scores'], cached['point_offsets']
            assert len(offsets) == len(frames)+1
            row = {'scene': scene, 'variants': {}, 'identity': {}, 'birth_audit': {}, 'shadow': {}, 'operations': {},
                'pilot_sha256': sha(args.pilot), 'research_only': True, 'original_offline_gate_passed': trained['offline_passed']}
            for mode in MODES:
                tracker = components.build('candidate')
                if mode != 'baseline':
                    tracker.model_scorer = OffsetScorer(tracker.model_scorer,
                        binary if mode == 'binary_absolute' else joint, tracker, offset if mode == 'joint_calibrated' else 0.)
                birth, diagnostic = BirthAudit(), Diagnostic()
                index = 0
                def run(points, _state):
                    nonlocal index
                    circle, _, labels = frames[index]
                    before = set(tracker.state.get('tracks', {}))
                    output = tracker(points, scores[offsets[index]:offsets[index+1]].tolist())
                    index += 1
                    birth.observe(points, labels, output, tracker, circle)
                    diagnostic.observe(points, labels, output, tracker, before, circle)
                    return output
                folder = args.out/mode
                folder.mkdir(exist_ok=True)
                stream = (folder/f'{scene}.jsonl').open('w',encoding='utf-8') if scene == names[0] else Discard()
                try:summary = replay_frames(frames, stream, run)
                finally:
                    if hasattr(stream, 'close'):stream.close()
                row['variants'][mode] = summary
                row['identity'][mode] = diagnostic.result()['counts']
                row['birth_audit'][mode] = birth.result()
                row['shadow'][mode] = shadow_identity(summary)
                row['operations'][mode] = dict(getattr(tracker.model_scorer, 'counts', {}))
            save(args.out/f'{scene}.json', row)
            print(scene, 'research pilot complete', round(monotonic()-started, 1), flush=True)


if __name__ == '__main__':main()
