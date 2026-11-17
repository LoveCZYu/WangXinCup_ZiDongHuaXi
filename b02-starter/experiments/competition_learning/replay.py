"""Run the fixed default head only after its declared offline gate passes."""
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
from experiments.competition_learning.runtime import Tracker, MODES, load_model
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.robust_validation.oracles import shadow_identity
from experiments.v2_association.run_stage1 import Diagnostic, Discard, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'integration',
                 'weights', 'training_report', 'out'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    integration = json.loads(args.integration.read_text(encoding='utf-8'))
    assert integration['learning_protocol_sha256'] == sha(args.protocol)
    for name, digest in integration['sources'].items():assert sha(ROOT/name) == digest, name
    trained = json.loads(args.training_report.read_text(encoding='utf-8'))
    assert trained['complete'] and trained['offline_passed'], 'Offline gate failed; no rollout permitted'
    assert trained['protocol_sha256'] == sha(args.protocol)
    expected = trained['models'][f"joint-{protocol['default_seed']}"]['selected_weights_sha256']
    assert sha(args.weights) == expected
    learned, saved = load_model(args.weights, args.protocol)
    assert saved['seed'] == protocol['default_seed']
    cache = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['complete'] and cache['data_sha256'] == sha(args.data) == protocol['data_sha256']
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    names = integration['fitting8']
    assert len(names) == 8 and set(names) <= set(plan['fit64'])
    assert not set(names) & set(plan['calibration']+plan['confirmation16']+plan['sealed_excluded'])
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    assert list(MODES) == integration['modes']
    args.out.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    with open_scenes(args.data) as (_, opener):
        for scene in names[args.index::args.shards]:
            with opener(scene) as source:frames = list(_scans(source))
            with np.load(args.cache/'fit'/f'{scene}.npz') as cached:scores, offsets = cached['point_scores'], cached['point_offsets']
            row = {'scene': scene, 'variants': {}, 'identity': {}, 'birth_audit': {}, 'shadow': {}, 'operations': {},
                'protocol_sha256': sha(args.protocol), 'integration_sha256': sha(args.integration), 'weights_sha256': sha(args.weights)}
            for mode in MODES:
                tracker = Tracker(components, learned, mode)
                shadow = components.build('candidate') if mode == 'baseline' else None
                birth, diagnostic = BirthAudit(), Diagnostic()
                index = 0
                def run(points, _state):
                    nonlocal index
                    circle, _, labels = frames[index]
                    values = scores[offsets[index]:offsets[index+1]].tolist()
                    before = set(tracker.state.get('tracks', {}))
                    output = tracker(points, values)
                    if shadow is not None:assert output == shadow(points, values), (scene, circle)
                    index += 1
                    birth.observe(points, labels, output, tracker, circle)
                    diagnostic.observe(points, labels, output, tracker, before, circle)
                    return output
                folder = args.out/mode
                folder.mkdir(exist_ok=True)
                stream = (folder/f'{scene}.jsonl').open('w', encoding='utf-8') if scene == names[0] else Discard()
                try:summary = replay_frames(frames, stream, run)
                finally:
                    if hasattr(stream, 'close'):stream.close()
                row['variants'][mode] = summary
                row['identity'][mode] = diagnostic.result()['counts']
                row['birth_audit'][mode] = birth.result()
                row['shadow'][mode] = shadow_identity(summary)
                row['operations'][mode] = dict(getattr(tracker.model_scorer, 'counts', {}))
            save(args.out/f'{scene}.json', row)
            print(scene, 'four-way learned replay complete', round(monotonic()-started, 1), flush=True)


if __name__ == '__main__':main()
