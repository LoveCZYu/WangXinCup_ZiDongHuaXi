"""Research-only transfer: learned column evidence, original row/null costs."""
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
from experiments.competition_learning.runtime import Tracker
from experiments.competition_learning.pilot import checkpoint
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.robust_validation.oracles import shadow_identity
from experiments.v2_association.run_stage1 import Diagnostic, Discard, save, sha

MODES = ('baseline', 'old_column', 'binary_column', 'joint_column')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'transfer',
                 'binary_weights', 'joint_weights', 'training_report', 'out'):
        p.add_argument('--'+name.replace('_','-'), dest=name, type=Path, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    transfer = json.loads(args.transfer.read_text(encoding='utf-8'))
    assert transfer['learning_protocol_sha256'] == sha(args.protocol) and transfer['research_only']
    for name, digest in transfer['sources'].items():assert sha(ROOT/name) == digest, name
    trained = json.loads(args.training_report.read_text(encoding='utf-8'))
    assert trained['complete'] and sha(args.training_report) == transfer['training_report_sha256']
    binary = checkpoint(args.binary_weights, 'binary', args.protocol, trained, protocol['default_seed'])
    joint = checkpoint(args.joint_weights, 'joint', args.protocol, trained, protocol['default_seed'])
    cache = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['complete'] and cache['data_sha256'] == sha(args.data) == protocol['data_sha256']
    names = transfer['head_validation8']
    assert names == protocol['validation8'] and set(names).isdisjoint(protocol['training24'])
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    assert list(MODES) == transfer['modes']
    args.out.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    with open_scenes(args.data) as (_, opener):
        for scene in names[args.index::args.shards]:
            with opener(scene) as source:frames = list(_scans(source))
            with np.load(args.cache/'fit'/f'{scene}.npz') as cached:scores, offsets = cached['point_scores'], cached['point_offsets']
            assert len(offsets) == len(frames)+1
            row = {'scene': scene, 'variants': {}, 'identity': {}, 'birth_audit': {}, 'shadow': {},
                   'transfer_sha256': sha(args.transfer), 'research_only': True}
            for mode in MODES:
                tracker = Tracker(components, binary if mode == 'binary_column' else joint,
                    'baseline' if mode == 'baseline' else 'old_column_identity' if mode == 'old_column' else 'learned_column')
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
                stream = (folder/f'{scene}.jsonl').open('w', encoding='utf-8') if scene == names[0] else Discard()
                try:summary = replay_frames(frames, stream, run)
                finally:
                    if hasattr(stream, 'close'):stream.close()
                row['variants'][mode] = summary
                row['identity'][mode] = diagnostic.result()['counts']
                row['birth_audit'][mode] = birth.result()
                row['shadow'][mode] = shadow_identity(summary)
            save(args.out/f'{scene}.json', row)
            print(scene, 'column transfer complete', round(monotonic()-started, 1), flush=True)


if __name__ == '__main__':main()
