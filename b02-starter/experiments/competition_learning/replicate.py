"""One prespecified second-seed replication of the exploratory binary column."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'replication',
                 'weights', 'training_report', 'out'):
        p.add_argument('--'+name.replace('_','-'), dest=name, type=Path, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    rep = json.loads(args.replication.read_text(encoding='utf-8'))
    assert rep['learning_protocol_sha256'] == sha(args.protocol)
    for name, digest in rep['sources'].items():assert sha(ROOT/name) == digest, name
    trained = json.loads(args.training_report.read_text(encoding='utf-8'))
    assert trained['complete'] and sha(args.training_report) == rep['training_report_sha256']
    seed = protocol['seeds'][1]
    assert seed == rep['seed']
    learned = checkpoint(args.weights, 'binary', args.protocol, trained, seed)
    cache = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['complete'] and cache['data_sha256'] == sha(args.data) == protocol['data_sha256']
    names = rep['head_validation8']
    assert names == protocol['validation8'] and set(names).isdisjoint(protocol['training24'])
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    args.out.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    with open_scenes(args.data) as (_, opener):
        for scene in names[args.index::args.shards]:
            with opener(scene) as source:frames = list(_scans(source))
            with np.load(args.cache/'fit'/f'{scene}.npz') as data:scores, offsets = data['point_scores'], data['point_offsets']
            tracker = Tracker(components, learned, 'learned_column')
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
            summary = replay_frames(frames, Discard(), run)
            save(args.out/f'{scene}.json', {'scene': scene, 'summary': summary,
                'identity': diagnostic.result()['counts'], 'birth_audit': birth.result(),
                'shadow': shadow_identity(summary), 'replication_sha256': sha(args.replication), 'seed': seed})
            print(scene, 'second binary seed complete', round(monotonic()-started, 1), flush=True)


if __name__ == '__main__':main()
