"""Fixed four-way full replay, with fitting and old calibration roles only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.replay import _scans, open_scenes, replay_frames
from experiments.identity_competition.scoring import CompetitionTracker
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.robust_validation.oracles import shadow_identity
from experiments.v2_association.run_stage1 import Diagnostic, Discard, save, sha

MODES = ('baseline', 'identity', 'volume', 'identity_volume')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'out'):
        p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--role', choices=('fitting', 'calibration'), required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    for name, digest in protocol['sources'].items():
        assert sha(ROOT/name) == digest, name
    assert sha(args.freeze) == protocol['frozen_protocol_sha256']
    cache = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['complete'] and cache['data_sha256'] == sha(args.data) == protocol['data_sha256']
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    names = protocol[args.role]
    assert set(names) <= set(plan['fit64'] if args.role == 'fitting' else plan['calibration'])
    assert not set(names) & set(plan['sealed_excluded'] + plan['confirmation16'] + plan['development_regression'])
    assert list(MODES) == protocol['modes']
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    assert cache['point_weights_sha256'] == components.freeze['artifacts']['point']
    args.out.mkdir(parents=True, exist_ok=False)
    start = monotonic()
    preview = names[0]
    with open_scenes(args.data) as (_, opener):
        for scene in names[args.index::args.shards]:
            with opener(scene) as source:
                frames = list(_scans(source))
            cache_path = args.cache/('fit' if args.role == 'fitting' else 'calibration')/f'{scene}.npz'
            with np.load(cache_path) as payload:
                scores, offsets = payload['point_scores'], payload['point_offsets']
            assert len(offsets) == len(frames)+1
            row = {'scene': scene, 'role': args.role, 'variants': {}, 'identity': {}, 'birth_audit': {},
                   'shadow': {}, 'interventions': {}, 'protocol_sha256': sha(args.protocol)}
            for mode in MODES:
                tracker = CompetitionTracker(components.scorer, components.affinity, components.head, components.fusion, mode)
                shadow = components.build('candidate') if mode == 'baseline' else None
                audit, diagnostic = BirthAudit(), Diagnostic()
                index = 0
                def run(points, _state):
                    nonlocal index
                    circle, _, labels = frames[index]
                    before = set(tracker.state.get('tracks', {}))
                    prediction = tracker(points, scores[offsets[index]:offsets[index+1]].tolist())
                    if shadow is not None:
                        assert prediction == shadow(points, scores[offsets[index]:offsets[index+1]].tolist()), (scene, circle)
                    index += 1
                    audit.observe(points, labels, prediction, tracker, circle)
                    diagnostic.observe(points, labels, prediction, tracker, before, circle)
                    return prediction
                folder = args.out/mode
                folder.mkdir(exist_ok=True)
                stream = (folder/f'{scene}.jsonl').open('w', encoding='utf-8') if scene == preview else Discard()
                try:
                    summary = replay_frames(frames, stream, run)
                finally:
                    if hasattr(stream, 'close'):stream.close()
                row['variants'][mode] = summary
                row['identity'][mode] = diagnostic.result()['counts']
                row['birth_audit'][mode] = audit.result()
                row['shadow'][mode] = shadow_identity(summary)
                row['interventions'][mode] = dict(tracker.model_scorer.counts)
            save(args.out/f'{scene}.json', row)
            print(scene, 'four modes complete', round(monotonic()-start, 1), flush=True)


if __name__ == '__main__':
    main()
