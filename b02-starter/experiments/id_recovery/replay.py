"""Full causal recovery replay on the fixed eight calibration files."""
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
from experiments.id_recovery.affinity import load_affinity
from experiments.id_recovery.tracker import CONFIGS, RecoveryTracker
from experiments.v2_association.run_stage1 import Diagnostic, Discard, save, sha
from experiments.v3_tracking.diagnose import reference_equal
from experiments.v3_tracking.runtime import load_scorer


def classify_recovery(tracker, points, labels, emitted_history, outputs, decisions):
    selected = {(p['BatchID'], p['PntID']) for p in outputs}
    for operation, batch, index, rival in tracker.decisions:
        point_id = points[index]['PntID']
        truth = labels[point_id]
        prior = emitted_history.get(batch, {})
        previous = max(prior, key=prior.get) if prior else None
        if truth > 0 and truth == previous:
            label = 'same_prior_truth'
        elif truth > 0 and previous is not None:
            label = 'other_truth'
        elif truth == 0:
            label = 'known_clutter'
        else:
            label = 'unknown_or_no_prior_truth'
        decisions[operation+'_accepted'] += 1
        decisions[operation+'_'+label] += 1
        decisions[operation+'_visible'] += (batch, point_id) in selected
        if rival is not None and truth > 0:
            rival_history = emitted_history.get(rival, {})
            decisions['active_rival_also_supported_truth'] += truth in rival_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'reference', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--affinity', type=Path)
    parser.add_argument('--transfer', type=Path)
    parser.add_argument('--method', choices=sorted(CONFIGS)+['active_conflict'], required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    cache = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['data_sha256'] == sha(args.data) and cache['calibration'] == plan['calibration']
    assert not set(cache['calibration']) & set(plan['sealed_excluded'] + plan['confirmation16'])
    assert 0 <= args.index < args.shards
    reference = json.loads(args.reference.read_text(encoding='utf-8'))['scenes']
    shared, memory, _, _ = load_scorer(args.prior)
    assert memory
    model = None if args.method == 'baseline' else load_affinity(args.affinity, args.prior)
    if args.method == 'active_conflict':
        from experiments.id_recovery import transfer
        weights = torch.load(args.transfer, map_location='cpu', weights_only=True)
        assert weights['affinity_sha256'] == sha(args.affinity) and weights['source_sha256'] == sha(transfer.__file__)
        rescoring = transfer.TransferModel().eval()
        rescoring.load_state_dict(weights['model'], strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    start = monotonic()
    with open_scenes(args.data) as (_, opener):
        for scene in plan['calibration'][args.index::args.shards]:
            with opener(scene) as source:
                frames = list(_scans(source))
            with np.load(args.cache/'calibration'/f'{scene}.npz') as data:
                scores, offsets = data['point_scores'], data['point_offsets']
            assert len(offsets) == len(frames)+1
            if args.method == 'active_conflict':
                tracker = transfer.TransferTracker(shared, model, rescoring)
            else:
                tracker = RecoveryTracker(shared, model, CONFIGS[args.method])
            diagnostic = Diagnostic()
            decisions = Counter()
            index = 0

            def run(points, _state):
                nonlocal index
                circle, _, labels = frames[index]
                before = set(tracker.state.get('tracks', {}))
                predictions = tracker(points, scores[offsets[index]:offsets[index+1]].tolist())
                index += 1
                classify_recovery(tracker, points, labels, diagnostic.owners, predictions, decisions)
                diagnostic.observe(points, labels, predictions, tracker, before, circle)
                return predictions

            stream = (args.out/f'{scene}.jsonl').open('w', encoding='utf-8') if scene == 'train_032' else Discard()
            try:
                summary = replay_frames(frames, stream, run)
            finally:
                if hasattr(stream, 'close'):
                    stream.close()
            if args.method == 'baseline':
                assert reference_equal(summary, reference[scene]), scene
            save(args.out/f'{scene}.json', {'scene': scene, 'variants': {args.method: summary},
                                           'identity': diagnostic.result()['counts'], 'plans': dict(decisions+tracker.counters)})
            print(scene, dict(decisions), round(monotonic()-start, 1), flush=True)


if __name__ == '__main__':
    main()
