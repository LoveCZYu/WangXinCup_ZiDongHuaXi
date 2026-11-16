"""Sharded replay using an immutable data/cache pair with fixed scene roles."""
import argparse
from pathlib import Path
import json
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.replay import open_scenes, _scans, replay_frames
from experiments.v2_association.run_stage1 import Diagnostic, Discard, save, sha
from experiments.v3_tracking.diagnose import reference_equal
from experiments.v3_tracking.runtime import load_scorer
from experiments.v3_tracking.tracker import Tracker
from experiments.joint_matching.diagnose import FragmentationAudit, IN_GATE, WITH_GATE


def stats(outputs, frames, trace=None):
    iterator = iter(outputs)
    if trace:
        with trace.open('w', encoding='utf-8') as target:
            return replay_frames(frames, target, lambda _p, _s: next(iterator))
    return replay_frames(frames, Discard(), lambda _p, _s: next(iterator))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'reference', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--method', choices=('audit', 'continuity_rule', 'learned'), default='audit')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--scene', nargs='+')
    args = parser.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    reference = json.loads(args.reference.read_text(encoding='utf-8'))['scenes']
    manifest = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['data_sha256'] == sha(args.data) and manifest['calibration'] == plan['calibration']
    scorer, memory, _, _ = load_scorer(args.prior)
    assert memory and 0 <= args.index < args.shards
    names = args.scene or plan['calibration']
    assert len(set(names)) == len(names) and set(names) <= set(plan['calibration'])
    model = None
    if args.method != 'audit':
        from experiments.joint_matching.model import JointTracker, PlanModel
        if args.method == 'learned':
            state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
            assert state['prior_sha256'] == sha(args.prior)
            assert state['model_sha256'] == sha(Path(__file__).with_name('model.py'))
            model = PlanModel().eval()
            model.load_state_dict(state['model'])
        else:
            assert args.checkpoint is None
    args.out.mkdir(parents=True, exist_ok=False)
    start = monotonic()
    with open_scenes(args.data) as (_, opener):
        for scene in names[args.index::args.shards]:
            with opener(scene) as source:
                frames = list(_scans(source))
            with np.load(args.cache/'calibration'/f'{scene}.npz') as cache:
                scores, offsets = cache['point_scores'], cache['point_offsets']
            assert len(offsets) == len(frames)+1
            tracker = Tracker(scorer, memory=True) if args.method == 'audit' else JointTracker(scorer, model=model, rule=args.method == 'continuity_rule')
            audit = FragmentationAudit() if args.method == 'audit' else None
            diagnostic = Diagnostic()
            outputs = []
            for i, (circle, points, labels) in enumerate(frames):
                if audit:
                    before = audit.snapshot(tracker)
                active = set(tracker.state.get('tracks', {}))
                result = tracker(points, scores[offsets[i]:offsets[i+1]].tolist())
                if audit:
                    audit.observe(circle, points, labels, result, tracker, before)
                diagnostic.observe(points, labels, result, tracker, active, circle)
                outputs.append(result)
            if not audit:
                result = stats(outputs, frames, args.out/f'{scene}.jsonl' if scene == 'train_032' else None)
                save(args.out/f'{scene}.json', {'scene': scene, 'variants': {args.method: result},
                                               'identity': diagnostic.result()['counts'], 'plans': dict(tracker.plan_counts)})
                print(scene, dict(tracker.plan_counts), round(monotonic()-start, 1), flush=True)
                continue
            baseline = stats(outputs, frames)
            assert reference_equal(baseline, reference[scene]), scene
            breakdown = audit.result()
            variants = {'baseline': baseline}
            operations = {}
            for name, causes, size in (('existing_candidate', IN_GATE, None), ('include_gate_near', WITH_GATE, None),
                                       ('short_fragments_3', None, 3), ('all_secondary', None, None)):
                repaired, operations[name] = audit.repair(outputs, frames, breakdown['fragments'], causes, size)
                variants[name] = stats(repaired, frames)
                for key in ('matched_target_points', 'clutter_outputs', 'unknown_outputs', 'outputs'):
                    assert variants[name][key] == baseline[key]
            save(args.out/f'{scene}.json', {'scene': scene, 'variants': variants, 'repair': operations,
                                           'attribution': breakdown, 'identity': diagnostic.result()['counts']})
            print(scene, len(breakdown['fragments']), round(monotonic()-start, 1), flush=True)


if __name__ == '__main__':
    main()
