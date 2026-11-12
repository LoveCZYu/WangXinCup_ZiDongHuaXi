"""Full dev-scene replay of every final point-scorer arm, without tuning."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--project', type=Path, required=True)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--seconds', type=int, default=210)
p.add_argument('--filter-stage', choices=('pre', 'post'), default='pre')
a = p.parse_args()
sys.path.insert(0, str(a.project.resolve()))
import torch
from model import PointScorer, Handler, point_features, HISTORY
from b02.algorithm import process_circle
from b02.replay import replay

torch.set_num_threads(4)
started = time.monotonic()
out = a.run/('full-dev' if a.filter_stage == 'pre' else 'full-dev-postfilter')
out.mkdir(exist_ok=False)
shutil.copy2(__file__, out/'full_replay.py')
report = {'scenes': ['train_015', 'train_016'], 'threshold': .5, 'soft_seconds': a.seconds,
          'filter_stage': a.filter_stage,
          'selection': 'All four final checkpoints; no validation-based selection or threshold changes.',
          'holdout_used': False, 'runs': [], 'complete': False}
manifest = json.loads((a.run/'manifest.json').read_text())
assert hashlib.sha256((Path(__file__).parent/'model.py').read_bytes()).hexdigest() == manifest['source_sha256']['model.py']
items = [('baseline', None)] + [(f'{seed}-{variant}', a.run/f'{seed}-{variant}.pt')
                               for seed in manifest['seeds'] for variant in manifest['variants']]

def guarded(handler):
    def call(points, state):
        if time.monotonic()-started > a.seconds:
            raise TimeoutError('Full replay budget exceeded; incomplete output is not a final score')
        return handler(points, state)
    return call


class PostFilterHandler:
    """Keep baseline association state unchanged; filter only its emitted points."""
    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, points, state):
        history = state.setdefault('point_history', deque(maxlen=HISTORY-1))
        features = point_features(points, list(history))
        scores = self.model(torch.as_tensor(features, device='cuda')).sigmoid().cpu().numpy()
        keep = {p['PntID'] for p, score in zip(points, scores) if score >= .5}
        history.append(points)
        output = process_circle(points, state.setdefault('association', {}))
        return [p for p in output if p['PntID'] in keep]

try:
    for name, checkpoint in items:
        torch.cuda.reset_peak_memory_stats()
        if checkpoint is None:
            handler = process_circle
        else:
            saved = torch.load(checkpoint, map_location='cpu')
            assert saved['step'] == manifest['steps_per_arm']
            model = PointScorer(temporal=saved['variant'] == 'temporal').cuda()
            model.load_state_dict(saved['model'], strict=True)
            handler = Handler(model, threshold=.5) if a.filter_stage == 'pre' else PostFilterHandler(model)
        replay(a.data, out/name, guarded(handler),
               {'kind': 'point_scorer_diagnostic', 'variant': name, 'threshold': .5, 'filter_stage': a.filter_stage,
                'checkpoint': str(checkpoint) if checkpoint else None}, scenes=report['scenes'])
        summary = json.loads((out/name/'summary.json').read_text())
        total = {key: sum(scene[key] for scene in summary['scenes'].values())
                 for key in ['circles', 'points', 'truth_points', 'matched_target_points',
                             'clutter_outputs', 'unknown_outputs', 'truth_tracks', 'matched_truth_tracks', 'split_truth_tracks']}
        known = total['matched_target_points']+total['clutter_outputs']
        total['recall'] = total['matched_target_points']/total['truth_points']
        total['precision'] = total['matched_target_points']/known if known else 0
        total['point_f1'] = 2*total['matched_target_points']/(total['truth_points']+known)
        item = {'variant': name, **total, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
        report['runs'].append(item)
        print(json.dumps(item), flush=True)
        (out/'report.json').write_text(json.dumps(report, indent=2))
        if checkpoint is not None:
            del model, handler, saved
            torch.cuda.empty_cache()
    report['complete'] = True
except TimeoutError as error:
    report['budget_stop'] = str(error)
finally:
    report['elapsed_seconds'] = time.monotonic()-started
    (out/'report.json').write_text(json.dumps(report, indent=2))
print(json.dumps({'event': 'full_replay_complete', 'complete': report['complete'],
                  'elapsed_seconds': report['elapsed_seconds']}), flush=True)
