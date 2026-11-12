"""Bounded original-point classification experiment on fixed B02 clips."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project', type=Path, required=True)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--steps', type=int, default=1200)
parser.add_argument('--seconds', type=int, default=600)
args = parser.parse_args()
sys.path.insert(0, str(args.project.resolve()))
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F
from model import PointScorer, point_features, HISTORY, FEATURE_DIM, SPATIAL_DIM
from b02.algorithm import process_circle
from b02.replay import open_scenes, _scans, _checked_outputs, _identity

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
started = time.monotonic()
args.out.mkdir(parents=True, exist_ok=False)
source = args.out/'source'; source.mkdir()
for path in [*Path(__file__).parent.glob('*.py'), args.project/'b02/algorithm.py', args.project/'b02/replay.py']:
    shutil.copy2(path, source/path.name)

train_names = ['train_001', 'train_002', 'train_004', 'train_005']
dev_names = ['train_015', 'train_016']
split = json.loads((args.project/'splits/v1.json').read_text())
assert set(train_names) <= set(split['train']) and set(dev_names) <= set(split['dev'])
assert not set(train_names+dev_names) & set(split['holdout'])
train_specs = [(name, start, 4) for name in train_names for start in (100, 300)]
specs = {
    'seen': train_specs,
    'same_file_unseen': [(name, start, 20) for name in train_names for start in (200, 400)],
    'different_file': [(name, start, 20) for name in dev_names for start in (200, 400)],
}

def covered(items):
    return {(name, circle) for name, start, count in items
            for circle in range(max(0, start-HISTORY+1), start+count)}

assert not covered(train_specs) & covered(specs['same_file_unseen'])
manifest = {
    'seeds': [20261112, 20261113], 'variants': ['spatial', 'temporal'],
    'steps_per_arm': args.steps, 'total_soft_seconds': args.seconds,
    'train_specs': train_specs, 'evaluation': specs, 'holdout_used': False,
    'history': HISTORY, 'feature_dim': FEATURE_DIM, 'spatial_dim': SPATIAL_DIM,
    'fields': ['Azimuth', 'Distance', 'Elevation'], 'point_cap': None,
    'optimizer': 'Adam', 'learning_rate': .001, 'clip_norm': 10,
    'batch': 'one 4-frame clip, fixed round robin', 'loss': 'BCE with training-only neg/pos weight; -1 ignored',
    'primary_threshold': .5, 'diagnostic_thresholds': [.25, .5, .75, .9],
    'checkpoint_selection': 'Final fixed step only. No dev tuning or checkpoint selection.',
    'comparison_limits': 'Same clips/exposures as coverage count=8, but different architecture, optimizer, objective, association and compute; not a pure regression-head ablation.',
    'probability_note': 'Class-weighted sigmoid scores are not calibrated probabilities.',
    'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
    'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},
}

def emit(event):
    event['elapsed_seconds'] = time.monotonic()-started
    with (args.out/'events.jsonl').open('a') as file:
        file.write(json.dumps(event)+'\n')
    print(json.dumps(event), flush=True)

frames = {}
with open_scenes(args.data) as (_, opener):
    for name in train_names+dev_names:
        with opener(name) as file:
            frames[name] = []
            for frame in _scans(file):
                assert frame[0] == len(frames[name])
                frames[name].append(frame)
                if frame[0] >= 419:
                    break

cache = {}
for entries in specs.values():
    for name, start, count in entries:
        for circle in range(start, start+count):
            if (name, circle) in cache:
                continue
            _, current, labels = frames[name][circle]
            # Feature extractor receives points only, no target annotations.
            previous = [item[1] for item in frames[name][max(0, circle-5):circle]]
            features = point_features(current, previous)
            label_array = np.asarray([labels[p['PntID']] for p in current])
            cache[name, circle] = (features, label_array)

training = []
positives = negatives = unknowns = 0
for name, start, count in train_specs:
    x = np.concatenate([cache[name, c][0] for c in range(start, start+count)])
    y = np.concatenate([cache[name, c][1] for c in range(start, start+count)])
    positives += int((y > 0).sum()); negatives += int((y == 0).sum()); unknowns += int((y < 0).sum())
    keep = y >= 0
    training.append((torch.tensor(x[keep], device='cuda'), torch.tensor((y[keep] > 0).astype(np.float32), device='cuda')))
assert positives > 0 and negatives > 0
manifest['training_counts'] = {'positive': positives, 'negative': negatives, 'unknown_ignored': unknowns}
manifest['positive_weight'] = negatives/positives
manifest['parameter_count'] = sum(p.numel() for p in PointScorer().parameters())
(args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
emit({'event': 'prepared', 'training_counts': manifest['training_counts'], 'parameters': manifest['parameter_count']})

def metrics(counts):
    known = counts['hits']+counts['clutter']
    truth = counts['truth']
    return {**counts, 'recall': counts['hits']/truth if truth else None,
            'precision': counts['hits']/known if known else 0,
            'point_f1': 2*counts['hits']/(truth+known) if truth+known else 0}

def evaluate_group(entries, probabilities=None, threshold=.5, tracking=False):
    counts = Counter(truth=0, hits=0, clutter=0, unknown=0)
    identity_tp = 0; details = []
    for name, start, count in entries:
        state = {}; pairs = Counter(); local = Counter(truth=0, hits=0, clutter=0, unknown=0)
        for circle in range(start, start+count):
            _, points, labels = frames[name][circle]
            if probabilities is None:
                selected = points
            else:
                selected = [p for p, score in zip(points, probabilities[name, circle]) if score >= threshold]
            if tracking:
                output = process_circle(selected, state)
            else:
                # Temporary unique IDs only for output-contract checking; no identity score.
                output = [{**p, 'BatchID': index+1} for index, p in enumerate(selected)]
            _checked_outputs(output, {_identity(p) for p in points})
            local['truth'] += sum(v > 0 for v in labels.values())
            local['hits'] += sum(labels[p['PntID']] > 0 for p in output)
            local['clutter'] += sum(labels[p['PntID']] == 0 for p in output)
            local['unknown'] += sum(labels[p['PntID']] == -1 for p in output)
            if tracking:
                pairs.update((labels[p['PntID']], p['BatchID']) for p in output if labels[p['PntID']] > 0)
        counts.update(local)
        if tracking and pairs:
            targets = sorted({t for t, i in pairs}); ids = sorted({i for t, i in pairs})
            matrix = np.asarray([[pairs[t, i] for i in ids] for t in targets])
            rows, cols = linear_sum_assignment(-matrix)
            identity_tp += int(matrix[rows, cols].sum())
        details.append({'scene': name, 'start': start, **metrics(local)})
    result = {**metrics(counts), 'per_clip': details}
    if tracking:
        denominator = counts['truth']+counts['hits']+counts['clutter']
        result.update(identity_tp=identity_tp, identity_f1=2*identity_tp/denominator if denominator else 0)
    return result

@torch.no_grad()
def evaluate(model, final=False):
    model.eval()
    probabilities = {key: model(torch.tensor(value[0], device='cuda')).sigmoid().cpu().numpy()
                     for key, value in cache.items()}
    result = {}
    for group, entries in specs.items():
        item = {'selected_points': evaluate_group(entries, probabilities),
                'filtered_tracker': evaluate_group(entries, probabilities, tracking=True)}
        if final:
            item['threshold_diagnostics'] = {
                str(t): evaluate_group(entries, probabilities, threshold=t) for t in manifest['diagnostic_thresholds']}
        result[group] = item
    return result

report = {'manifest': manifest,
          'baseline': {group: evaluate_group(entries, tracking=True) for group, entries in specs.items()},
          'evaluations': [], 'arms': [], 'complete': False}
weight = torch.tensor(negatives/positives, device='cuda')
budget_stop = False
for seed in manifest['seeds']:
    for variant in manifest['variants']:
        if time.monotonic()-started > args.seconds-45:
            budget_stop = True
            break
        torch.manual_seed(seed)
        model = PointScorer(temporal=variant == 'temporal').cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        train_seconds = 0.; losses = []; actual_step = 0
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.steps+1):
            deadline = time.monotonic()-started > args.seconds-45
            if step in (0, 300, args.steps) or deadline:
                result = evaluate(model, final=step == args.steps or deadline)
                record = {'seed': seed, 'variant': variant, 'step': step, 'budget_stop': deadline, 'metrics': result}
                report['evaluations'].append(record)
                emit({'event': 'evaluation', 'seed': seed, 'variant': variant, 'step': step,
                      'metrics': {g: {mode: {k: m[k] for k in ('recall', 'precision', 'point_f1')}
                                         for mode, m in item.items() if mode != 'threshold_diagnostics'}
                                  for g, item in result.items()}})
                (args.out/'report.json').write_text(json.dumps(report, indent=2))
            if step == args.steps or deadline:
                budget_stop |= deadline
                break
            model.train(); optimizer.zero_grad(set_to_none=True)
            x, y = training[step % len(training)]
            torch.cuda.synchronize(); t = time.monotonic()
            loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite point-scoring loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize()
            train_seconds += time.monotonic()-t; actual_step = step+1
            losses.append(float(loss.detach()))
            if actual_step % 300 == 0:
                emit({'event': 'train', 'seed': seed, 'variant': variant, 'step': actual_step,
                      'loss_ma100': sum(losses[-100:])/len(losses[-100:])})
        model.eval()
        checkpoint = args.out/f'{seed}-{variant}.pt'
        torch.save({'model': model.state_dict(), 'variant': variant, 'seed': seed,
                    'step': actual_step, 'manifest': manifest}, checkpoint)
        # Verify checkpoint round-trip with the same inference input.
        restored = PointScorer(temporal=variant == 'temporal').cuda()
        restored.load_state_dict(torch.load(checkpoint, map_location='cuda')['model']); restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(training[0][0]), restored(training[0][0]), rtol=0, atol=0)
        report['arms'].append({'seed': seed, 'variant': variant, 'steps': actual_step,
                               'training_seconds': train_seconds, 'checkpoint_roundtrip': True,
                               'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30})
        del model, restored, optimizer
        torch.cuda.empty_cache()
        if budget_stop:
            break
    if budget_stop:
        break
report['complete'] = len(report['arms']) == 4 and all(arm['steps'] == args.steps for arm in report['arms'])
report['elapsed_seconds'] = time.monotonic()-started
report['budget_stop'] = budget_stop
(args.out/'report.json').write_text(json.dumps(report, indent=2))
emit({'event': 'complete', 'complete': report['complete'], 'budget_stop': budget_stop,
      'updates': sum(arm['steps'] for arm in report['arms'])})
