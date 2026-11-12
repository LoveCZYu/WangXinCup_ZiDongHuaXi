"""Expand real training coverage; select output policy inside train, then freeze dev."""
import argparse
from collections import Counter, deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--project', type=Path, required=True)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--plan', type=Path, required=True)
p.add_argument('--old-run', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--seconds', type=int, default=1140)
a = p.parse_args()
sys.path[:0] = [str(a.project.resolve()), str((a.project/'experiments/point_scorer').resolve())]
import numpy as np
import torch
from torch.nn import functional as F
from model import PointScorer, point_features, HISTORY
from policy import OutputGate, Policy, POLICIES, check
from b02.algorithm import process_circle
from b02.replay import open_scenes, _scans, _identity, _checked_outputs
from b02.scorecard import build_scorecard, write_scorecard

torch.set_num_threads(4)
started = time.monotonic()
a.out.mkdir(parents=True, exist_ok=False)
source = a.out/'source'; source.mkdir()
for path in [*Path(__file__).parent.glob('*.py'), a.project/'experiments/point_scorer/model.py',
             *(a.project/'b02'/name for name in ('algorithm.py', 'replay.py', 'scorecard.py'))]:
    shutil.copy2(path, source/path.name)
plan = json.loads(a.plan.read_text())
split = json.loads((a.project/'splits/v1.json').read_text())
fit, calibration, dev = (plan[k] for k in ('fit_scenes', 'calibration_scenes', 'dev_scenes'))
assert set(fit+calibration) <= set(split['train']) and set(dev) == set(split['dev'])
assert not set(fit)&set(calibration) and not set(fit+calibration+dev)&set(split['holdout'])
assert not set(calibration)&{'train_001','train_002','train_004','train_005'}
manifest = {**plan, 'policies': [asdict(x) for x in POLICIES], 'soft_seconds': a.seconds,
            'optimizer': {'name': 'Adam', 'lr': .001, 'clip_norm': 10.},
            'selection': {'data': 'training-only calibration, two-seed mean',
                          'recall_max_drop': .05, 'continuous_score_max_drop': 5.,
                          'clutter_max_ratio': .5, 'objective': 'maximum local weighted total among feasible policies',
                          'fallback': 'if none feasible, maximum weighted total with failed_constraints flag'},
            'dev_use': 'All 30 complete files exactly once after common policy frozen; no retuning.',
            'data_sha256': hashlib.sha256(a.data.read_bytes()).hexdigest(),
            'source_sha256': {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in source.iterdir()},
            'torch': torch.__version__, 'numpy': np.__version__}

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def emit(event):
    event['elapsed_seconds'] = time.monotonic()-started
    with (a.out/'events.jsonl').open('a') as f:
        f.write(json.dumps(event)+'\n')
    print(json.dumps(event), flush=True)

def budget():
    if time.monotonic()-started > a.seconds:
        raise TimeoutError('Expanded experiment soft budget exhausted')

save(a.out/'manifest.json', manifest)
report = {'complete': False, 'training': [], 'calibration_complete': False, 'dev_complete': False}
check()

def read_scene(opener, name):
    with opener(name) as file:
        return list(_scans(file))

def train():
    xs, ys = [], []; counts = Counter(); clips = []
    with open_scenes(a.data) as (_, opener):
        for name in fit:
            budget(); frames = read_scene(opener, name)
            assert all(c == i for i, (c, _, _) in enumerate(frames))
            starts = np.linspace(0, len(frames)-plan['frames_per_clip'], plan['clips_per_scene'], dtype=int)
            assert len(set(starts)) == plan['clips_per_scene']
            for start in starts:
                clips.append([name, int(start), plan['frames_per_clip']])
                for circle in range(start, start+plan['frames_per_clip']):
                    _, points, labels = frames[circle]
                    features = point_features(points, [x[1] for x in frames[max(0,circle-5):circle]])
                    label_array = np.asarray([labels[x['PntID']] for x in points])
                    counts['positive'] += int((label_array > 0).sum())
                    counts['negative'] += int((label_array == 0).sum())
                    counts['unknown_ignored'] += int((label_array < 0).sum())
                    keep = label_array >= 0
                    xs.append(features[keep]); ys.append((label_array[keep] > 0).astype(np.float32))
                    counts['frames'] += 1
            emit({'event': 'features', 'scene': name, 'frames_so_far': counts['frames']})
    x = torch.as_tensor(np.concatenate(xs), device='cuda')
    y = torch.as_tensor(np.concatenate(ys), device='cuda')
    del xs, ys, frames
    manifest['training_counts'] = dict(counts)
    manifest['training_clips'] = clips
    manifest['positive_weight'] = counts['negative']/counts['positive']
    save(a.out/'manifest.json', manifest)
    weight = torch.tensor(manifest['positive_weight'], device='cuda')
    for seed in plan['seeds']:
        budget(); torch.manual_seed(seed)
        model = PointScorer(temporal=True).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        torch.cuda.reset_peak_memory_stats(); began = time.monotonic(); losses = []
        for step in range(1, plan['steps_per_seed']+1):
            budget(); model.train(); optimizer.zero_grad(set_to_none=True)
            index = torch.randint(len(x), (plan['batch_points'],), device='cuda')
            loss = F.binary_cross_entropy_with_logits(model(x[index]), y[index], pos_weight=weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite expanded point-scoring loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
            if step % 1000 == 0:
                emit({'event': 'train', 'seed': seed, 'step': step, 'loss_ma100': sum(losses[-100:])/100})
        checkpoint = a.out/f'{seed}-temporal.pt'
        torch.save({'model': model.state_dict(), 'seed': seed, 'step': step, 'variant': 'temporal', 'manifest': manifest}, checkpoint)
        model.eval(); restored = PointScorer().cuda()
        restored.load_state_dict(torch.load(checkpoint, map_location='cuda')['model']); restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x[:256]), restored(x[:256]), rtol=0, atol=0)
        report['training'].append({'seed': seed, 'step': step, 'seconds': time.monotonic()-began,
                                   'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                                   'checkpoint_roundtrip': True})
        del model, optimizer, restored
        torch.cuda.empty_cache()
    del x, y, weight
    torch.cuda.empty_cache()
    save(a.out/'report.json', report)

def load_models(include_old):
    models = {}
    for seed in plan['seeds']:
        paths = [('expanded', a.out/f'{seed}-temporal.pt')]
        if include_old:
            paths.append(('small', a.old_run/f'{seed}-temporal.pt'))
        for kind, path in paths:
            saved = torch.load(path, map_location='cpu')
            assert saved['manifest']['source_sha256']['model.py'] == manifest['source_sha256']['model.py']
            model = PointScorer().cuda(); model.load_state_dict(saved['model']); model.eval()
            models[f'{kind}_{seed}'] = model
    return models

def finalize_accumulator(acc, truth, circles, points_count):
    targets = []
    for label, meta in sorted(truth.items()):
        found = acc['ids'].get(label, Counter())
        targets.append({'truth_id': label, 'truth_points': meta['count'], 'matched_points': sum(found.values()),
                        'output_ids': dict(sorted(found.items())),
                        'first_output_delay_circles': acc['first'][label]-meta['first'] if label in acc['first'] else None,
                        'most_matched_points_with_one_id': max(found.values(), default=0)})
    times = sorted(acc['times']); counts = dict(acc['counts']); n = sum(t['count'] for t in truth.values())
    return {**counts, 'circles': circles, 'points': points_count, 'truth_points': n,
            'target_point_recall': counts['matched_target_points']/n if n else None,
            'truth_tracks': len(targets), 'matched_truth_tracks': sum(t['matched_points'] > 0 for t in targets),
            'split_truth_tracks': sum(len(t['output_ids']) > 1 for t in targets),
            'p95_handler_ms': times[(95*len(times)-1)//100], 'max_handler_ms': max(times), 'per_target': targets}

@torch.no_grad()
def evaluate(names, phase, policies_by_model):
    models = load_models(include_old=phase == 'dev')
    variants = {'baseline': (None, None)}
    for key, policies in policies_by_model.items():
        for policy in policies:
            variants[f'{key}__{policy.name}'] = (key, policy)
    summaries = {name: {'kind': 'local_diagnostics', 'config': {'variant': name, 'phase': phase}, 'scenes': {}} for name in variants}
    root = a.out/phase; root.mkdir()
    with open_scenes(a.data) as (_, opener):
        for name in names:
            budget(); history = deque(maxlen=HISTORY-1); base_state = {}; truth = {}; points_count = circles = 0
            accumulators = {key: {'counts': Counter(outputs=0, matched_target_points=0, clutter_outputs=0, unknown_outputs=0, empty_circles=0),
                                  'ids': {}, 'first': {}, 'times': []} for key in variants}
            gates = {key: OutputGate(policy) for key, (_, policy) in variants.items() if policy is not None}
            trace_files = {}
            if phase == 'dev' and name in ('train_015', 'train_016'):
                for key in variants:
                    if key == 'baseline' or '20261112' in key:
                        directory = root/key; directory.mkdir(exist_ok=True)
                        trace_files[key] = (directory/f'{name}.jsonl').open('w')
            try:
                with opener(name) as file:
                    for circle, points, labels in _scans(file):
                        budget(); circles += 1; points_count += len(points)
                        for label in labels.values():
                            if label > 0:
                                meta = truth.setdefault(label, {'first': circle, 'count': 0}); meta['count'] += 1
                        t = time.monotonic(); original = process_circle(points, base_state); base_ms = (time.monotonic()-t)*1000
                        _checked_outputs(original, {_identity(p) for p in points})
                        t = time.monotonic(); features = point_features(points, list(history)); history.append(points)
                        tensor = torch.as_tensor(features, device='cuda'); feature_ms = (time.monotonic()-t)*1000
                        scores = {}; model_ms = {}
                        for key, model in models.items():
                            t = time.monotonic(); predicted = model(tensor).sigmoid().cpu().numpy()
                            scores[key] = {p['PntID']: float(s) for p,s in zip(points, predicted)}
                            model_ms[key] = (time.monotonic()-t)*1000
                        originals = {(p['PntID'], p['BatchID']) for p in original}
                        for key, (model_key, policy) in variants.items():
                            if model_key is None:
                                output = original; duration = base_ms
                            else:
                                t = time.monotonic(); output = gates[key].apply(original, scores[model_key], circle)
                                duration = base_ms+feature_ms+model_ms[model_key]+(time.monotonic()-t)*1000
                                assert all((p['PntID'],p['BatchID']) in originals for p in output)
                            acc = accumulators[key]; acc['times'].append(duration)
                            acc['counts']['outputs'] += len(output); acc['counts']['empty_circles'] += not points
                            for point in output:
                                label = labels[point['PntID']]
                                if label > 0:
                                    acc['counts']['matched_target_points'] += 1
                                    acc['ids'].setdefault(label, Counter())[point['BatchID']] += 1
                                    acc['first'].setdefault(label, circle)
                                else:
                                    acc['counts']['clutter_outputs' if label == 0 else 'unknown_outputs'] += 1
                            if key in trace_files:
                                trace_files[key].write(json.dumps({'circle': circle, 'milliseconds': duration, 'tracks': output})+'\n')
            finally:
                for f in trace_files.values(): f.close()
            for key, acc in accumulators.items():
                summaries[key]['scenes'][name] = finalize_accumulator(acc, truth, circles, points_count)
                # Shared evaluator must reproduce the earlier complete replays.
                if phase == 'dev' and name in ('train_015', 'train_016') and (key == 'baseline' or key.startswith('small_')):
                    old_variant = 'baseline' if key == 'baseline' else key.split('__')[0].split('_')[1]+'-temporal'
                    previous = json.loads((a.old_run/'full-dev-postfilter'/old_variant/'summary.json').read_text())['scenes'][name]
                    actual = json.loads(json.dumps(summaries[key]['scenes'][name]))
                    for field in previous:
                        if field not in ('p95_handler_ms', 'max_handler_ms'):
                            assert actual[field] == previous[field], (name, key, field, 'reference replay mismatch')
                directory = root/key; directory.mkdir(exist_ok=True)
                save(directory/'summary.json', summaries[key])
            emit({'event': 'scene_evaluated', 'phase': phase, 'scene': name, 'completed_scenes': len(summaries['baseline']['scenes']),
                  'total_scenes': len(names), 'circles': circles, 'points': points_count})
    result = {}
    for key, summary in summaries.items():
        card = build_scorecard(summary, key, summaries['baseline'] if key != 'baseline' else None,
                               'baseline' if key != 'baseline' else None)
        write_scorecard(root/key, card)
        counts = {field: sum(s[field] for s in summary['scenes'].values()) for field in
                  ('truth_points', 'matched_target_points', 'clutter_outputs', 'unknown_outputs', 'circles', 'points')}
        known = counts['matched_target_points']+counts['clutter_outputs']
        result[key] = {**counts, 'recall': counts['matched_target_points']/counts['truth_points'],
                       'precision': counts['matched_target_points']/known if known else 0,
                       'point_f1': 2*counts['matched_target_points']/(counts['truth_points']+known),
                       'local_scorecard': card['overall']['values']}
    save(root/'comparison.json', result)
    del models
    torch.cuda.empty_cache()
    return result

try:
    train()
    calibration_result = evaluate(calibration, 'calibration', {f'expanded_{s}': POLICIES for s in plan['seeds']})
    report['calibration_complete'] = True
    reference = calibration_result['baseline']; choices = []
    for policy in POLICIES:
        rows = [calibration_result[f'expanded_{s}__{policy.name}'] for s in plan['seeds']]
        recall = sum(r['recall'] for r in rows)/len(rows)
        clutter = sum(r['clutter_outputs'] for r in rows)/len(rows)
        continuous = sum(r['local_scorecard']['continuous_score'] for r in rows)/len(rows)
        score = sum(r['local_scorecard']['weighted_total'] for r in rows)/len(rows)
        feasible = (recall >= reference['recall']-.05 and clutter <= reference['clutter_outputs']*.5 and
                    continuous >= reference['local_scorecard']['continuous_score']-5.)
        choices.append({'policy': asdict(policy), 'recall': recall, 'clutter': clutter,
                        'continuous_score': continuous, 'weighted_total': score, 'feasible': feasible})
    eligible = [row for row in choices if row['feasible']]
    best = max(eligible or choices, key=lambda row: (row['weighted_total'], row['recall'], row['policy']['name']))
    selected = Policy(**best['policy'])
    selection = {'selected': asdict(selected), 'failed_constraints': not eligible,
                 'baseline': reference, 'all_candidates': choices, 'selected_before_dev': True}
    save(a.out/'selection.json', selection)
    report['selection'] = selection
    emit({'event': 'policy_frozen', 'policy': asdict(selected), 'constraints_passed': bool(eligible)})
    fixed = Policy('fixed_0.5', .5, .5)
    dev_policies = {f'small_{s}': [fixed] for s in plan['seeds']}
    dev_policies.update({f'expanded_{s}': list({p.name:p for p in (fixed, selected)}.values()) for s in plan['seeds']})
    report['dev'] = evaluate(dev, 'dev', dev_policies)
    report['dev_complete'] = report['complete'] = True
except TimeoutError as error:
    report['budget_stop'] = str(error)
finally:
    report['elapsed_seconds'] = time.monotonic()-started
    save(a.out/'report.json', report)
emit({'event': 'complete', 'complete': report['complete'], 'elapsed_seconds': report['elapsed_seconds']})
