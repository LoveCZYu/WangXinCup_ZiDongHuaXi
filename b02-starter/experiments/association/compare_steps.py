"""Compare two checkpoints from each training trajectory on internal calibration."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
from time import monotonic


class DiscardTrace:
    def write(self, value):
        pass


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'data', 'training-run', 'association-run', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=450)
    args = parser.parse_args()
    sys.path[:0] = [str(args.project.resolve()), str((args.project/'experiments/point_scorer').resolve()),
                    str((args.project/'experiments/point_scorer_expanded').resolve())]
    import numpy as np
    import torch
    from torch.nn import functional as F
    from b02.replay import open_scenes, _scans, replay_scene
    from b02.scorecard import build_scorecard, load_summary, write_scorecard
    from model import PointScorer, point_features
    from policy import Policy
    from experiments.association.run import make_handler

    torch.set_num_threads(4)
    started = monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    training = json.loads((args.training_run/'manifest.json').read_text())
    association = json.loads((args.association_run/'manifest.json').read_text())
    chosen = json.loads((args.association_run/'selection.json').read_text())['selected']
    policy = Policy(**json.loads((args.training_run/'selection.json').read_text())['selected'])
    split = json.loads((args.project/'splits/v1.json').read_text())
    fit, calibration = training['fit_scenes'], training['calibration_scenes']
    assert not set(fit) & set(calibration) and set(fit+calibration) <= set(split['train'])
    assert calibration == association['calibration'] and training['seeds'] == association['seeds']
    assert training['data_sha256'] == association['data_sha256'] == sha256(args.data)
    assert training['source_sha256']['model.py'] == sha256(args.project/'experiments/point_scorer/model.py')
    assert policy.name == association['policy'] and chosen == 'global_score'
    score_weight = association['weights'][chosen]
    steps = (4000, 8000)
    metrics = ('point_recall', 'continuous_score', 'clutter_outputs', 'weighted_total', 'unknown_outputs')
    manifest = {'data_sha256': training['data_sha256'], 'fit': fit, 'calibration': calibration,
                'training_clips': training['training_clips'], 'seeds': training['seeds'],
                'steps': steps, 'batch_points': training['batch_points'], 'optimizer': training['optimizer'],
                'association': chosen, 'score_weight': score_weight, 'output_rule': vars(policy),
                'soft_seconds': args.seconds, 'torch': str(torch.__version__),
                'source_sha256': {name: sha256(Path(__file__).with_name(name))
                                  for name in ('compare_steps.py', 'run.py', 'assignment.py')}}
    save(args.out/'manifest.json', manifest)

    def log(event, **fields):
        row = {'event': event, 'seconds': monotonic()-started, **fields}
        with (args.out/'events.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    def budget():
        if monotonic()-started > args.seconds:
            raise TimeoutError('Calibration experiment soft budget reached')

    clips = defaultdict(list)
    for scene, start, size in training['training_clips']:
        clips[scene].append((start, size))
    assert list(clips) == fit
    report = {'complete': False, 'reproduction': {}, 'by_seed': {}}
    try:
        xs, ys, counts = [], [], Counter()
        with open_scenes(args.data) as (_, opener):
            for scene in fit:
                budget()
                with opener(scene) as source:
                    frames = list(_scans(source))
                assert all(circle == i for i, (circle, _, _) in enumerate(frames))
                for start, size in clips[scene]:
                    for circle in range(start, start+size):
                        _, points, labels = frames[circle]
                        features = point_features(points, [f[1] for f in frames[max(0,circle-5):circle]])
                        label_array = np.asarray([labels[p['PntID']] for p in points])
                        counts['positive'] += int((label_array > 0).sum())
                        counts['negative'] += int((label_array == 0).sum())
                        counts['unknown_ignored'] += int((label_array < 0).sum())
                        keep = label_array >= 0
                        xs.append(features[keep])
                        ys.append((label_array[keep] > 0).astype(np.float32))
                        counts['frames'] += 1
        assert dict(counts) == training['training_counts']
        x = torch.as_tensor(np.concatenate(xs), device='cuda')
        y = torch.as_tensor(np.concatenate(ys), device='cuda')
        weight = torch.tensor(training['positive_weight'], device='cuda')
        del xs, ys, frames
        log('training_data_ready', **counts)
        for seed in training['seeds']:
            torch.manual_seed(seed)
            model = PointScorer(temporal=True).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=training['optimizer']['lr'])
            original = torch.load(args.training_run/f'{seed}-temporal.pt', map_location='cuda')['model']
            losses = []
            for step in range(1, steps[-1]+1):
                budget()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                index = torch.randint(len(x), (training['batch_points'],), device='cuda')
                loss = F.binary_cross_entropy_with_logits(model(x[index]), y[index], pos_weight=weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite point-scoring loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training['optimizer']['clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                if step % 1000 == 0:
                    log('train', seed=seed, step=step, loss_ma100=mean(losses[-100:]))
                if step not in steps:
                    continue
                state = model.state_dict()
                if step == steps[0]:
                    report['reproduction'][seed] = all(torch.equal(value, original[key]) for key, value in state.items())
                    log('reference_weights', seed=seed, exact=report['reproduction'][seed])
                torch.save({'model': state, 'optimizer': optimizer.state_dict(), 'seed': seed,
                            'step': step, 'variant': 'temporal', 'manifest': manifest,
                            'cpu_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state()},
                           args.out/f'{seed}-step{step}.pt')
            del model, optimizer, original, state
        del x, y, weight
        torch.cuda.empty_cache()
        for seed in training['seeds']:
            previous = load_summary(args.association_run/'calibration'/f'{chosen}_{seed}')
            baseline = load_summary(args.training_run/'calibration'/'baseline')
            report['by_seed'][seed] = {}
            for step in steps:
                out = args.out/f'{seed}-step{step}'
                out.mkdir()
                checkpoint = torch.load(args.out/f'{seed}-step{step}.pt', map_location='cpu')
                model = PointScorer(temporal=True).cuda()
                model.load_state_dict(checkpoint['model'])
                handler = make_handler(model.eval(), policy, score_weight, budget)
                summary = {'kind': 'local_diagnostics', 'config': {'seed': seed, 'step': step}, 'scenes': {}}
                with open_scenes(args.data) as (_, opener):
                    for scene in calibration:
                        budget()
                        with opener(scene) as source:
                            result = replay_scene(source, DiscardTrace(), handler)
                        if step == steps[0] and report['reproduction'][seed]:
                            expected = previous['scenes'][scene]
                            for key, value in json.loads(json.dumps(result)).items():
                                if key not in ('p95_handler_ms', 'max_handler_ms'):
                                    assert value == expected[key], (seed, scene, key)
                        summary['scenes'][scene] = result
                        save(out/'summary.json', summary)
                        log('scene', seed=seed, step=step, scene=scene, completed=len(summary['scenes']))
                card = build_scorecard(summary, out.name, baseline, '原基线')
                write_scorecard(out, card)
                row = {key: card['overall']['values'][key] for key in metrics}
                report['by_seed'][seed][step] = row
                log('calibration', seed=seed, step=step, **row)
                del model, handler, checkpoint
        report['means'] = {step: {key: mean(r[step][key] for r in report['by_seed'].values()) for key in metrics}
                           for step in steps}
        report['baseline'] = {key: build_scorecard(baseline, 'baseline')['overall']['values'][key] for key in metrics}
        report['complete'] = True
    except TimeoutError as error:
        report['budget_stop'] = str(error)
    finally:
        report['elapsed_seconds'] = monotonic()-started
        save(args.out/'report.json', report)
    log('complete', complete=report['complete'], means=report.get('means'))


if __name__ == '__main__':
    main()
