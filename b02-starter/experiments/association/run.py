"""Compare association changes with frozen point scores and output rules."""
import argparse
from collections import deque
from contextlib import nullcontext
from functools import partial
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
from time import monotonic


def make_tracker(model, score_weight, budget):
    import torch
    from assignment import assign
    from b02.algorithm import process_circle
    from model import point_features, HISTORY
    device = next(model.parameters()).device

    @torch.no_grad()
    def tracker(points, state):
        budget()
        history = state.setdefault('history', deque(maxlen=HISTORY-1))
        features = point_features(points, list(history))
        history.append(points)
        scores = model(torch.as_tensor(features, device=device)).sigmoid().cpu().tolist()
        output = process_circle(points, state.setdefault('tracker', {}),
                                assignment=partial(assign, scores=scores, score_weight=score_weight))
        return output, dict(zip((p['PntID'] for p in points), scores))

    return tracker


def make_handler(model, policy, score_weight, budget):
    from policy import OutputGate

    tracker = make_tracker(model, score_weight, budget)

    def handler(points, state):
        output, scores = tracker(points, state)
        gate = state.setdefault('gate', OutputGate(policy))
        return gate.apply(output, scores, state['tracker']['scan']-1)

    return handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=1140)
    args = parser.parse_args()
    sys.path[:0] = [str(args.project), str(args.project/'experiments/point_scorer'),
                    str(args.project/'experiments/point_scorer_expanded')]
    import torch
    from b02.replay import open_scenes, replay_scene
    from b02.scorecard import build_scorecard, load_summary, write_scorecard
    from model import PointScorer
    from policy import Policy

    torch.set_num_threads(4)
    started = monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.reference/'manifest.json').read_text())
    split = json.loads((args.project/'splits/v1.json').read_text())
    assert set(manifest['dev_scenes']) == set(split['dev'])
    assert set(manifest['calibration_scenes']) <= set(split['train'])
    assert not set(manifest['calibration_scenes']) & set(manifest['fit_scenes'])
    digest = hashlib.sha256()
    with args.data.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == manifest['data_sha256']
    assert hashlib.sha256((args.project/'experiments/point_scorer/model.py').read_bytes()).hexdigest() == manifest['source_sha256']['model.py']
    policy = Policy(**json.loads((args.reference/'selection.json').read_text())['selected'])
    weights = {'global_geometry': 0.0, 'global_score': .50}
    seeds = manifest['seeds']
    models = {}
    for seed in seeds:
        saved = torch.load(args.reference/f'{seed}-temporal.pt', map_location='cpu')
        model = PointScorer().cuda()
        model.load_state_dict(saved['model'])
        models[seed] = model.eval()

    def save(name, value):
        path = args.out/name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    def log(**values):
        values['seconds'] = monotonic()-started
        with (args.out/'events.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(values)+'\n')
        print(json.dumps(values), flush=True)

    def budget():
        if monotonic()-started > args.seconds:
            raise TimeoutError('Soft budget reached')

    class DiscardTrace:
        def write(self, value):
            pass

    def replay_variant(phase, scenes, variant, seed):
        out = args.out/phase/f'{variant}_{seed}'
        out.mkdir(parents=True)
        reference = load_summary(args.reference/phase/f'expanded_{seed}__{policy.name}')
        baseline = load_summary(args.reference/phase/'baseline')
        summary = {'kind': 'local_diagnostics', 'config': {'variant': variant, 'seed': seed}, 'scenes': {}}
        handler = make_handler(models[seed], policy, weights[variant], budget)

        with open_scenes(args.data) as (_, opener):
            for scene in scenes:
                budget()
                trace = ((out/f'{scene}.jsonl').open('w', encoding='utf-8')
                         if phase == 'dev' and seed == seeds[0] and scene in ('train_015', 'train_123')
                         else nullcontext(DiscardTrace()))
                with opener(scene) as source, trace as sink:
                    summary['scenes'][scene] = replay_scene(source, sink, handler)
                save(out.relative_to(args.out)/'summary.json', summary)
                log(event='scene', phase=phase, variant=variant, seed=seed, scene=scene,
                    completed=len(summary['scenes']), total=len(scenes))
        card = build_scorecard(summary, out.name, reference, '现有输出过滤方案')
        write_scorecard(out, card)
        return {'candidate': card['overall']['values'],
                'current': card['overall']['baseline'],
                'baseline': build_scorecard(baseline, 'baseline')['overall']['values']}

    def aggregate(rows, key):
        return {metric: mean(row[key][metric] for row in rows)
                for metric in ('point_recall', 'continuous_score', 'clutter_outputs', 'weighted_total')}

    save('manifest.json', {'weights': weights, 'seeds': seeds, 'policy': policy.name,
                          'calibration': manifest['calibration_scenes'], 'dev': manifest['dev_scenes'],
                          'data_sha256': manifest['data_sha256'], 'soft_seconds': args.seconds,
                          'selection': 'Prefer recall and continuity gains with clutter <= 110% of current. '
                                       'Among feasible choose highest local total. Otherwise test highest total '
                                       'within clutter cap, or smallest clutter increase if all exceed it.'})
    report = {'complete': False, 'calibration': {}}
    try:
        for variant in weights:
            rows = [replay_variant('calibration', manifest['calibration_scenes'], variant, seed) for seed in seeds]
            report['calibration'][variant] = {key: aggregate(rows, key) for key in rows[0]}
        def rank(variant):
            group = report['calibration'][variant]
            new, old = group['candidate'], group['current']
            cap = new['clutter_outputs'] <= 1.10 * old['clutter_outputs']
            both = new['point_recall'] > old['point_recall'] and new['continuous_score'] > old['continuous_score']
            return cap, cap and both, new['weighted_total'] if cap else -new['clutter_outputs']

        chosen = max(weights, key=rank)
        report['selected'] = chosen
        save('selection.json', {'selected': chosen, 'calibration': report['calibration']})
        log(event='frozen', selected=chosen)
        rows = [replay_variant('dev', manifest['dev_scenes'], chosen, seed) for seed in seeds]
        report['dev'] = {key: aggregate(rows, key) for key in rows[0]}
        report['dev_by_seed'] = dict(zip(seeds, rows))
        report['complete'] = True
    except TimeoutError as error:
        report['budget_stop'] = str(error)
    finally:
        report['elapsed_seconds'] = monotonic()-started
        save('report.json', report)
    log(event='complete', complete=report['complete'], dev=report.get('dev'))


if __name__ == '__main__':
    main()
