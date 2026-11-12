"""Compare lower 8000-step output thresholds on the internal calibration split."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from statistics import mean
import sys
from time import monotonic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'data', 'steps-run', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=330)
    parser.add_argument('--grid', choices=('proportional', 'separate'), default='proportional')
    args = parser.parse_args()
    sys.path[:0] = [str(args.project.resolve()), str((args.project/'experiments/point_scorer').resolve()),
                    str((args.project/'experiments/point_scorer_expanded').resolve())]
    import torch
    from b02.replay import _scans, open_scenes, replay_frames, replay_scene
    from b02.scorecard import build_scorecard, load_summary, write_scorecard
    from model import PointScorer
    from policy import OutputGate, Policy
    from experiments.association.compare_steps import DiscardTrace, save, sha256
    from experiments.association.run import make_handler, make_tracker

    torch.set_num_threads(4)
    started = monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.steps_run/'manifest.json').read_text())
    previous = json.loads((args.steps_run/'report.json').read_text())
    split = json.loads((args.project/'splits/v1.json').read_text())
    scenes, seeds = manifest['calibration'], manifest['seeds']
    assert previous['complete'] and manifest['association'] == 'global_score'
    assert len(scenes) == 8 and set(scenes) <= set(split['train'])
    assert not set(scenes) & set(manifest['fit'] + split['dev'] + split['holdout'])
    assert sha256(args.data) == manifest['data_sha256']
    assert sha256(Path(__file__).with_name('assignment.py')) == manifest['source_sha256']['assignment.py']
    original = Policy(**manifest['output_rule'])
    if args.grid == 'proportional':
        policies = [original] + [
            Policy(f'enter_{entry:.2f}', entry, round(original.maintain * entry / original.enter, 4), original.grace)
            for entry in (.45, .40, .35, .30, .25)
        ]
    else:
        policies = [original] + [
            Policy(f'enter_{entry:.2f}_maintain_{maintain:.3f}', entry, maintain, original.grace)
            for entry, maintain in ((.50,.10), (.50,.05), (.45,.10), (.45,.075), (.45,.05), (.40,.15), (.35,.15))
        ]
    display_scene = 'train_085'
    assert display_scene in scenes
    keys = ('point_recall', 'continuous_score', 'clutter_outputs', 'weighted_total', 'unknown_outputs')
    report = {'complete': False, 'reference': previous['means']['4000'],
              'reference_by_seed': {str(seed): previous['by_seed'][str(seed)]['4000'] for seed in seeds},
              'calibration': scenes, 'viewer_scene': display_scene, 'by_seed': {},
              'grid': args.grid,
              'policies': {policy.name: vars(policy) for policy in policies},
              'association': manifest['association'], 'score_weight': manifest['score_weight'],
              'data_sha256': manifest['data_sha256']}

    def budget():
        if monotonic()-started > args.seconds:
            raise TimeoutError('Threshold sweep soft budget reached')

    def log(**fields):
        fields['seconds'] = monotonic()-started
        with (args.out/'events.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(fields)+'\n')
        print(json.dumps(fields), flush=True)

    def load_model(seed, step):
        checkpoint = torch.load(args.steps_run/f'{seed}-step{step}.pt', map_location='cpu')
        assert checkpoint['seed'] == seed and checkpoint['step'] == step
        assert json.loads(json.dumps(checkpoint['manifest'])) == manifest
        model = PointScorer(temporal=True).cuda()
        model.load_state_dict(checkpoint['model'])
        return model.eval()

    def check_reference(result, expected):
        for key, value in json.loads(json.dumps(result)).items():
            if key not in ('p95_handler_ms', 'max_handler_ms'):
                assert value == expected[key], (key, value, expected[key])

    try:
        with open_scenes(args.data) as (_, opener):
            for seed in seeds:
                base = load_summary(args.steps_run/f'{seed}-step4000')
                before = load_summary(args.steps_run/f'{seed}-step8000')
                model = load_model(seed, 8000)
                tracker = make_tracker(model, manifest['score_weight'], budget)
                summaries = {p.name: {'kind': 'local_diagnostics', 'config': vars(p), 'scenes': {}} for p in policies}
                for scene in scenes:
                    with opener(scene) as source:
                        frames = list(_scans(source))
                    state = {}
                    associated = []
                    for _, points, _ in frames:
                        outputs, scores = tracker(points, state)
                        associated.append((outputs, {p['PntID']: scores[p['PntID']] for p in outputs}))
                    for policy in policies:
                        budget()
                        gate = OutputGate(policy)
                        results = iter(zip(frames, associated))

                        def handler(points, state):
                            (circle, _, _), (outputs, scores) = next(results)
                            return gate.apply(outputs, scores, circle)

                        out = args.out/f'{seed}-{policy.name}'
                        out.mkdir(exist_ok=True)
                        trace = ((out/f'{scene}.jsonl').open('w', encoding='utf-8')
                                 if seed == seeds[0] and scene == display_scene else nullcontext(DiscardTrace()))
                        with trace as sink:
                            summary = replay_frames(iter(frames), sink, handler)
                        if policy == original:
                            check_reference(summary, before['scenes'][scene])
                        summaries[policy.name]['scenes'][scene] = summary
                    log(event='scene', seed=seed, scene=scene)
                    del frames, associated
                report['by_seed'][seed] = {}
                for policy in policies:
                    out = args.out/f'{seed}-{policy.name}'
                    summary = summaries[policy.name]
                    save(out/'summary.json', summary)
                    card = build_scorecard(summary, out.name, base, '4000 steps')
                    write_scorecard(out, card)
                    report['by_seed'][seed][policy.name] = {key: card['overall']['values'][key] for key in keys}
                if seed == seeds[0]:
                    out = args.out/'reference-view'
                    out.mkdir()
                    handler = make_handler(load_model(seed, 4000), original, manifest['score_weight'], budget)
                    with opener(display_scene) as source, (out/f'{display_scene}.jsonl').open('w', encoding='utf-8') as trace:
                        observed = replay_scene(source, trace, handler)
                    check_reference(observed, base['scenes'][display_scene])
                    save(out/'summary.json', base)
                    del handler
                del tracker, model
        means = {}
        for policy in policies:
            rows = [report['by_seed'][seed][policy.name] for seed in seeds]
            averages = {key: mean(row[key] for row in rows) for key in keys}
            averages['within_each_seed_clutter_cap'] = all(
                row['clutter_outputs'] <= report['reference_by_seed'][str(seed)]['clutter_outputs']
                for seed, row in zip(seeds, rows))
            averages['improves_both_means'] = all(
                averages[key] > report['reference'][key] for key in ('point_recall', 'continuous_score'))
            means[policy.name] = averages
        report['means'] = means
        eligible = [policy.name for policy in policies[1:] if means[policy.name]['within_each_seed_clutter_cap']]
        report['selected'] = max(eligible, key=lambda name: (means[name]['improves_both_means'],
                                                           means[name]['weighted_total']), default=None)
        report['complete'] = True
    except TimeoutError as error:
        report['budget_stop'] = str(error)
    finally:
        report['elapsed_seconds'] = monotonic()-started
        save(args.out/'report.json', report)
    log(event='complete', complete=report['complete'], selected=report.get('selected'), means=report.get('means'))


if __name__ == '__main__':
    main()
