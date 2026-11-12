"""Evaluate the frozen 8000-step policy on all development scenes, without tuning."""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
from statistics import mean
import sys
from time import monotonic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'data', 'steps-run', 'training-run', 'association-run', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=1100)
    args = parser.parse_args()
    args.project = args.project.resolve()
    sys.path[:0] = [str(args.project), str(args.project/'experiments/point_scorer'),
                    str(args.project/'experiments/point_scorer_expanded')]
    import torch
    from b02.replay import open_scenes, replay_scene
    from b02.scorecard import build_scorecard, load_summary, write_scorecard
    from model import PointScorer
    from policy import Policy
    from experiments.association.compare_steps import DiscardTrace, save, sha256
    from experiments.association.run import make_handler

    torch.set_num_threads(4)
    started = monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    training = json.loads((args.training_run/'manifest.json').read_text())
    steps = json.loads((args.steps_run/'manifest.json').read_text())
    steps_report = json.loads((args.steps_run/'report.json').read_text())
    association = json.loads((args.association_run/'manifest.json').read_text())
    split = json.loads((args.project/'splits/v1.json').read_text())
    scenes, seeds = split['dev'], steps['seeds']
    policy = Policy('step8000_enter050_maintain005', .50, .05, 3)
    assert len(scenes) == 30 and len(set(scenes)) == 30
    assert set(scenes) == set(training['dev_scenes']) == set(association['dev'])
    assert not set(scenes) & set(steps['fit'] + steps['calibration'] + split['holdout'])
    assert steps_report['complete'] and all(steps_report['reproduction'].values())
    assert seeds == training['seeds'] == association['seeds']
    assert steps['association'] == 'global_score' and steps['score_weight'] == .50
    assert steps['output_rule'] == dict(name='history_05_015', enter=.50, maintain=.15, grace=3)
    assert sha256(args.data) == training['data_sha256'] == steps['data_sha256']
    assert sha256(args.project/'experiments/point_scorer/model.py') == training['source_sha256']['model.py']
    assert sha256(Path(__file__).with_name('assignment.py')) == steps['source_sha256']['assignment.py']
    baseline = load_summary(args.training_run/'dev/baseline')
    assert set(baseline['scenes']) == set(scenes)
    references = {seed: load_summary(args.association_run/'dev'/f'global_score_{seed}') for seed in seeds}
    assert all(set(ref['scenes']) == set(scenes) for ref in references.values())
    manifest = {'dev': scenes, 'seeds': seeds, 'step': 8000, 'policy': vars(policy),
                'association': steps['association'], 'score_weight': steps['score_weight'],
                'data_sha256': steps['data_sha256'], 'soft_seconds': args.seconds,
                'selection': 'Frozen before development evaluation; no development threshold search.',
                'references': {'baseline': str(args.training_run/'dev/baseline'),
                               '4000': str(args.association_run/'dev')},
                'viewer_scenes': ['train_015', 'train_123'],
                'source_sha256': {str(path.relative_to(args.project)): sha256(path) for path in
                                  [Path(__file__), Path(__file__).with_name('assignment.py'),
                                   Path(__file__).with_name('run.py'), args.project/'experiments/point_scorer/model.py',
                                   args.project/'experiments/point_scorer_expanded/policy.py',
                                   args.project/'b02/algorithm.py', args.project/'b02/replay.py',
                                   args.project/'b02/scorecard.py']},
                'checkpoint_sha256': {str(seed): sha256(args.steps_run/f'{seed}-step8000.pt') for seed in seeds}}
    save(args.out/'manifest.json', manifest)
    report = {'complete': False, 'baseline': build_scorecard(baseline, 'baseline')['overall']['values'],
              'by_seed': {}, 'reference_reproduced': {}}

    def budget():
        if monotonic()-started > args.seconds:
            raise TimeoutError('Development evaluation soft budget reached')

    def log(**fields):
        row = {'seconds': monotonic()-started, **fields}
        with (args.out/'events.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    try:
        with open_scenes(args.data) as (_, opener):
            for seed in seeds:
                ref = references[seed]
                old = torch.load(args.training_run/f'{seed}-temporal.pt', map_location='cpu')['model']
                reproduced = torch.load(args.steps_run/f'{seed}-step4000.pt', map_location='cpu')
                assert all(torch.equal(value, reproduced['model'][key]) for key, value in old.items())
                model = PointScorer(temporal=True).cuda()
                model.load_state_dict(reproduced['model'])
                reference_handler = make_handler(model.eval(), Policy(**steps['output_rule']), .50, budget)
                with opener('train_015') as source:
                    observed = replay_scene(source, DiscardTrace(), reference_handler)
                assert all(value == ref['scenes']['train_015'][key] for key, value in
                           json.loads(json.dumps(observed)).items() if key not in ('p95_handler_ms', 'max_handler_ms'))
                report['reference_reproduced'][seed] = True
                checkpoint = torch.load(args.steps_run/f'{seed}-step8000.pt', map_location='cpu')
                assert checkpoint['seed'] == seed and checkpoint['step'] == 8000
                assert json.loads(json.dumps(checkpoint['manifest'])) == steps
                model.load_state_dict(checkpoint['model'])
                handler = make_handler(model.eval(), policy, .50, budget)
                out = args.out/f'{seed}-step8000-maintain005'
                out.mkdir()
                summary = {'kind': 'local_diagnostics', 'config': {'seed': seed, 'step': 8000,
                           'policy': vars(policy), 'score_weight': .50}, 'scenes': {}}
                for scene in scenes:
                    budget()
                    trace = ((out/f'{scene}.jsonl').open('w', encoding='utf-8')
                             if seed == seeds[0] and scene in manifest['viewer_scenes']
                             else nullcontext(DiscardTrace()))
                    with opener(scene) as source, trace as sink:
                        summary['scenes'][scene] = replay_scene(source, sink, handler)
                    save(out/'summary.json', summary)
                    log(event='scene', seed=seed, scene=scene, completed=len(summary['scenes']), total=len(scenes))
                card = build_scorecard(summary, out.name, ref, f'4000-{seed}')
                write_scorecard(out, card)
                # Checks all scene signatures against the original baseline as well.
                build_scorecard(summary, out.name, baseline, 'baseline')
                report['by_seed'][seed] = {'8000': card['overall']['values'], '4000': card['overall']['baseline']}
                save(args.out/'report.json', report)
                del model, handler, reference_handler, checkpoint, old, reproduced
        keys = tuple(report['baseline'])
        report['means'] = {step: {key: mean(row[step][key] for row in report['by_seed'].values()) for key in keys}
                           for step in ('4000', '8000')}
        report['per_scene'] = {}
        for scene in scenes:
            variants = [load_summary(args.out/f'{seed}-step8000-maintain005') for seed in seeds]
            report['per_scene'][scene] = {step: {key: mean(build_scorecard(
                {'scenes': {scene: summary['scenes'][scene]}}, 'scene')['overall']['values'][key]
                for summary in (variants if step == '8000' else list(references.values())))
                for key in ('point_recall', 'continuous_score', 'clutter_outputs', 'weighted_total')}
                for step in ('4000', '8000')}
        with (args.out/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['scope', 'version', *keys])
            for version, values in [('baseline', report['baseline']), *report['means'].items()]:
                writer.writerow(['all30_mean_of_seeds', version, *(values[key] for key in keys)])
            for seed, variants in report['by_seed'].items():
                for version, values in variants.items():
                    writer.writerow([f'all30_seed{seed}', version, *(values[key] for key in keys)])
        report['complete'] = True
    except TimeoutError as error:
        report['budget_stop'] = str(error)
    finally:
        report['elapsed_seconds'] = monotonic()-started
        save(args.out/'report.json', report)
    log(event='complete', complete=report['complete'], means=report.get('means'))


if __name__ == '__main__':
    main()
