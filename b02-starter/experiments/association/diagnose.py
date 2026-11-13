"""Read-only calibration diagnosis; truth-based oracles are diagnostic only."""
import argparse
from collections import Counter, deque
from copy import deepcopy
from dataclasses import replace
from heapq import nsmallest
import json
from pathlib import Path
from statistics import mean
import sys
from time import monotonic, perf_counter


def oracle(summary, *, clean=False, identity=False):
    result = deepcopy(summary)
    for scene in result['scenes'].values():
        if clean:
            scene['outputs'] = scene['matched_target_points']
            scene['clutter_outputs'] = scene['unknown_outputs'] = 0
        if identity:
            for target in scene['per_target']:
                count = target['matched_points']
                target['output_ids'] = {str(target['truth_id']): count} if count else {}
                target['most_matched_points_with_one_id'] = count
            scene['split_truth_tracks'] = 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'data', 'steps-run', 'threshold-run', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--seed', type=int, action='append')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    args.project = args.project.resolve()
    sys.path[:0] = [str(args.project), str(args.project/'experiments/point_scorer'),
                    str(args.project/'experiments/point_scorer_expanded'), str(Path(__file__).resolve().parent)]
    import torch
    from assignment import assign
    from b02.algorithm import DEFAULT_CONFIG, POSITION, process_circle
    from b02.replay import _scans, open_scenes, replay_frames
    from b02.scorecard import build_scorecard, load_summary, write_scorecard
    from model import HISTORY, PointScorer, point_features
    from policy import OutputGate, Policy
    from experiments.association.compare_steps import DiscardTrace, save, sha256

    torch.set_num_threads(args.threads)
    started = monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.steps_run/'manifest.json').read_text())
    split = json.loads((args.project/'splits/v1.json').read_text())
    scenes, seeds = manifest['calibration'], manifest['seeds']
    if args.seed:
        assert set(args.seed) <= set(seeds)
        seeds = args.seed
    assert len(scenes) == 8 and set(scenes) <= set(split['train'])
    assert not set(scenes) & set(manifest['fit'] + split['dev'] + split['holdout'])
    assert sha256(args.data) == manifest['data_sha256']
    assert sha256(Path(__file__).with_name('assignment.py')) == manifest['source_sha256']['assignment.py']
    policy = Policy('frozen_050_005', .50, .05, 3)
    save(args.out/'manifest.json', {'scenes': scenes, 'seeds': seeds, 'step': 8000,
         'policy': vars(policy), 'data_sha256': manifest['data_sha256'], 'device': args.device,
         'soft_seconds': args.seconds, 'threads': args.threads,
         'truth_usage': 'Diagnostics and oracle scorecards only; never tracker inputs.',
         'source_sha256': {str(p.relative_to(args.project)): sha256(p) for p in
                          [Path(__file__).resolve(), args.project/'b02/algorithm.py',
                           Path(__file__).with_name('assignment.py'), args.project/'experiments/point_scorer/model.py']}})
    report = {'complete': False, 'by_seed': {}, 'counterfactuals': {}, 'examples': []}

    def budget():
        if monotonic()-started > args.seconds:
            raise TimeoutError('Diagnosis budget reached')

    def log(**fields):
        row = {'seconds': monotonic()-started, **fields}
        print(json.dumps(row), flush=True)
        with (args.out/'events.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(row)+'\n')

    try:
        with open_scenes(args.data) as (_, opener):
            for seed in seeds:
                checkpoint = torch.load(args.steps_run/f'{seed}-step8000.pt', map_location='cpu', weights_only=True)
                assert checkpoint['seed'] == seed and checkpoint['step'] == 8000
                model = PointScorer().to(args.device).eval()
                model.load_state_dict(checkpoint['model'])
                expected = load_summary(args.threshold_run/f'{seed}-enter_0.50_maintain_0.050')
                current = {'kind': 'local_diagnostics', 'scenes': {}}
                raw_summary = {'kind': 'local_diagnostics', 'scenes': {}}
                totals = Counter()
                scene_diagnostics = {}
                out = args.out/str(seed)
                out.mkdir()
                for scene_name in scenes:
                    budget()
                    with opener(scene_name) as source:
                        frames = list(_scans(source))
                    history, state, gate = deque(maxlen=HISTORY-1), {}, OutputGate(policy)
                    last_final, observations, births = {}, Counter(), {}
                    counters, switches = Counter(), Counter()
                    current_cache, raw_cache, measured_ms = [], [], []
                    examples_by_kind = Counter()
                    for circle, points, labels in frames:
                        budget()
                        inference_start = perf_counter()
                        with torch.no_grad():
                            scores = model(torch.as_tensor(point_features(points, list(history)), device=args.device)).sigmoid().cpu().tolist()
                        history.append(points)
                        score_map = dict(zip((p['PntID'] for p in points), scores))
                        active = state.get('tracks', {})
                        positive_labels = {value for value in labels.values() if value > 0}
                        prior_ids = {last_final[label]['batch'] for label in positive_labels if label in last_final}
                        before = {batch: replace(active[batch]) for batch in prior_ids if batch in active}
                        old_ids = set(active)
                        matches = {}

                        def assignment(*positional):
                            selected = assign(*positional, scores=scores, score_weight=.50)
                            matches.update(selected)
                            return selected

                        raw = process_circle(points, state, assignment=assignment)
                        strong_before = dict(gate.last_strong)
                        outputs = gate.apply(raw, score_map, circle)
                        measured_ms.append((perf_counter()-inference_start)*1000)
                        raw_cache.append(raw)
                        current_cache.append(outputs)
                        current_ids = {p['PntID']: p['BatchID'] for p in outputs}
                        raw_ids = {p['PntID']: p['BatchID'] for p in raw}
                        matched_indices = {index: batch for batch, index in matches.items()}
                        new_ids = sorted(set(state['tracks']) - old_ids)
                        free = (i for i in range(len(points)) if i not in matched_indices)
                        new_indices = nsmallest(len(new_ids), free, key=lambda i: (-points[i]['Amp'], i))
                        assert len(new_indices) == len(new_ids)
                        born_indices = set(new_indices)
                        for batch, index in zip(new_ids, new_indices):
                            births[batch] = circle
                        for p in raw:
                            label = labels[p['PntID']]
                            if label == 0:
                                counters['raw_clutter_updates'] += 1
                                counters['hidden_clutter_updates'] += p['PntID'] not in current_ids
                            elif label == -1:
                                counters['raw_unknown_updates'] += 1
                        for index, point in enumerate(points):
                            label, pid = labels[point['PntID']], point['PntID']
                            if label <= 0:
                                continue
                            observations[label] += 1
                            counters['truth_points'] += 1
                            if pid in current_ids:
                                counters['hits'] += 1
                                previous = last_final.get(label)
                                batch = current_ids[pid]
                                if previous and batch != previous['batch']:
                                    old = previous['batch']
                                    if old not in state['tracks']:
                                        kind = 'old_expired'
                                    elif old not in matches:
                                        kind = 'old_alive_unmatched'
                                    else:
                                        destination = labels[points[matches[old]]['PntID']]
                                        kind = ('old_matched_clutter' if destination == 0 else
                                                'old_matched_unknown' if destination == -1 else
                                                'old_matched_other_truth' if destination != label else 'old_matched_same_truth')
                                    switches[kind] += 1
                                    switches['all'] += 1
                                    switches['new_id_born_before_previous_hit'] += births[batch] <= previous['circle']
                                    switches['consecutive_truth_observations'] += observations[label] == previous['observation']+1
                                    if examples_by_kind[kind] < 2:
                                        report['examples'].append({'seed': seed, 'scene': scene_name, 'truth_id': label,
                                            'circle': circle, 'previous_circle': previous['circle'], 'old_id': old,
                                            'new_id': batch, 'new_id_birth': births[batch], 'kind': kind,
                                            'score': score_map[pid]})
                                        examples_by_kind[kind] += 1
                                last_final[label] = {'batch': batch, 'circle': circle, 'observation': observations[label]}
                                continue
                            counters['misses'] += 1
                            if pid in raw_ids:
                                batch = raw_ids[pid]
                                counters['miss_filter'] += 1
                                reason = 'filter_below_maintain' if score_map[pid] < policy.maintain else 'filter_no_recent_strong'
                                recent = batch in strong_before and circle-strong_before[batch] <= policy.grace
                                assert score_map[pid] < policy.enter and (not recent or score_map[pid] < policy.maintain)
                            elif index in matched_indices:
                                counters['miss_before_filter'] += 1
                                assert state['tracks'][matched_indices[index]].hits < DEFAULT_CONFIG.confirm_hits
                                reason = 'tentative_initial' if observations[label] <= 2 else 'tentative_after_start'
                            else:
                                counters['miss_before_filter'] += 1
                                reason = 'new_birth_initial' if observations[label] <= 2 and index in born_indices else (
                                         'new_birth_after_start' if index in born_indices else 'no_birth_capacity')
                            counters[reason] += 1
                            if pid not in raw_ids and label in last_final:
                                batch = last_final[label]['batch']
                                if batch not in before:
                                    old_reason = 'previous_id_absent'
                                else:
                                    track = before[batch]
                                    elapsed = circle+1-track.last_scan
                                    predicted = tuple(x+v*elapsed for x,v in zip(track.position, track.velocity))
                                    position = tuple(point[key]/scale for key,scale in zip(POSITION, DEFAULT_CONFIG.gate))
                                    radius = min(1+.2*(elapsed-1), 1.6)
                                    distance = sum((x-y)**2 for x,y in zip(predicted, position)) / radius**2
                                    old_reason = ('previous_id_outside_gate' if distance > 1 else
                                                  'previous_id_cost_rejected' if distance+.5*(1-score_map[pid]) >= 1.15
                                                  else 'previous_id_inside_gate_not_assigned')
                                counters[old_reason] += 1
                    def replay_cache(cache, sink):
                        iterator = iter(cache)
                        class MeasuredTrace:
                            def write(self, line):
                                row = json.loads(line)
                                row['milliseconds'] = round(measured_ms[row['circle']], 4)
                                sink.write(json.dumps(row)+'\n')
                        summary = replay_frames(iter(frames), MeasuredTrace(), lambda points, state: next(iterator))
                        times = sorted(measured_ms)
                        summary['p95_handler_ms'] = round(times[(95*len(times)-1)//100], 4)
                        summary['max_handler_ms'] = round(times[-1], 4)
                        return summary
                    if seed == seeds[0]:
                        with (out/f'{scene_name}.jsonl').open('w', encoding='utf-8') as sink:
                            observed = replay_cache(current_cache, sink)
                    else:
                        observed = replay_cache(current_cache, DiscardTrace())
                    raw_summary['scenes'][scene_name] = replay_cache(raw_cache, DiscardTrace())
                    for key, value in json.loads(json.dumps(observed)).items():
                        if key not in ('p95_handler_ms', 'max_handler_ms'):
                            assert value == expected['scenes'][scene_name][key], (seed, scene_name, key)
                    assert counters['miss_filter']+counters['miss_before_filter'] == counters['misses']
                    current['scenes'][scene_name] = observed
                    totals.update(counters)
                    totals.update({'switch_'+key: value for key,value in switches.items()})
                    scene_diagnostics[scene_name] = {'misses': dict(counters), 'switches': dict(switches)}
                    save(out/'summary.json', current)
                    save(out/'scene-diagnostics.json', scene_diagnostics)
                    log(event='scene', seed=seed, scene=scene_name, misses=counters['misses'],
                        filter=counters['miss_filter'], switches=switches['all'])
                variants = {'current': current, 'raw_no_filter': raw_summary,
                            'current_perfect_clutter_filter': oracle(current, clean=True),
                            'raw_perfect_filter': oracle(raw_summary, clean=True),
                            'current_perfect_id': oracle(current, identity=True),
                            'raw_perfect_filter_and_id': oracle(raw_summary, clean=True, identity=True)}
                report['counterfactuals'][seed] = {}
                for name, summary in variants.items():
                    variant_out = out/name
                    variant_out.mkdir()
                    save(variant_out/'summary.json', summary)
                    card = build_scorecard(summary, name)
                    write_scorecard(variant_out, card)
                    report['counterfactuals'][seed][name] = card['overall']['values']
                report['by_seed'][seed] = {'counts': dict(totals), 'scenes': scene_diagnostics, 'reproduced': True}
                write_scorecard(out, build_scorecard(current, f'diagnosis-{seed}'))
                save(args.out/'report.json', report)
        names = tuple(next(iter(report['counterfactuals'].values())))
        keys = ('point_recall', 'track_recall', 'clutter_outputs', 'fragmentation_score', 'timeliness_score',
                'continuous_score', 'weighted_total')
        report['means'] = {name: {key: mean(rows[name][key] for rows in report['counterfactuals'].values()) for key in keys}
                           for name in names}
        combined = Counter()
        for group in report['by_seed'].values():
            combined.update(group['counts'])
        report['combined_counts'] = dict(combined)
        report['complete'] = True
    except TimeoutError as error:
        report['budget_stop'] = str(error)
    finally:
        report['elapsed_seconds'] = monotonic()-started
        save(args.out/'report.json', report)
    log(event='complete', complete=report['complete'], means=report.get('means'))


if __name__ == '__main__':
    main()
