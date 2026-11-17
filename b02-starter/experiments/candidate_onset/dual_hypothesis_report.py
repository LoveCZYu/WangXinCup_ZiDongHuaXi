"""Compare temporal and single-scan scores with a fixed stream false flag budget."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.candidate_onset.propagation import sha

METHODS = ('single_score', 'onset_score', 'future_advantage', 'two_hypothesis_score')


def score(row, method):
    return -row['onset_gap'] if method == 'onset_score' else row[method]


def threshold(rows, method, budget):
    candidates = []
    for scene in rows:
        allowance = int(budget * scene['counts'].get('eligible_1', 0))
        controls = sorted(score(r, method) for r in scene['rows'] if r['kind'] == 1)
        if controls:
            candidates.append(controls[-allowance - 1] if len(controls) > allowance
                              else float(np.nextafter(controls[0], -np.inf)))
    if not candidates:
        raise ValueError('No correct contention updates in calibration scenes')
    return max(candidates)


def measure(scenes, method, cutoffs):
    total = Counter(eligible_correct=0, eligible_wrong_target=0, contested=0, recoverable=0, flags=0,
                    correct_flags=0, recovered_updates=0, unsupported_flags=0, unresolved_flags=0)
    possible, recovered = set(), set()
    split = {}
    for scene, cutoff in zip(scenes, cutoffs, strict=True):
        counts = Counter(eligible_correct=scene['counts'].get('eligible_1', 0),
                         eligible_wrong_target=scene['counts'].get('eligible_2', 0), contested=len(scene['rows']))
        name = scene['scene']
        for row in scene['rows']:
            key = name, *sorted(row['pair'])
            counts['recoverable'] += row['recoverable']
            if row['recoverable']:
                possible.add(key)
            if score(row, method) <= cutoff:
                continue
            counts['flags'] += 1
            if row['recoverable']:
                counts['recovered_updates'] += 1
                recovered.add(key)
            elif row['kind'] == 1:
                counts['correct_flags'] += 1
            elif row['kind'] == 0:
                counts['unresolved_flags'] += 1
            else:
                counts['unsupported_flags'] += 1
        split[name] = dict(counts, threshold=cutoff)
        total.update(counts)
    return dict(total, possible_distinct_pairs=len(possible), recovered_distinct_pairs=len(recovered),
                recovered_scenes=len({key[0] for key in recovered}), scenes=split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.protocol.read_bytes())
    digest = sha(args.protocol)
    budget = plan['false_flag_budget_per_10000'] / 10000
    groups = {}
    for role in ('development', 'work'):
        groups[role] = []
        for name in plan[role]['scenes']:
            row = json.loads((args.root / f'dual-hypothesis-{role}' / (name + '.json')).read_bytes())
            assert (row['protocol_sha256'], row['scene'], row['role']) == (digest, name, role)
            baseline = json.loads((args.root / f'stream-census-{role}' / (name + '.json')).read_bytes())
            previous = baseline['variants']['census']['operations']
            for index, key in enumerate(('unresolved', 'correct', 'other_target', 'clutter')):
                assert row['counts'].get(f'eligible_{index}', 0) == previous.get(f'eligible_{key}', 0), name
            groups[role].append(row)
    dev, work = groups['development'], groups['work']
    previous = json.loads((args.root / 'update-stream-screen-report.json').read_bytes())
    assert set(previous['work']) == set(plan['work']['scenes'])
    realized = sum(row['refit_flagged']['correct'] for row in previous['work'].values())
    pooled = dict(counts=dict(eligible_1=sum(r['counts']['eligible_1'] for r in work)),
                  rows=[row for scene in work for row in scene['rows']])
    realized_budget = (realized + .5) / pooled['counts']['eligible_1']
    result = dict(protocol_sha256=digest, previous_work_realized_correct_flags=realized, methods={})
    for method in METHODS:
        cutoff = threshold(dev, method, budget)
        folds = [threshold(dev[:i] + dev[i + 1:], method, budget) for i in range(len(dev))]
        diagnostic_cutoff = threshold([pooled], method, budget)
        realized_cutoff = threshold([pooled], method, realized_budget)
        result['methods'][method] = dict(threshold=cutoff, development_loso=measure(dev, method, folds),
                                         work=measure(work, method, [cutoff] * len(work)),
                                         work_posthoc_equal_budget=measure(work, method, [diagnostic_cutoff] * len(work)),
                                         work_posthoc_previous_realized=measure(work, method, [realized_cutoff] * len(work)))
    current = result['methods']['two_hypothesis_score']['work']
    ref = max(result['methods'][key]['work']['recovered_distinct_pairs'] for key in ('single_score', 'onset_score'))
    result['whole_scene_gate'] = {
        'false_budget': current['correct_flags'] <= int(budget * current['eligible_correct']),
        'distinct_pair_gain': current['recovered_distinct_pairs'] - ref >= plan['trigger']['min_additional_distinct_pairs'],
        'scene_coverage': current['recovered_scenes'] >= plan['trigger']['min_scenes'],
    }
    result['passed'] = all(result['whole_scene_gate'].values())
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    summary = {name: {key: value for key, value in row['work'].items() if key != 'scenes'}
               for name, row in result['methods'].items()}
    print(json.dumps(dict(methods=summary, gate=result['whole_scene_gate'], passed=result['passed']), indent=2))


if __name__ == '__main__':
    main()
