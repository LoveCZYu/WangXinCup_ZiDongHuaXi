"""Calibrate on development only and compare complete work scenes."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.candidate_onset.propagation import sha
from experiments.candidate_onset.propagation_work_summary import read
from experiments.candidate_onset.update_stream_report import TIMINGS, paired


def same_replay(observed, reference):
    assert observed['identity'] == reference['identity']
    assert {k: v for k, v in observed['summary'].items() if k not in TIMINGS} == {
        k: v for k, v in reference['summary'].items() if k not in TIMINGS}


def scene_cutoff(scores, budget):
    scores = sorted(scores)
    if not scores:
        return None
    if len(scores) > budget:
        return scores[-budget - 1]
    return float(np.nextafter(scores[0], -np.inf))


def calibrate(root, plan, digest):
    per_scene = {}
    for scene in plan['development']['scenes']:
        row = read(root / 'joint-cost-development' / f'{scene}.json')
        previous = read(root / 'stream-census-development' / f'{scene}.json')
        assert (row['scene'], row['role'], row['protocol_sha256']) == (scene, 'development', digest)
        assert previous['protocol_sha256'] == plan['previous_census_sha256']
        original = previous['variants']['census']
        same_replay(row, original)
        normal = original['operations']['eligible_correct']
        budget = int(normal * plan['false_flag_budget_per_10000'] / 10000)
        per_scene[scene] = dict(eligible_correct=normal, budget=budget,
                                cutoff=scene_cutoff(row['controls'], budget), controls=row['controls'])
    cutoff = max(r['cutoff'] for r in per_scene.values() if r['cutoff'] is not None)
    for row in per_scene.values():
        row['displaced_correct_bound'] = sum(value > cutoff for value in row.pop('controls'))
        assert row['displaced_correct_bound'] <= row['budget']
    return dict(protocol_sha256=digest, scenes=plan['development']['scenes'], cutoff=cutoff, details=per_scene)


def summarize(root, plan, digest, calibration_path):
    names = plan['work']['scenes']
    assert {path.stem for path in (root / 'joint-cost-work').glob('train_*.json')} == set(names)
    base, changed = {}, {}
    for name in names:
        original = read(root / 'stream-census-work' / f'{name}.json')
        actual = read(root / 'joint-cost-work' / f'{name}.json')
        assert original['protocol_sha256'] == plan['previous_census_sha256'] and original['scene'] == name
        assert actual['protocol_sha256'] == digest and actual['calibration_sha256'] == sha(calibration_path)
        assert actual['scene'] == name and actual['role'] == 'work'
        base[name] = original['variants']['census']
        changed[name] = actual
    baseline = paired(names, base, base, plan['guards'])
    outcome = paired(names, base, changed, plan['guards'])
    for group, rows in ((baseline, base), (outcome, changed)):
        group['short_extra_ids'] = sum(r['birth_audit']['extra_ids_1or2'] for r in rows.values())
    original_correct = baseline['operations']['eligible_correct']
    budget = int(original_correct * plan['false_flag_budget_per_10000'] / 10000)
    actual = outcome['operations'].get('displaced_correct', 0)
    outcome['checks']['false_budget'] = actual <= budget
    for name in names:
        controls = base[name]['operations']['eligible_correct']
        observed = changed[name]['operations'].get('displaced_correct', 0)
        local_budget = int(controls * plan['false_flag_budget_per_10000'] / 10000)
        outcome['scenes'][name].update(correct_update_denominator=controls, displaced_correct_budget=local_budget,
                                       false_budget_passed=observed <= local_budget)
    return dict(protocol_sha256=digest, calibration_sha256=sha(calibration_path),
                cutoff=read(calibration_path)['cutoff'], scenes=names, baseline=baseline, changed=outcome,
                budget=dict(denominator=original_correct, allowed=budget, displaced_correct=actual,
                            per_10000=actual * 10000 / original_correct),
                passed=all(outcome['checks'].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    plan, digest = read(args.protocol), sha(args.protocol)
    if args.calibration is None:
        data = calibrate(args.root, plan, digest)
        print(json.dumps(data, indent=2))
    else:
        assert read(args.calibration)['protocol_sha256'] == digest
        data = summarize(args.root, plan, digest, args.calibration)
        print(json.dumps({name: {k: v for k, v in data[name].items() if k != 'scenes'}
                          for name in ('baseline', 'changed')}, indent=2))
        print('budget', data['budget'], 'passed', data['passed'])
    args.out.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
