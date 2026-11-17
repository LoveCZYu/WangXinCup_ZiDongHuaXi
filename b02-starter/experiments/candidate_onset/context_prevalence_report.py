"""Measure recurrence and freeze at most one development-only write policy."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

import numpy as np

from experiments.candidate_onset.propagation import sha
from experiments.candidate_onset.propagation_work_summary import read
from experiments.candidate_onset.update_stream_report import paired


def improved(row):
    return row['wrong_reduction'] > 0 and row['correct_gain'] >= 0


def harmed(row):
    return row['wrong_reduction'] < 0 or row['correct_gain'] < 0


def counts(rows):
    return dict(tested=len(rows), improved=sum(improved(row) for row in rows), harmed=sum(harmed(row) for row in rows),
                wrong_reduction=sum(row['wrong_reduction'] for row in rows),
                correct_gain=sum(row['correct_gain'] for row in rows),
                onset_prevented=sum(row['onset_prevented'] for row in rows),
                onset_became_correct=sum(row['onset_prevented'] and row['held'][row['onset']-row['start']-1] == 'correct'
                                        for row in rows))


def describe(dataset):
    totals = {}
    for kind in ('precursor', 'control'):
        scenes = {r['scene']: counts([row for row in r['rows'] if row['kind'] == kind]) for r in dataset}
        pooled = Counter()
        for row in scenes.values():
            pooled.update(row)
        totals[kind] = dict(pooled, scenes_improved=sum(r['improved'] > 0 for r in scenes.values()), scenes=scenes)
    totals['eligible_precursors'] = sum(r['precursor_candidates'] for r in dataset)
    totals['eligible_controls'] = sum(r['normal_candidates'] for r in dataset)
    totals['scored_writes'] = sum(r['writes'] for r in dataset)
    return totals


def select_policy(development, features, plan):
    settings = plan['policy']
    ds = np.quantile(features[:, 0], settings['delta_quantiles'])
    affinities = np.quantile(features[:, 1], settings['affinity_quantiles'])
    outcomes = []
    for delta, affinity in product(ds, affinities):
        rate = float(((features[:, 0] >= delta) & (features[:, 1] <= affinity)).mean())
        if rate > settings['max_dev_hold_rate']:
            continue
        chosen = [row for r in development for row in r['rows'] if row['delta'] >= delta and row['affinity'] <= affinity]
        benefit = sum(improved(row) and row['kind'] == 'precursor' for row in chosen)
        harm = sum(harmed(row) for row in chosen)
        outcomes.append(dict(delta_min=float(delta), affinity_max=float(affinity), dev_hold_rate=rate,
                             dev_benefits=int(benefit), dev_harms=int(harm), dev_cases_selected=len(chosen)))
    if not outcomes:
        raise ValueError('No threshold pair satisfies the development write budget')
    best = max(outcomes, key=lambda x: (x['dev_benefits'] - 2*x['dev_harms'], x['dev_benefits'], -x['dev_hold_rate']))
    return best, outcomes


def load(root, role, plan, digest):
    directory = root / f'context-prevalence-audit-{role}'
    assert {p.stem for p in directory.glob('train_*.json')} == set(plan[role]['scenes'])
    rows = [read(directory / f'{name}.json') for name in plan[role]['scenes']]
    assert all(r['protocol_sha256'] == digest and r['role'] == role for r in rows)
    return rows


def audit(args, plan, digest):
    dataset = {role: load(args.root, role, plan, digest) for role in ('development', 'work')}
    scopes = {role: describe(rows) for role, rows in dataset.items()}
    work = scopes['work']['precursor']
    checks = dict(scenes=work['scenes_improved'] >= plan['prevalence']['min_scenes'],
                  fraction=work['improved'] >= work['tested']*plan['prevalence']['min_benefit_fraction'],
                  benefits=work['improved'] > work['harmed'] and work['wrong_reduction'] > 0)
    policy = dict(enabled=all(checks.values()), protocol_sha256=digest)
    report = dict(protocol_sha256=digest, scopes=scopes, checks=checks)
    if policy['enabled']:
        blocks = []
        for row in dataset['development']:
            path = args.root / 'context-prevalence-audit-development' / (row['scene']+'.npz')
            with np.load(path) as archive:
                block = archive['features']
            assert block.shape == (row['writes'], 2)
            blocks.append(block)
        best, options = select_policy(dataset['development'], np.concatenate(blocks), plan)
        policy.update(best)
        report['development_options'] = options
    args.policy.write_text(json.dumps(policy, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(work_preceding={k: v for k, v in work.items() if k != 'scenes'},
                          work_controls={k: v for k, v in scopes['work']['control'].items() if k != 'scenes'},
                          checks=checks, policy=policy), indent=2))
    return report


def full(args, plan, digest):
    assert read(args.policy)['enabled'] and read(args.policy)['protocol_sha256'] == digest
    names = plan['full_scenes']
    control, proposed = {}, {}
    for name in names:
        previous = read(args.root / 'stream-census-work' / (name+'.json'))
        assert previous['protocol_sha256'] == plan['previous_census_sha256']
        control[name] = previous['variants']['census']
        row = read(args.root / 'context-prevalence-full-work' / (name+'.json'))
        assert row['scene'] == name and row['protocol_sha256'] == digest and row['policy_sha256'] == sha(args.policy)
        proposed[name] = row
    result = dict(protocol_sha256=digest, policy_sha256=sha(args.policy), scopes={})
    for scope, used in (('all16', names), ('other15', [n for n in names if n != 'train_034'])):
        baseline = {n: control[n] for n in used}
        treatment = {n: proposed[n] for n in used}
        scores = dict(combined=paired(used, baseline, baseline, plan['guards']),
                      withheld=paired(used, baseline, treatment, plan['guards']))
        for name, group in (('combined', baseline), ('withheld', treatment)):
            scores[name]['short_extra_ids'] = sum(row['birth_audit']['extra_ids_1or2'] for row in group.values())
        result['scopes'][scope] = scores
        print(scope, {name: {k: row[k] for k in ('mixed','metrics','operations','checks')} for name, row in scores.items()})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'protocol', 'policy', 'out'):
        p.add_argument(name, type=Path)
    p.add_argument('--full', action='store_true')
    args = p.parse_args()
    plan, digest = read(args.protocol), sha(args.protocol)
    result = full(args, plan, digest) if args.full else audit(args, plan, digest)
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
