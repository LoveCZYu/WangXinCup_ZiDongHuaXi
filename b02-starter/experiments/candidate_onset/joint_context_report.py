"""Compare held context against both controls on the fixed internal scenes."""
import argparse
import json
from pathlib import Path

from experiments.candidate_onset.propagation import sha
from experiments.candidate_onset.propagation_work_summary import read
from experiments.candidate_onset.update_stream_report import paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    plan = read(args.protocol)
    digest = sha(args.protocol)
    names = plan['scenes']
    assert {p.stem for p in (args.root / 'joint-context-work').glob('train_*.json')} == set(names)
    base, joint, context = {}, {}, {}
    for name in names:
        control = read(args.root / 'stream-census-work' / f'{name}.json')
        assert control['scene'] == name and control['protocol_sha256'] == plan['previous_census_sha256']
        base[name] = control['variants']['census']
        joint[name] = read(args.root / 'joint-cost-work' / f'{name}.json')
        assert joint[name]['scene'] == name and joint[name]['protocol_sha256'] == plan['joint_cost_protocol_sha256']
        assert joint[name]['calibration_sha256'] == plan['calibration_sha256']
        context[name] = read(args.root / 'joint-context-work' / f'{name}.json')
        assert context[name]['scene'] == name and context[name]['protocol_sha256'] == digest
    report = dict(protocol_sha256=digest, scopes={})
    for scope, members in (('all16', names), ('other15', [n for n in names if n != 'train_034']), ('train_034', ['train_034'])):
        reference = {n: base[n] for n in members}
        denominator = sum(r['operations']['eligible_correct'] for r in reference.values())
        budget = int(denominator * plan['false_flag_budget_per_10000'] / 10000)
        models = {}
        for name, rows in (('combined', base), ('joint', joint), ('context_hold', context)):
            outcome = paired(members, reference, {n: rows[n] for n in members}, plan['guards'])
            outcome['short_extra_ids'] = sum(rows[n]['birth_audit']['extra_ids_1or2'] for n in members)
            errors = outcome['operations'].get('displaced_correct', 0)
            outcome['budget'] = dict(correct_displaced=errors, allowed=budget, denominator=denominator)
            outcome['checks']['false_budget'] = errors <= budget
            models[name] = outcome
        report['scopes'][scope] = models
    args.out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    for scope, models in report['scopes'].items():
        for name, row in models.items():
            print(scope, name, row['mixed'], {k: round(row['metrics'][k], 5) for k in (
                'point_recall', 'fragmentation_score', 'continuous_score', 'weighted_total')}, row['budget'], row['checks'])


if __name__ == '__main__':
    main()
