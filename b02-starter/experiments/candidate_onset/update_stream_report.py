"""Summarize whole-stream classification and paired tracking outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path

from b02.scorecard import build_scorecard
from experiments.candidate_onset.propagation import sha
from experiments.candidate_onset.propagation_work_summary import load, read

METRICS = ('weighted_total', 'point_recall', 'fragmentation_score', 'continuous_score')
TIMINGS = ('p95_handler_ms', 'max_handler_ms')


def totals(by_scene):
    tables = {}
    for report in by_scene.values():
        for key, row in report.items():
            if isinstance(row, dict):
                tables.setdefault(key, Counter()).update(row)
            else:
                totals_for_key = tables.setdefault(key, [0] * len(row))
                for index, value in enumerate(row):
                    totals_for_key[index] += value
    return tables


def paired(names, baseline, treatment, guards):
    reference = dict(scenes={n: baseline[n]['summary'] for n in names})
    card = build_scorecard(dict(scenes={n: treatment[n]['summary'] for n in names}), 'treatment', reference, 'census')
    counts = Counter()
    for row in treatment.values():
        counts.update(row['operations'])
    scene_rows = {}
    for name in names:
        row = treatment[name]
        scene_rows[name] = dict(operations=row['operations'],
                                delta={k: card['scenes'][name]['delta'][k] for k in METRICS},
                                mixed=row['identity']['mixed_minority_truth_points'],
                                clutter=row['summary']['clutter_outputs'])
    mixed = sum(row['mixed'] for row in scene_rows.values())
    reference_mixed = sum(row['identity']['mixed_minority_truth_points'] for row in baseline.values())
    delta = card['overall']['delta']
    checks = dict(mixed=mixed <= reference_mixed * (1 - guards['mixed_reduction_fraction']),
                  point_recall=delta['point_recall'] >= -guards['point_recall_drop_pp'],
                  fragmentation=delta['fragmentation_score'] >= -guards['fragmentation_drop'],
                  continuity=delta['continuous_score'] >= -guards['continuous_drop'],
                  weighted=delta['weighted_total'] >= -guards['weighted_drop'])
    return dict(metrics=card['overall']['values'], delta=delta, operations=dict(counts), mixed=mixed,
                clutter=sum(r['clutter'] for r in scene_rows.values()), checks=checks, scenes=scene_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    plan = read(args.protocol)
    digest = sha(args.protocol)
    previous = args.protocol.with_name('work-protocol.json')
    frozen = load(args.root, 'work16-*/train_*.json', sha(previous))
    census = load(args.root, 'stream-census-work/train_*.json', digest)
    assert set(census) == set(plan['work']['scenes']) == set(frozen)
    for scene, actual in census.items():
        measured = actual['variants']['census']['summary']
        reference = frozen[scene]['variants']['combined']
        assert {k: v for k, v in measured.items() if k not in TIMINGS} == {
            k: v for k, v in reference.items() if k not in TIMINGS}, scene
        assert actual['variants']['census']['identity'] == frozen[scene]['identity']['combined'], scene
    screening = read(args.root / 'update-stream-screen-report.json')
    assert screening['protocol_sha256'] == digest and set(screening['work']) == set(census)
    replay = load(args.root, 'stream-replay-work/train_*.json', digest)
    names = list(plan['focal']) if set(replay) == set(plan['focal']) else plan['work']['scenes']
    assert set(names) == set(replay) and all(r['policy_sha256'] == screening['policy_sha256'] for r in replay.values())
    base = {n: census[n]['variants']['census'] for n in names}
    out = dict(protocol_sha256=digest, policy_sha256=screening['policy_sha256'],
               work16_diagnostic_frontier=screening['work16_diagnostic_frontier'],
               screen={key: totals(group) for key, group in (
                   ('development', screening['development_out_of_fold']), ('work16', screening['work']),
                   ('work4', {n: screening['work'][n] for n in plan['focal']}),
                   ('remaining_work12', {n: screening['work'][n] for n in set(census) - set(plan['focal'])}))},
               replay={mode: paired(names, base, {
                   n: replay[n]['variants'][mode] if mode != 'census' else base[n] for n in names
               }, plan['guards']) for mode in ('census', 'motion', 'full')})
    args.out.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    keys = ('total', 'old_flagged', 'old_budget_flagged', 'refit_flagged', 'hard')
    print(json.dumps(dict(screen={name: {k: row[k] for k in keys} for name, row in out['screen'].items()},
                          replay={mode: {k: row[k] for k in ('metrics', 'operations', 'mixed', 'clutter', 'checks')}
                                  for mode, row in out['replay'].items()}), indent=2))


if __name__ == '__main__':
    main()
