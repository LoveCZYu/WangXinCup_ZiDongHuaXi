"""Aggregate scene-level mixed-point accounting and pick bounded replay cases."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.candidate_onset.propagation import sha
from experiments.candidate_onset.propagation_work_summary import read


def combine(rows, key):
    total = Counter()
    for row in rows:
        total.update(row[key])
    return dict(total)


def summarized(rows, plan):
    rollups = {}
    for gap in plan['gaps']:
        values = [row['rollups'][str(gap)] for row in rows]
        rollups[str(gap)] = {key: combine(values, key) for key in ('starts','weighted_from_first_error','decomposition')}
        for key in ('continued_long','continued_long_reciprocal'):
            rollups[str(gap)][key] = sum(r[key] for r in values)
    events = [event for row in rows for event in row['episodes']]
    starts = [e['onset'] for e in events if e['onset']['seen_expected'] and not e['onset']['tied']]
    healthy = [row for row in starts if row['pure_prefix']]
    detail = Counter(row['reason'] for row in starts)
    direct = combine(rows, 'current_state_counts')
    count = sum(row['mixed'] for row in rows)
    assert sum(direct.values()) == count
    return dict(mixed=count, gaps=rollups, per_point_current_state=direct, established_first_details=dict(detail),
                pure_prefix_first=Counter(row['category'] for row in healthy),
                selected_despite_true_candidate=sum(row['category']=='available_but_not_selected' for row in starts),
                with_true_candidate_and_other_owner=sum(row['true_point_taken'] for row in starts),
                scenes={r['scene']: dict(mixed=r['mixed'], **r['rollups'][str(plan['main_gap'])]) for r in rows})


def eligible(row, plan, *, present=True):
    gate = plan['replay_gate']
    candidates = []
    for event in row['episodes']:
        onset = event['onset']
        if event['count'] < plan['long_min'] or not (onset['pure_prefix'] and onset['active_before'] and onset['internal_same']):
            continue
        if onset['tied'] or (onset['category'] != 'expected_point_absent') != present:
            continue
        if event['start']+gate['horizon'] >= len(row['frozen_outputs']):
            continue
        candidates.append(event)
    return sorted(candidates, key=lambda e: (-e['count'], e['start'], e['batch']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('protocol', type=Path)
    p.add_argument('report', type=Path)
    p.add_argument('cases', type=Path)
    args = p.parse_args()
    plan, digest = read(args.protocol), sha(args.protocol)
    rows = []
    for name in plan['scenes']:
        r = read(args.root/(name+'.json'))
        assert (r['scene'], r['protocol_sha256']) == (name, digest)
        rows.append(r)
    aggregate = summarized(rows, plan)
    other = summarized([row for row in rows if row['scene'] != 'train_034'], plan)
    all_eligible = {r['scene']: eligible(r,plan) for r in rows}
    eligible_scenes = [name for name, events in all_eligible.items() if events]
    gate = plan['replay_gate']
    continued = aggregate['gaps'][str(plan['main_gap'])]['decomposition'].get('continued',0)
    enabled = continued / aggregate['mixed'] >= gate['min_continued_share'] and len(eligible_scenes) >= gate['min_eligible_scenes']
    best = sorted(eligible_scenes, key=lambda name: (-all_eligible[name][0]['count'],name))[:gate['max_scenes']]
    selected = [dict(scene=name, **event) for name in best for event in all_eligible[name][:gate['max_cases_per_scene']]] if enabled else []
    cases = dict(protocol_sha256=digest, enabled=enabled, cases=selected,
                 horizon=gate['horizon'])
    args.cases.write_text(json.dumps(cases, indent=2)+'\n', encoding='utf8')
    report = dict(protocol_sha256=digest, all16=aggregate, other15=other,
                  replay=dict(eligible_scenes=eligible_scenes, eligible_episodes=sum(map(len,all_eligible.values())),
                              selected=selected, enabled=enabled, case_manifest_sha256=sha(args.cases)))
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(all16={k:v for k,v in aggregate.items() if k!='scenes'},
                          replay=dict(eligible_scenes=eligible_scenes, enabled=enabled, enrolled=[
                              (r['scene'],r['onset']['circle'],r['count']) for r in selected])), indent=2))


if __name__ == '__main__':
    main()
