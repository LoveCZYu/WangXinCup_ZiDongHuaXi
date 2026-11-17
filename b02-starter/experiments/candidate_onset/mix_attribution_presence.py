"""Separate temporary missing input observations from end-of-scene absence."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import nullcontext
from csv import DictReader
import json
from pathlib import Path
from zipfile import ZipFile

from b02.replay import open_scenes
from experiments.candidate_onset.mix_attribution_report import eligible
from experiments.candidate_onset.propagation import sha
from experiments.v2_association.run_stage1 import save


def read_presence(raw):
    observed = defaultdict(set)
    for row in DictReader(raw):
        truth = int(float(row['Batch_ID']))
        if truth > 0:
            observed[truth].add(int(float(row['Pt_Circle'])))
    return {truth: sorted(circles) for truth,circles in observed.items()}


def absence_kind(onset, visible):
    circle = onset['circle']
    appearances = visible[onset['expected']]
    after = bisect_right(appearances, circle)
    assert after == 0 or appearances[after-1] != circle
    return ('never_observed_again' if after == len(appearances) else 'returns_in_next_four'
            if appearances[after] - circle <= 4 else 'returns_after_four')


def summarize(visible, episodes):
    counts = Counter()
    for event in episodes:
        onset = event['onset']
        if onset['category'] != 'expected_point_absent' or not onset['seen_expected'] or onset['tied']:
            continue
        label = absence_kind(onset, visible)
        counts[label+'_episodes'] += 1
        counts[label+'_mixed_points'] += event['count']
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data','protocol','evidence','out'):
        p.add_argument(name, type=Path)
    p.add_argument('--cases', type=Path)
    args = p.parse_args()
    plan, digest = json.loads(args.protocol.read_bytes()), sha(args.protocol)
    assert sha(args.data) == plan['data_sha256']
    scenes = {}
    total = Counter()
    best_per_scene = []
    source = nullcontext(None) if args.evidence.is_dir() else ZipFile(args.evidence)
    with open_scenes(args.data) as (names, opener), source as archive:
        assert set(names) == set(plan['scenes'])
        for name in plan['scenes']:
            raw = archive.read(f'mix-attribution/{name}.json') if archive is not None else (
                args.evidence / (name+'.json')).read_bytes()
            row = json.loads(raw)
            assert (row['scene'], row['protocol_sha256']) == (name,digest)
            with opener(name) as source:
                visible = read_presence(source)
            counts = summarize(visible, row['episodes'])
            expected = row['rollups'][str(plan['main_gap'])]['weighted_from_first_error'].get('expected_point_absent',0)
            assert sum(k for label,k in counts.items() if label.endswith('_mixed_points')) == expected
            total.update(counts)
            scenes[name] = counts
            if args.cases is not None:
                target = next((event for event in eligible(row, plan, present=False)
                               if absence_kind(event['onset'], visible) == 'returns_in_next_four'), None)
                if target is not None:
                    best_per_scene.append(dict(scene=name, **target))
    save(args.out, dict(protocol_sha256=digest, totals=total, scenes=scenes))
    if args.cases is not None:
        selected = sorted(best_per_scene, key=lambda event: (-event['count'],event['scene']))[:plan['replay_gate']['max_scenes']]
        save(args.cases, dict(protocol_sha256=digest, enabled=bool(selected), cases=selected,
                              horizon=plan['replay_gate']['horizon'], selection='Longest per scene with a pure active prefix and a raw observation returning within four scans.'))
    print(dict(total))


if __name__ == '__main__':
    main()
