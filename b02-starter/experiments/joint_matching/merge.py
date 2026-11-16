"""Aggregate calibration outputs and compute the predefined version checks."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.scorecard import build_scorecard, write_scorecard
from experiments.v2_association.run_stage1 import save


def checks(metrics, base):
    return {'point_recall': metrics['point_recall'] >= base['point_recall']-.2,
            'track_recall': metrics['track_recall'] >= base['track_recall']-.2,
            'continuous': metrics['continuous_score'] >= base['continuous_score']-.3,
            'timeliness': metrics['timeliness_score'] >= base['timeliness_score']-.5,
            'clutter': metrics['clutter_outputs'] <= 1.05*base['clutter_outputs'],
            'weighted_total': metrics['weighted_total'] >= base['weighted_total'],
            'fragmentation_major_gain': metrics['fragmentation_score'] >= base['fragmentation_score']+3.}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+')
    args = parser.parse_args()
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    reference = json.loads(args.baseline.read_text(encoding='utf-8'))
    names = args.scenes or plan['calibration']
    assert set(names) <= set(plan['calibration'])
    variants, fragments = defaultdict(dict), Counter()
    first, sizes, hist, plans, diagnostic = Counter(), Counter(), Counter(), Counter(), Counter()
    operations = defaultdict(Counter)
    seen = set()
    args.out.mkdir(parents=True, exist_ok=False)
    for part in args.parts:
        for path in sorted(part.glob('train_*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            scene = row['scene']
            assert scene not in seen and scene in names
            seen.add(scene)
            for key, value in row['variants'].items():
                variants[key][scene] = value
            diagnostic.update(row['identity'])
            plans.update(row.get('plans', {}))
            if 'attribution' in row:
                audit = row['attribution']
                first.update(audit['first_causes'])
                hist.update(audit['histogram'])
                sizes.update(audit['short_fragment_counts'])
                fragments['dominant_90pct_and_extra'] += audit['dominant_90pct_and_extra']
                fragments['new_id_previously_output_other_truth'] += sum(f['new_id_prior_other_truth'] for f in audit['fragments'])
                for key, value in row['repair'].items():
                    operations[key].update(value)
            trace = part/f'{scene}.jsonl'
            if trace.exists():
                (args.out/trace.name).write_bytes(trace.read_bytes())
    assert set(names) == seen
    result = {'scenes': names, 'variants': {}, 'identity': dict(diagnostic), 'plans': dict(plans), 'holdout_used': False}
    if first:
        result['attribution'] = {'first_causes': dict(first), 'id_count_histogram': dict(hist), 'short_fragment_counts': dict(sizes), **dict(fragments)}
        result['repair'] = {key: dict(value) for key, value in operations.items()}
    for method, scenes in variants.items():
        assert set(scenes) == seen
        summary = {'kind': 'local_diagnostics', 'config': {'method': method}, 'scenes': {scene: scenes[scene] for scene in names}}
        path = args.out if len(variants) == 1 else args.out/method
        path.mkdir(exist_ok=True)
        save(path/'summary.json', summary)
        card = build_scorecard(summary, method, reference, 'v2-memory')
        write_scorecard(path, card)
        overall = card['overall']
        result['variants'][method] = {'metrics': overall['values'], 'delta': overall['delta'],
                                      'checks': checks(overall['values'], overall['baseline'])}
    save(args.out/'report.json', result)
    print(json.dumps({method: {key: round(row['metrics'][key], 4) for key in ('weighted_total','fragmentation_score','continuous_score','point_recall')}
                      for method, row in result['variants'].items()}), flush=True)


if __name__ == '__main__':
    main()
