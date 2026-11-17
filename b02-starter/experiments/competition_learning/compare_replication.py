"""Report both fixed binary-column seeds; never replace the default seed."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.scorecard import build_scorecard, write_scorecard
from experiments.v2_association.run_stage1 import save, sha


def compare(root):
    protocol_path = Path(__file__).with_name('replication.json')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    paths = sorted(root.glob('replica-*/train_*.json'))
    assert len(paths) == 8
    rows = {}
    for path in paths:
        row = json.loads(path.read_text(encoding='utf-8'))
        assert row['replication_sha256'] == sha(protocol_path) and row['seed'] == protocol['seed']
        rows[row['scene']] = row
    assert set(rows) == set(protocol['head_validation8'])
    baseline = json.loads((root/'transfer/baseline/summary.json').read_text(encoding='utf-8'))
    summary = {'kind': 'local_diagnostics', 'config': {'role': 'exploratory_second_seed'},
        'scenes': {name: rows[name]['summary'] for name in protocol['head_validation8']}}
    folder = root/'replication'
    save(folder/'summary.json', summary)
    card = build_scorecard(summary, 'binary_column_second_seed', baseline, 'baseline')
    write_scorecard(folder, card)
    counts, shadow = Counter(), Counter()
    for row in rows.values():
        counts.update(row['identity'])
        shadow.update({k:v for k,v in row['shadow'].items() if isinstance(v,int)})
    second = {'metrics': card['overall']['values'], 'delta': card['overall']['delta'],
        'identity': dict(counts), 'short_ids_1or2': sum(row['birth_audit']['extra_ids_1or2'] for row in rows.values()),
        'strict_one_to_one_recall': 100*shadow['one_to_one_correct_points']/shadow['truth_points'],
        'scenes': {name: scope['delta'] for name,scope in card['scenes'].items()}}
    previous = json.loads((root/'transfer-comparison.json').read_text(encoding='utf-8'))['models']
    first = previous['binary_column']
    b = previous['baseline']
    rule = protocol['primary_checks']
    checks = {}
    for seed, result in (('20261118', first), (str(protocol['seed']), second)):
        m, baseline_metrics = result['metrics'], b['metrics']
        checks[seed] = {'total': m['weighted_total'] >= baseline_metrics['weighted_total']+rule['both_seeds_total_gain_min'],
            'fragmentation': m['fragmentation_score'] >= baseline_metrics['fragmentation_score']+rule['both_seeds_frag_gain_min'],
            'point': m['point_recall'] >= baseline_metrics['point_recall']-rule['both_seeds_point_drop_pp_max'],
            'continuous': m['continuous_score'] >= baseline_metrics['continuous_score']-rule['both_seeds_continuous_drop_max'],
            'mixed': result['identity']['mixed_minority_truth_points'] <= b['identity']['mixed_minority_truth_points']*rule['both_seeds_mixed_ratio_max'],
            'short': result['short_ids_1or2'] <= b['short_ids_1or2']*rule['both_seeds_short_ids_ratio_max']}
    report = {'seeds': {'20261118': first, str(protocol['seed']): second}, 'baseline': b,
        'consistency_checks': checks, 'both_passed': all(all(c.values()) for c in checks.values()),
        'no_seed_selection': True, 'version_promoted': False, 'original_joint_failure_overridden': False,
        'data_role': 'exploratory replication on already exposed head validation8', 'protocol_sha256': sha(protocol_path)}
    save(root/'replication-comparison.json', report)
    print(json.dumps({'checks': checks, 'second': {'metrics': second['metrics'], 'mixed': second['identity']['mixed_minority_truth_points'],
        'short': second['short_ids_1or2'], 'shadow': second['strict_one_to_one_recall']}}, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    compare(p.parse_args().root)
