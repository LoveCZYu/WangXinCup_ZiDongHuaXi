"""Compare all frozen modes and apply the predeclared primary-only gate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.scorecard import build_scorecard, write_scorecard
from experiments.identity_competition.experiment import MODES
from experiments.v2_association.run_stage1 import save, sha


def compare(root, role):
    protocol_path = Path(__file__).with_name('column-protocol.json')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    paths = sorted(root.glob(f'{role}-*/train_*.json'))
    assert len(paths) == len(protocol[role]) == 8
    rows = {}
    for path in paths:
        row = json.loads(path.read_text(encoding='utf-8'))
        assert row['protocol_sha256'] == sha(protocol_path)
        assert row['scene'] not in rows and row['role'] == role
        rows[row['scene']] = row
    assert set(rows) == set(protocol[role])
    base = {'scenes': {name: rows[name]['variants']['baseline'] for name in protocol[role]}}
    report = {}
    for mode in MODES:
        summary = {'kind': 'local_diagnostics', 'config': {'column_mode': mode, 'data_role': role},
            'scenes': {name: rows[name]['variants'][mode] for name in protocol[role]}}
        folder = root/role/mode
        save(folder/'summary.json', summary)
        card = build_scorecard(summary, mode, base, 'baseline')
        write_scorecard(folder, card)
        identity, operations, shadows = Counter(), Counter(), Counter()
        for row in rows.values():
            identity.update(row['identity'][mode])
            operations.update(row['interventions'][mode])
            shadows.update({k: v for k, v in row['shadow'][mode].items() if isinstance(v, int)})
        report[mode] = {'metrics': card['overall']['values'], 'delta': card['overall']['delta'],
            'identity': dict(identity), 'interventions': dict(operations),
            'short_ids_1or2': sum(row['birth_audit'][mode]['extra_ids_1or2'] for row in rows.values()),
            'strict_one_to_one_recall': 100*shadows['one_to_one_correct_points']/shadows['truth_points'],
            'scenes': {name: scope['delta'] for name, scope in card['scenes'].items()}}
        for shard in root.glob(f'{role}-*'):
            trace = shard/mode/f'{protocol[role][0]}.jsonl'
            if trace.is_file():shutil.copyfile(trace, folder/trace.name)
    b, c = report['baseline']['metrics'], report['identity_volume']['metrics']
    rule = protocol['fitting_gate_to_calibration' if role == 'fitting' else 'calibration_success']
    checks = {'fragmentation_gain': c['fragmentation_score']-b['fragmentation_score'] >= rule['fragmentation_gain_min'],
        'weighted_gain': c['weighted_total']-b['weighted_total'] >= rule['weighted_gain_min'],
        'point_recall': c['point_recall'] >= b['point_recall']-rule['point_drop_pp_max'],
        'track_recall': c['track_recall'] >= b['track_recall']-rule['track_drop_pp_max'],
        'continuous': c['continuous_score'] >= b['continuous_score']-rule['continuous_drop_max'],
        'timeliness': c['timeliness_score'] >= b['timeliness_score']-rule['timeliness_drop_max'],
        'clutter': c['clutter_outputs'] <= b['clutter_outputs']*rule['clutter_ratio_max'],
        'mixed_identity': report['identity_volume']['identity']['mixed_minority_truth_points'] <=
            report['baseline']['identity']['mixed_minority_truth_points']*rule['mixed_points_ratio_max']}
    result = {'role': role, 'models': report, 'primary': protocol['primary'], 'checks': checks,
        'primary_passed': all(checks.values()), 'protocol_sha256': sha(protocol_path),
        'new_training': False, 'independent_validation': False, 'sealed_test_used': False}
    save(root/f'{role}-comparison.json', result)
    print(json.dumps({'role': role, 'primary_passed': result['primary_passed'], 'checks': checks,
        'models': {name: {'total': row['metrics']['weighted_total'], 'fragmentation': row['metrics']['fragmentation_score'],
            'continuous': row['metrics']['continuous_score'], 'point': row['metrics']['point_recall'],
            'mixed': row['identity']['mixed_minority_truth_points'], 'short': row['short_ids_1or2']} for name, row in report.items()}}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--role', choices=('fitting', 'calibration'), required=True)
    args = p.parse_args()
    compare(args.root, args.role)
