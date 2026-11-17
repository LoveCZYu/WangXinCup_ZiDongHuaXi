"""Summarize the final protected-column study without choosing a winner."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.scorecard import build_scorecard, write_scorecard
from experiments.competition_learning.column_transfer import MODES
from experiments.v2_association.run_stage1 import save, sha


def compare(root):
    protocol_path = Path(__file__).with_name('column-transfer.json')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    names = protocol['head_validation8']
    rows = {}
    paths = sorted(root.glob('transfer-*/train_*.json'))
    assert len(paths) == len(names) == 8
    for p in paths:
        row = json.loads(p.read_text(encoding='utf-8'))
        assert row['transfer_sha256'] == sha(protocol_path) and row['research_only']
        assert row['scene'] not in rows
        rows[row['scene']] = row
    assert set(rows) == set(names)
    baseline = {'scenes': {name: rows[name]['variants']['baseline'] for name in names}}
    result = {}
    for mode in MODES:
        folder = root/'transfer'/mode
        summary = {'kind': 'local_diagnostics', 'config': {'mode': mode, 'role': 'head_validation_research_only'},
                   'scenes': {name: rows[name]['variants'][mode] for name in names}}
        save(folder/'summary.json', summary)
        card = build_scorecard(summary, mode, baseline, 'baseline')
        write_scorecard(folder, card)
        counts, shadow = Counter(), Counter()
        for row in rows.values():
            counts.update(row['identity'][mode])
            shadow.update({k: v for k, v in row['shadow'][mode].items() if isinstance(v, int)})
        result[mode] = {'metrics': card['overall']['values'], 'delta': card['overall']['delta'], 'identity': dict(counts),
            'short_ids_1or2': sum(row['birth_audit'][mode]['extra_ids_1or2'] for row in rows.values()),
            'strict_one_to_one_recall': 100*shadow['one_to_one_correct_points']/shadow['truth_points'],
            'scenes': {name: scope['delta'] for name, scope in card['scenes'].items()}}
        for shard in root.glob('transfer-*'):
            trace = shard/mode/f'{names[0]}.jsonl'
            if trace.is_file():shutil.copyfile(trace, folder/trace.name)
    b, c = result['baseline']['metrics'], result['joint_column']['metrics']
    rule = protocol['evidence_checks']
    checks = {'fragmentation': c['fragmentation_score'] >= b['fragmentation_score']+rule['fragmentation_gain_min'],
        'weighted': c['weighted_total'] >= b['weighted_total']+rule['weighted_gain_min'],
        'point': c['point_recall'] >= b['point_recall']-rule['point_drop_pp_max'],
        'track': c['track_recall'] >= b['track_recall']-rule['track_drop_pp_max'],
        'continuous': c['continuous_score'] >= b['continuous_score']-rule['continuous_drop_max'],
        'timeliness': c['timeliness_score'] >= b['timeliness_score']-rule['timeliness_drop_max'],
        'clutter': c['clutter_outputs'] <= b['clutter_outputs']*rule['clutter_ratio_max'],
        'mixed': result['joint_column']['identity']['mixed_minority_truth_points'] <=
                 result['baseline']['identity']['mixed_minority_truth_points']*rule['mixed_ratio_max'],
        'shadow': result['joint_column']['strict_one_to_one_recall'] >=
                  result['baseline']['strict_one_to_one_recall']-rule['one_to_one_drop_pp_max']}
    report = {'models': result, 'primary_checks': checks, 'all_primary_checks_passed': all(checks.values()),
        'transfer_protocol_sha256': sha(protocol_path), 'research_only': True, 'version_promoted': False,
        'original_offline_gate_overridden': False, 'scope': 'same head validation8, already exposed development data'}
    save(root/'transfer-comparison.json', report)
    if (root/'pilot-comparison.json').exists():
        prior = json.loads((root/'pilot-comparison.json').read_text(encoding='utf-8'))['models']['baseline']['metrics']
        for key, value in b.items():
            assert (value is None and prior[key] is None) or abs(value-prior[key]) < 1e-9, key
    print(json.dumps({'primary_checks': checks, 'models': {mode: {'total': row['metrics']['weighted_total'],
        'fragmentation': row['metrics']['fragmentation_score'], 'continuous': row['metrics']['continuous_score'],
        'point': row['metrics']['point_recall'], 'mixed': row['identity']['mixed_minority_truth_points'],
        'short': row['short_ids_1or2']} for mode, row in result.items()}}, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    compare(p.parse_args().root)
