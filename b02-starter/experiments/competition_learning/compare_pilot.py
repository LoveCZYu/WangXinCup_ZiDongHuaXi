"""Keep failed offline training and research-only rollout conclusions separate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.scorecard import build_scorecard, write_scorecard
from experiments.competition_learning.pilot import MODES
from experiments.v2_association.run_stage1 import save, sha


def compare(root):
    pilot_path = Path(__file__).with_name('pilot-protocol.json')
    protocol = json.loads(pilot_path.read_text(encoding='utf-8'))
    names = protocol['head_validation8']
    paths = sorted(root.glob('pilot-*/train_*.json'))
    assert len(paths) == len(names) == 8
    rows = {}
    for path in paths:
        row = json.loads(path.read_text(encoding='utf-8'))
        assert row['research_only'] and row['pilot_sha256'] == sha(pilot_path)
        assert row['scene'] not in rows
        rows[row['scene']] = row
    assert set(rows) == set(names)
    baseline = {'scenes': {name: rows[name]['variants']['baseline'] for name in names}}
    variants = {}
    for mode in MODES:
        summary = {'kind': 'local_diagnostics', 'config': {'mode': mode, 'role': 'head_validation_research_only'},
            'scenes': {name: rows[name]['variants'][mode] for name in names}}
        folder = root/'pilot'/mode
        save(folder/'summary.json', summary)
        card = build_scorecard(summary, mode, baseline, 'baseline')
        write_scorecard(folder, card)
        counts, shadow = Counter(), Counter()
        for row in rows.values():
            counts.update(row['identity'][mode])
            shadow.update({k: v for k, v in row['shadow'][mode].items() if isinstance(v, int)})
        variants[mode] = {'metrics': card['overall']['values'], 'delta': card['overall']['delta'],
            'identity': dict(counts), 'short_ids_1or2': sum(row['birth_audit'][mode]['extra_ids_1or2'] for row in rows.values()),
            'strict_one_to_one_recall': 100*shadow['one_to_one_correct_points']/shadow['truth_points'],
            'scenes': {name: scope['delta'] for name, scope in card['scenes'].items()}}
        for shard in root.glob('pilot-*'):
            trace = shard/mode/f'{names[0]}.jsonl'
            if trace.is_file():shutil.copyfile(trace, folder/trace.name)
    c, raw = variants['joint_calibrated']['metrics'], variants['joint_raw']['metrics']
    delta = {k: c[k]-raw[k] for k in c if isinstance(c[k], (int, float)) and isinstance(raw[k], (int, float))}
    result = {'models': variants, 'calibrated_minus_raw': delta, 'pilot_protocol_sha256': sha(pilot_path),
        'research_only': True, 'original_offline_gate_overridden': False, 'version_promoted': False,
        'independent_validation': False, 'sealed_test_used': False,
        'scope': 'Full trajectories of new-head validation8, upstream ancestors used these scenes; not whole-system unseen validation'}
    save(root/'pilot-comparison.json', result)
    print(json.dumps({'research_only': True, 'models': {name: {'total': row['metrics']['weighted_total'],
        'fragmentation': row['metrics']['fragmentation_score'], 'continuous': row['metrics']['continuous_score'],
        'point_recall': row['metrics']['point_recall'], 'clutter': row['metrics']['clutter_outputs'],
        'mixed': row['identity']['mixed_minority_truth_points'], 'short_ids': row['short_ids_1or2'],
        'one_to_one': row['strict_one_to_one_recall']} for name, row in variants.items()}, 'calibrated_minus_raw': delta}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    compare(p.parse_args().root)
