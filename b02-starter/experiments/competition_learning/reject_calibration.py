"""Fit one null offset on training scenes, then diagnose held-out head rows."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.competition_learning.model import Model
from experiments.competition_learning.samples import load
from experiments.competition_learning.train import predict
from experiments.v2_association.run_stage1 import save, sha


def row_state(data, pair, null):
    ids, mask = data['row_index'], data['row_mask']
    scores = np.where(mask, pair[ids], -1e4)
    winner = scores.argmax(1)
    top = scores[np.arange(len(scores)), winner]
    correct = data['edge_positive'][ids][np.arange(len(scores)), winner]
    positives = (data['edge_positive'][ids] & mask).any(1)
    unknown_counts = np.bincount(data['edge_row'][~data['edge_known']], minlength=len(ids))
    # Evaluate only rows whose full geometric candidate list has known labels.
    allowed = data['row_allowed'] & (unknown_counts == 0)
    return {'margin': top-null, 'correct': correct, 'positive': allowed & positives,
            'absent': allowed & ~positives, 'excluded_unknown_rows': int((unknown_counts > 0).sum())}


def rates(state, offset):
    accepted = state['margin'] > offset
    positive = state['positive']
    absent = state['absent']
    return {'positive_match_recall': float((accepted & state['correct'])[positive].mean()),
        'absent_false_accept_rate': float(accepted[absent].mean()), 'positive_rows': int(positive.sum()),
        'absent_rows': int(absent.sum()), 'wrong_target_on_present_rows': int((accepted & ~state['correct'] & positive).sum()),
        'excluded_unknown_rows': state['excluded_unknown_rows']}


def fit_offset(binary_state, joint_state):
    assert np.array_equal(binary_state['positive'], joint_state['positive'])
    target = int(((binary_state['margin'] > 0) & binary_state['correct'] & binary_state['positive']).sum())
    good = joint_state['margin'][joint_state['positive'] & joint_state['correct']]
    if len(good) < target or not target:return None
    value = np.sort(good)[::-1][target-1]
    # Include ties at the target rank, without consulting validation labels.
    return float(np.nextafter(value, np.asarray(-np.inf, dtype=good.dtype)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('samples', 'protocol', 'calibration', 'training', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    diagnostic = json.loads(args.calibration.read_text(encoding='utf-8'))
    assert diagnostic['learning_protocol_sha256'] == sha(args.protocol)
    assert diagnostic['source_sha256'] == sha(Path(__file__))
    fit, _ = load(args.samples, protocol['training24'], args.protocol)
    val, _ = load(args.samples, protocol['validation8'], args.protocol)
    reports = {}
    for seed in protocol['seeds']:
        states = {}
        for kind in ('binary', 'joint'):
            path = args.training/f'{kind}-{seed}'/'selected.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True)
            assert saved['protocol_sha256'] == sha(args.protocol)
            assert saved['model_source_sha256'] == sha(ROOT/'experiments/competition_learning/model.py')
            model = Model().cuda().eval()
            model.load_state_dict(saved['model'], strict=True)
            fit_pair, fit_null = predict(model, fit, 'cuda')
            val_pair, val_null = predict(model, val, 'cuda')
            states[kind] = (row_state(fit, fit_pair, fit_null), row_state(val, val_pair, val_null))
            del model
        offset = fit_offset(states['binary'][0], states['joint'][0])
        record = {'offset_fitted_on_training24_only': offset,
            'binary_fit': rates(states['binary'][0], 0.), 'joint_fit_raw': rates(states['joint'][0], 0.),
            'binary_validation': rates(states['binary'][1], 0.), 'joint_validation_raw': rates(states['joint'][1], 0.)}
        if offset is not None:
            record['joint_fit_calibrated'] = rates(states['joint'][0], offset)
            record['joint_validation_calibrated'] = rates(states['joint'][1], offset)
        reports[str(seed)] = record
    save(args.out, {'kind': 'post_training_calibration_diagnostic', 'seeds': reports,
        'protocol_sha256': sha(args.protocol), 'diagnostic_protocol_sha256': sha(args.calibration),
        'original_offline_gate_overridden': False, 'rollout_authorized_by_this_diagnostic': False,
        'independent_validation': False, 'new_training': False})
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':main()
