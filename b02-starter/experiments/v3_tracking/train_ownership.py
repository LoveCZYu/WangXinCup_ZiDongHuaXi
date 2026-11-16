"""Fit a residual scorer on sets of tracks competing for one observation."""
import argparse
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.v2_association.model import AssociationModel
from experiments.v2_association.run_stage1 import save, sha
from experiments.v3_tracking.ownership import EDGE_DIM, FEATURE_VERSION, TRACK_DIM


def loss_and_accuracy(model, track, edges, prior, valid, target):
    # A column has one observation and multiple track-feature vectors.
    shape = prior.shape
    raw = model(track.flatten(0, 1), edges.flatten(0, 1).unsqueeze(1))[:, 0].reshape(shape)
    minimum = raw.masked_fill(~valid, float('inf')).amin(dim=1, keepdim=True)
    residual = (raw - minimum).clamp(0, 4)
    logits = (-prior/.25 - residual).masked_fill(~valid, -1e9)
    true_logits = logits.masked_fill(~target, -1e9)
    loss = (logits.logsumexp(1) - true_logits.logsumexp(1)).mean()
    accuracy = target.gather(1, logits.argmax(1, keepdim=True)).float().mean()
    return loss + 1e-4 * raw[valid].square().mean(), accuracy


def load_columns(folder, scenes):
    parts = {key: [] for key in ('track', 'edges', 'prior', 'valid', 'target')}
    for name in scenes:
        with np.load(folder/f'{name}.npz') as archive:
            for key in parts:
                parts[key].append(archive[key])
    arrays = [np.concatenate(value) for value in parts.values()]
    assert len(arrays[0]) and np.all(arrays[-1].any(1)) and np.all(arrays[-1] <= arrays[-2])
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pairs', 'prior', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--shards', type=int, default=6)
    args = parser.parse_args()
    start = monotonic()
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    split = ROOT/'splits/v3_tracking.json'
    plan = json.loads(split.read_text(encoding='utf-8'))
    names = plan['fit64'] + plan['calibration']
    prior_hash = sha(args.prior)
    evidence = {}
    for i in range(args.shards):
        path = args.pairs/f'shard-{i}.json'
        shard = json.loads(path.read_text(encoding='utf-8'))
        assert shard['prior_sha256'] == prior_hash and shard['split_sha256'] == sha(split)
        assert set(shard['scenes']) == set(names[i::args.shards])
        evidence.update(shard['scenes'])
    fit = load_columns(args.pairs/'fit', plan['fit64'])
    calibration = load_columns(args.pairs/'calibration', plan['calibration'])
    model = AssociationModel(TRACK_DIM, EDGE_DIM).cuda()
    with torch.no_grad():
        for prefix, values in (('track', fit[0][fit[3]]), ('edge', fit[1][fit[3]])):
            getattr(model, prefix + '_mean').copy_(torch.from_numpy(values.mean(0)))
            getattr(model, prefix + '_std').copy_(torch.from_numpy(values.std(0).clip(.01)))
    fit_gpu = [torch.from_numpy(array).cuda() for array in fit]
    cal_gpu = [torch.from_numpy(array).cuda() for array in calibration]
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    prior = torch.load(args.prior, map_location='cpu', weights_only=True)
    metadata = {'format': 'b02-v3-ownership-training-v1', 'seed': args.seed, 'memory': True,
                'feature_version': FEATURE_VERSION, 'prior_sha256': prior_hash,
                'pairs': {'fit': plan['fit64'], 'calibration': plan['calibration']},
                'runtime_sha256': {name: sha(Path(__file__).with_name(name)) for name in ('ownership.py', 'tracker.py', 'memory.py')},
                'fit_columns': len(fit[0]), 'calibration_columns': len(calibration[0]), 'data_report': evidence}
    args.out.mkdir(parents=True, exist_ok=False)
    history = []
    checkpoints = {0, 250, 1000, 3000}
    for step in range(max(checkpoints) + 1):
        if step in checkpoints:
            with torch.no_grad():
                size = len(calibration[0])
                totals = np.zeros(2)
                for i in range(0, size, 2048):
                    batch = [array[i:i+2048] for array in cal_gpu]
                    totals += len(batch[0]) * np.array([float(v) for v in loss_and_accuracy(model, *batch)])
            row = {'step': step, 'calibration_loss': float(totals[0]/size), 'column_accuracy': float(totals[1]/size)}
            history.append(row)
            torch.save({'model': {k: v.cpu() for k, v in model.state_dict().items()}, 'prior': prior,
                        'step': step, 'metadata': metadata}, args.out/f'step-{step:05d}.pt')
            save(args.out/'history.json', history)
            print(json.dumps(row), flush=True)
        if step == max(checkpoints):
            break
        selection = torch.randint(len(fit[0]), (512,), device='cuda')
        loss, _ = loss_and_accuracy(model, *(array[selection] for array in fit_gpu))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    save(args.out/'selected.json', {'selected': min(history, key=lambda row: row['calibration_loss'])})
    save(args.out/'report.json', {'complete': True, 'step': step, 'wall_seconds': monotonic()-start,
                                 'peak_cuda_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    main()
