"""Learn whether a real conflict should transfer the selected point to an older ID."""
import argparse
from collections import Counter, defaultdict
import json
from math import log1p
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.replay import _scans, open_scenes
from experiments.id_recovery.affinity import FEATURE_COUNT, load_affinity
from experiments.id_recovery.tracker import CONFIGS, RecoveryTracker, cutoff
from experiments.v2_association.run_stage1 import save, sha
from experiments.v3_tracking.runtime import load_scorer

DIMENSION = 2*FEATURE_COUNT+8


class TransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(DIMENSION))
        self.register_buffer('std', torch.ones(DIMENSION))
        self.net = nn.Sequential(nn.Linear(DIMENSION, 48), nn.ReLU(), nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, values):
        return self.net(((values-self.mean)/self.std).clamp(-8, 8)).squeeze(-1)


class TransferTracker(RecoveryTracker):
    def __init__(self, scorer, affinity, model=None):
        super().__init__(scorer, affinity, CONFIGS['active'])
        self.transfer_model = model
        self.samples = None

    def reconnect_active(self, matched, events, scan, scores):
        self.samples = None
        proposals, logits, features = self.active_options(matched, events, scan, scores)
        if not proposals:
            return
        extra = []
        for (batch, _, _, winner), (old, rival) in zip(proposals, logits, strict=True):
            recent = scan-self.last_emission[batch]
            rival_recent = scan-self.last_emission.get(winner, scan-20)
            support = self.emissions[batch]
            rival_support = self.emissions[winner]
            extra.append([np.clip(old, -10, 10), np.clip(rival, -10, 10), log1p(support), log1p(rival_support),
                          recent/6, min(20, rival_recent)/6, support/self.state['tracks'][batch].hits,
                          rival_support/self.state['tracks'][winner].hits])
        values = np.concatenate((features, np.asarray(extra, dtype=np.float32)), axis=1)
        if self.transfer_model is None:
            self.samples = proposals, values
            return
        with torch.inference_mode():
            scores = self.transfer_model(torch.from_numpy(values)).numpy()
        confirmed = [(float(p), batch, index, winner) for (batch, index, _, winner), p, (old, _) in zip(proposals, scores, logits, strict=True)
                     if p >= cutoff(.65) and old >= cutoff(.65)]
        self.apply_transfers(matched, confirmed)


def target(truth, old, rival, histories):
    if truth <= 0:
        return 0.
    old_truth, rival_truth = histories.get(old, {}), histories.get(rival, {})
    support, competing = old_truth.get(truth, 0), rival_truth.get(truth, 0)
    if not support or max(old_truth, key=old_truth.get) != truth:
        return 0.
    if not competing or max(rival_truth, key=rival_truth.get) != truth:
        return 1.
    # Keep the baseline if both IDs are equally plausible owners.
    return float(support >= competing+2)


def collect(args, plan):
    assert 0 <= args.index < args.shards
    manifest = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['data_sha256'] == sha(args.data) and set(plan['fit32']) <= set(manifest['fit'])
    assert not set(plan['fit32']) & set(plan['calibration'] + plan['confirmation16'] + plan['sealed_excluded'])
    scorer, memory, _, _ = load_scorer(args.prior)
    assert memory
    affinity = load_affinity(args.affinity, args.prior)
    args.out.mkdir(parents=True, exist_ok=True)
    with open_scenes(args.data) as (_, opener):
        for scene in plan['fit32'][args.index::args.shards]:
            with opener(scene) as source:
                frames = list(_scans(source))
            with np.load(args.cache/'fit'/f'{scene}.npz') as cache:
                scores, offsets = cache['point_scores'], cache['point_offsets']
            assert len(offsets) == len(frames)+1
            tracker = TransferTracker(scorer, affinity)
            histories = defaultdict(Counter)
            data, labels = [], []
            for i, (_, points, truth) in enumerate(frames):
                predictions = tracker(points, scores[offsets[i]:offsets[i+1]].tolist())
                if tracker.samples:
                    proposals, features = tracker.samples
                    for (batch, p, _, winner), row in zip(proposals, features, strict=True):
                        label = truth[points[p]['PntID']]
                        if label >= 0:
                            data.append(row)
                            labels.append(target(label, batch, winner, histories))
                for point in predictions:
                    label = truth[point['PntID']]
                    if label > 0:
                        histories[point['BatchID']][label] += 1
                for batch in tracker.deleted:
                    histories.pop(batch, None)
            destination = args.out/f'{scene}.npz'
            assert not destination.exists()
            np.savez_compressed(destination, data=np.asarray(data, dtype=np.float32).reshape(-1, DIMENSION),
                                labels=np.array(labels, dtype=np.float32), affinity=sha(args.affinity))
            print(scene, len(labels), int(sum(labels)), flush=True)


def read_samples(folder, scenes, digest):
    data, labels = [], []
    for scene in scenes:
        with np.load(folder/f'{scene}.npz') as payload:
            assert payload['affinity'].item() == digest
            data.append(payload['data'])
            labels.append(payload['labels'])
    return torch.from_numpy(np.concatenate(data)), torch.from_numpy(np.concatenate(labels))


def train(args, plan):
    torch.manual_seed(20261116)
    validation = plan['fit32'][::4]
    fit = [name for name in plan['fit32'] if name not in validation]
    digest = sha(args.affinity)
    inputs, labels = read_samples(args.samples, fit, digest)
    val, expected = read_samples(args.samples, validation, digest)
    model = TransferModel()
    model.mean.copy_(inputs.mean(0))
    model.std.copy_(inputs.std(0).clamp_min(.03))
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best = float('inf')
    rows = []
    args.out.mkdir(parents=True, exist_ok=False)
    for step in range(1201):
        if step in (0, 100, 300, 600, 1200):
            with torch.no_grad():
                logits = model(val)
                loss = float(loss_fn(logits, expected))
                picked = logits.sigmoid() >= .65
                hits = float(expected[picked].sum())
                row = {'step': step, 'loss': loss, 'selected': int(picked.sum()),
                       'precision': hits/max(1, int(picked.sum())), 'recall': hits/max(1, int(expected.sum()))}
            rows.append(row)
            print(json.dumps(row), flush=True)
            if loss < best:
                best = loss
                torch.save({'model': model.state_dict(), 'step': step, 'affinity_sha256': digest,
                            'source_sha256': sha(__file__)}, args.out/'selected.pt')
        if step < 1200:
            index = torch.randint(len(inputs), (256,))
            loss = loss_fn(model(inputs[index]), labels[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    save(args.out/'report.json', {'fit_pairs': len(inputs), 'validation_pairs': len(val), 'history': rows,
                                 'fit_scenes': fit, 'validation_scenes': validation})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('collect','train'))
    for name in ('data', 'cache', 'prior', 'affinity', 'out', 'samples'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    (collect if args.command == 'collect' else train)(args, plan)


if __name__ == '__main__':
    main()
