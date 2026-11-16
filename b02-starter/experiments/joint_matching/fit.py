"""Collect and fit plan rankings on designated fitting scenes only."""
import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.replay import _scans, open_scenes
from experiments.joint_matching.model import FEATURES, MAX_OPTIONS, JointTracker, PlanModel
from experiments.v2_association.run_stage1 import historical_identity, save, sha
from experiments.v3_tracking.runtime import load_scorer


def training_examples(groups, points, labels, owners, history, future):
    for rows, options, features, costs in groups:
        truths = {point: labels[points[point]['PntID']] for _, point in options[0][0]}
        # Diagnose identity on real, distinct targets. Unlabelled and clutter
        # observations do not provide a label for this objective.
        if any(value <= 0 for value in truths.values()) or len(set(truths.values())) != len(truths):
            continue
        if not any(truth in owners for truth in truths.values()):
            continue
        anchors = {row: historical_identity(history.get(row, ())) for row in rows}
        utility = []
        for proposal, _ in options:
            value = 0.
            for row, point in proposal:
                truth = truths[point]
                previous = owners.get(truth, {})
                if previous and row == max(previous, key=previous.get):
                    value += 3. + .2 * future[truth]
                elif row in previous:
                    value += 1.
                if anchors[row] == truth:
                    value += 1.
                elif anchors[row] is not None and anchors[row] > 0:
                    value -= 2.
            utility.append(value)
        best = max(utility)
        target = np.zeros(MAX_OPTIONS, dtype=bool)
        target[:len(options)] = np.isclose(utility, best)
        if target[:len(options)].all():
            continue
        packed = np.zeros((MAX_OPTIONS, FEATURES), dtype=np.float32)
        distances = np.zeros(MAX_OPTIONS, dtype=np.float32)
        packed[:len(options)] = features
        distances[:len(options)] = costs
        yield packed, distances, np.arange(MAX_OPTIONS) < len(options), target


def collect(args, plan):
    manifest = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['data_sha256'] == sha(args.data)
    assert set(plan['fit32']) <= set(manifest['fit'])
    assert 0 <= args.index < args.shards
    names = plan['fit32'][args.index::args.shards]
    assert not set(names) & set(plan['calibration'] + plan['confirmation16'] + plan['sealed_excluded'])
    scorer, memory, _, _ = load_scorer(args.prior)
    assert memory
    args.out.mkdir(parents=True, exist_ok=True)
    prior_hash = sha(args.prior)
    with open_scenes(args.data) as (_, opener):
        for scene in names:
            with opener(scene) as source:
                frames = list(_scans(source))
            with np.load(args.cache/'fit'/f'{scene}.npz') as cache:
                scores, offsets = cache['point_scores'], cache['point_offsets']
            assert len(offsets) == len(frames)+1
            tracker = JointTracker(scorer)
            owners = defaultdict(Counter)
            history = defaultdict(lambda: deque(maxlen=8))
            records = []
            for i, (_, points, labels) in enumerate(frames):
                tracker.collect = i % 3 == 0
                predictions = tracker(points, scores[offsets[i]:offsets[i+1]].tolist())
                if tracker.groups:
                    future = Counter(label for _, _, later in frames[i+1:i+4] for label in set(later.values()) if label > 0)
                    records.extend(training_examples(tracker.groups, points, labels, owners, history, future))
                for row, point in (tracker.last_matches | tracker.last_births).items():
                    history[row].append(labels[points[point]['PntID']])
                for p in predictions:
                    truth = labels[p['PntID']]
                    if truth > 0:
                        owners[truth][p['BatchID']] += 1
                for row in tracker.deleted:
                    history.pop(row, None)
            destination = args.out/f'{scene}.npz'
            assert not destination.exists() and records
            np.savez_compressed(destination, plans=np.stack([r[0] for r in records]), costs=np.stack([r[1] for r in records]),
                                valid=np.stack([r[2] for r in records]), target=np.stack([r[3] for r in records]), prior_sha=prior_hash)
            print(scene, len(records), sum(not r[-1][0] for r in records), flush=True)


def tensors(args, names):
    fields = ('plans', 'costs', 'valid', 'target')
    parts = defaultdict(list)
    digest = sha(args.prior)
    for scene in names:
        with np.load(args.samples/f'{scene}.npz') as data:
            assert data['prior_sha'].item() == digest
            for field in fields:
                parts[field].append(data[field])
    return [torch.from_numpy(np.concatenate(parts[key])) for key in fields]


def objective(model, plans, costs, valid, target):
    prediction = model(plans, costs).masked_fill(~valid, -1e6)
    loss = (prediction.logsumexp(1)-prediction.masked_fill(~target, -1e6).logsumexp(1)).mean()
    return loss, target.gather(1, prediction.argmax(1, keepdim=True)).float().mean()


def train(args, plan):
    torch.manual_seed(args.seed)
    names = plan['fit32']
    validation = names[::4]
    training = [name for name in names if name not in validation]
    fit = tensors(args, training)
    val = tensors(args, validation)
    model = PlanModel()
    normal = fit[0][fit[2]]
    model.mean.copy_(normal.mean(0))
    model.std.copy_(normal.std(0).clamp_min(.03))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.001)
    args.out.mkdir(parents=True, exist_ok=False)
    history, best = [], float('inf')
    for step in range(1201):
        if step in (0, 100, 300, 600, 1200):
            with torch.no_grad():
                loss, accuracy = objective(model, *val)
            record = {'step': step, 'loss': float(loss), 'accuracy': float(accuracy)}
            history.append(record)
            print(json.dumps(record), flush=True)
            if float(loss) < best:
                best = float(loss)
                torch.save({'model': model.state_dict(), 'step': step, 'seed': args.seed, 'prior_sha256': sha(args.prior),
                            'model_sha256': sha(Path(__file__).with_name('model.py'))}, args.out/'selected.pt')
        if step < 1200:
            idx = torch.randint(len(fit[0]), (256,))
            loss, _ = objective(model, *(value[idx] for value in fit))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    save(args.out/'report.json', {'history': history, 'fit_groups': len(fit[0]), 'fit_scenes': training,
                                 'validation_groups': len(val[0]), 'validation_scenes': validation,
                                 'baseline_accuracy': float(val[-1][:, 0].float().mean())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'train'])
    for name in ('data', 'cache', 'prior', 'out', 'samples'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261116)
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    (collect if args.command == 'collect' else train)(args, plan)


if __name__ == '__main__':
    main()
