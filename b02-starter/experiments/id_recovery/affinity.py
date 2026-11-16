"""Absolute same-identity evidence, trained on existing causal fit pairs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.v2_association.run_stage1 import save, sha

TRACK = (35, 36, 37, 38, 39, 93, 94, 97)
EDGE = (*range(20), *range(20, 50), *range(50, 85))
FEATURE_COUNT = len(TRACK) + len(EDGE)


def pack(track, edges):
    return np.concatenate((track[:, TRACK], edges[:, EDGE]), axis=1).astype(np.float32, copy=False)


class Affinity(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(FEATURE_COUNT))
        self.register_buffer('std', torch.ones(FEATURE_COUNT))
        self.net = nn.Sequential(nn.Linear(FEATURE_COUNT, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features):
        return self.net(((features-self.mean)/self.std).clamp(-8, 8)).squeeze(-1)

    @torch.inference_mode()
    def score(self, features):
        return self(torch.from_numpy(features)).numpy()


def load_affinity(path, prior):
    weights = torch.load(path, map_location='cpu', weights_only=True)
    assert weights['source_sha256'] == sha(__file__) and weights['prior_sha256'] == sha(prior)
    model = Affinity().eval()
    model.load_state_dict(weights['model'], strict=True)
    return model


def examples(root, scenes):
    inputs, targets = [], []
    for name in scenes:
        with np.load(root/'fit'/f'{name}.npz') as pairs:
            edges = pairs['memory_edges']
            track = pairs['memory_track']
            valid = pairs['valid'][:, :-1] & pairs['known'][:, None] & (edges[:, :, 11] >= .5) & (track[:, 36, None] >= .2)
            rows, columns = np.where(valid)
            inputs.append(pack(track[rows], edges[rows, columns]))
            targets.append(pairs['target'][rows, columns])
    return torch.from_numpy(np.concatenate(inputs)), torch.from_numpy(np.concatenate(targets).astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pairs', 'prior', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261116)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    manifest = json.loads((args.pairs/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['fit'] == plan['fit64']
    names = plan['fit64']
    assert not set(names) & set(plan['calibration'] + plan['confirmation16'] + plan['sealed_excluded'])
    validation = names[::4]
    fit = [scene for scene in names if scene not in validation]
    x, y = examples(args.pairs, fit)
    validation_x, validation_y = examples(args.pairs, validation)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Affinity()
    model.mean.copy_(x.mean(0))
    model.std.copy_(x.std(0).clamp_min(.015))
    model.to(device)
    x, y, validation_x, validation_y = (tensor.to(device) for tensor in (x, y, validation_x, validation_y))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    objective = nn.BCEWithLogitsLoss()
    args.out.mkdir(parents=True, exist_ok=False)
    history = []
    best = float('inf')
    for step in range(2401):
        if step in (0, 300, 800, 1600, 2400):
            with torch.no_grad():
                logits = torch.cat([model(batch) for batch in validation_x.split(4096)])
                loss = float(objective(logits, validation_y))
                precision_recall = {}
                for threshold in (.70, .80, .85, .90, .95):
                    selected = logits.sigmoid() >= threshold
                    hits = float(validation_y[selected].sum())
                    precision_recall[str(threshold)] = {'precision': hits / max(1, int(selected.sum())),
                                                        'recall': hits / max(1, int(validation_y.sum()))}
            results = {'step': step, 'loss': loss, 'precision_recall': precision_recall}
            print(json.dumps(results), flush=True)
            history.append(results)
            if loss < best:
                best = loss
                torch.save({'model': {key: value.cpu() for key, value in model.state_dict().items()},
                            'step': step, 'seed': args.seed, 'prior_sha256': sha(args.prior),
                            'source_sha256': sha(__file__)}, args.out/'selected.pt')
        if step == 2400:
            break
        batch = torch.randint(len(x), (1024,), device=device)
        loss = objective(model(x[batch]), y[batch])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    save(args.out/'report.json', {'fit_edges': len(x), 'validation_edges': len(validation_x),
                                 'fit_scenes': fit, 'validation_scenes': validation, 'history': history})


if __name__ == '__main__':
    main()
