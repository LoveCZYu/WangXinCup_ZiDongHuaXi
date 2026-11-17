"""Compare the previous screen and a hard-control refit at stream prevalence."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from experiments.candidate_onset.propagation import sha
from experiments.candidate_onset.update_stream import CLASSES, risk


def load(directory, names, digest):
    result = {}
    for scene in names:
        meta = json.loads((directory / (scene + '.json')).read_bytes())
        assert meta['scene'] == scene and meta['protocol_sha256'] == digest
        with np.load(directory / (scene + '.npz')) as archive:
            rows = archive['updates']
        counts = meta['variants']['census']['operations']
        assert all((rows[:, 0] == i).sum() == counts.get('eligible_' + label, 0)
                   for i, label in enumerate(CLASSES))
        result[scene] = rows
    return result


def strata(rows, plan):
    b = plan['strata']
    age = np.digitize(rows[:, 8] * 100, np.asarray(b['age_scans']) - 1e-3)
    motion = np.digitize(rows[:, 6], b['motion_residual_squared'])
    gap = np.digitize(rows[:, 9] * 4, np.asarray(b['elapsed_scans']) + .001)
    return age * 9 + motion * 3 + gap


def budget_threshold(model, dataset, settings):
    controls = [risk(model, rows[rows[:, 0] == 1, 4:]) for rows in dataset.values()]
    controls = [values for values in controls if len(values) >= settings['minimum_scene_controls']]
    assert controls, 'Insufficient per-scene controls to set the false trigger budget'
    budget = settings['risk_budget_per_10000'] / 10000
    return max(float(np.sort(values)[-int(budget * len(values)) - 1]) for values in controls)


def fit(dataset, old, plan):
    settings = plan['fit']
    data = np.concatenate(list(dataset.values()))
    old_cut = np.quantile(risk(old, data[data[:, 0] == 1, 4:]), settings['hard_old_risk_quantile'])
    blocks, weights = [], []
    for rows in dataset.values():
        active = rows[rows[:, 0] != 0]
        normal = active[:, 0] == 1
        w = np.ones(len(active))
        buckets = strata(active[normal], plan)
        counts = np.bincount(buckets, minlength=27)
        w[normal] = np.minimum(4, np.sqrt(normal.sum() / (9 * counts[buckets])))
        hard = normal & ((active[:, 6] >= settings['hard_residual']) | (risk(old, active[:, 4:]) >= old_cut))
        w[hard] *= settings['hard_weight']
        positive_keys = [(int(row[2]), int(row[0])) for row in active[~normal]]
        groups = Counter(positive_keys)
        w[~normal] = [1 / groups[key] for key in positive_keys]
        blocks.append(active)
        weights.append(w)
    data = np.concatenate(blocks)
    w = np.concatenate(weights)
    x = data[:, 4:]
    y = data[:, 0] > 1
    assert y.any() and (~y).any()
    mean, scale = x.mean(0), np.maximum(x.std(0), .01)
    z = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    w[y] *= .5 / w[y].sum()
    w[~y] *= .5 / w[~y].sum()

    def objective(coef):
        prediction = z @ coef
        penalty = np.r_[coef[:-1], 0.]
        loss = np.dot(w, np.logaddexp(0., prediction) - y * prediction) + .05 * np.dot(penalty, penalty)
        grad = z.T @ (w * (expit(prediction) - y)) + .10 * penalty
        return loss, grad

    result = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B')
    if not result.success:
        raise ValueError(result.message)
    model = dict(mean=mean.tolist(), scale=scale.tolist(), coefficients=result.x.tolist())
    return dict(model, threshold=budget_threshold(model, dataset, settings),
                old_budget_threshold=budget_threshold(old, dataset, settings), old_hard_threshold=float(old_cut))


def count(rows, old, new, plan):
    x = rows[:, 4:]
    a, b = risk(old, x), risk(new, x)
    masks = dict(total=np.ones(len(rows), dtype=bool), old_flagged=a > old['threshold'],
                 old_budget_flagged=a > new['old_budget_threshold'],
                 refit_flagged=b > new['threshold'],
                 hard=(a >= new['old_hard_threshold']) | (x[:, 2] >= plan['fit']['hard_residual']))
    category = strata(rows, plan)
    report = {}
    for name, mask in masks.items():
        classified = rows[mask, 0]
        report[name] = dict(zip(CLASSES, np.bincount(classified.astype(int), minlength=4).tolist()))
        report[name + '_correct_strata'] = np.bincount(category[mask & (rows[:, 0] == 1)], minlength=27).tolist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--fit-only', action='store_true')
    args = parser.parse_args()
    digest = sha(args.protocol)
    plan = json.loads(args.protocol.read_bytes())
    old_path = args.protocol.with_name('screen-model.json')
    assert sha(old_path) == plan['sources']['experiments/candidate_onset/screen-model.json']
    old = json.loads(old_path.read_bytes())
    train = load(args.directory / 'stream-census-development', plan['development']['scenes'], digest)
    model = dict(fit(train, old, plan), protocol_sha256=digest, fit_scenes=list(train),
                 fit_source_sha256=sha(Path(__file__)))
    args.model.write_text(json.dumps(model, indent=2) + '\n', encoding='utf-8')
    report = dict(protocol_sha256=digest, policy_sha256=sha(args.model),
                  development_out_of_fold={name: count(rows, old, fit(
                      {n: x for n, x in train.items() if n != name}, old, plan), plan)
                      for name, rows in train.items()})
    if not args.fit_only:
        work = load(args.directory / 'stream-census-work', plan['work']['scenes'], digest)
        report['work'] = {name: count(rows, old, model, plan) for name, rows in work.items()}
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
