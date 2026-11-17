"""Equal-batch binary/joint training, frozen scene split and checkpoint rule."""
import argparse
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.competition_learning.metrics import evaluate
from experiments.competition_learning.model import Model, objective
from experiments.competition_learning.samples import load, Tensors
from experiments.id_recovery.affinity import load_affinity
from experiments.v2_association.run_stage1 import save, sha


@torch.inference_mode()
def predict(model, data, device):
    pair = np.concatenate([model.edges(torch.as_tensor(part, device=device)).cpu().numpy()
                          for part in np.array_split(data['pair'], max(1, (len(data['pair'])+8191)//8192))])
    null = np.concatenate([model.rejection(torch.as_tensor(part, device=device)).cpu().numpy()
                          for part in np.array_split(data['row'], max(1, (len(data['row'])+4095)//4096))])
    return pair, null


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('samples', 'protocol', 'prior', 'affinity', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    for name, digest in protocol['training_sources'].items():assert sha(ROOT/name) == digest, name
    assert sha(args.prior) == protocol['artifacts']['prior'] and sha(args.affinity) == protocol['artifacts']['affinity']
    fit, fit_meta = load(args.samples, protocol['training24'], args.protocol)
    val, val_meta = load(args.samples, protocol['validation8'], args.protocol)
    assert set(protocol['training24']).isdisjoint(protocol['validation8'])
    old = load_affinity(args.affinity, args.prior)
    reference = np.concatenate([old.score(part[:, :93]) for part in
        np.array_split(val['pair'], max(1, (len(val['pair'])+4095)//4096))])
    old_metrics = evaluate(val, reference, np.zeros(len(val['row']), dtype=np.float32), reference)
    args.out.mkdir(parents=True, exist_ok=False)
    device = 'cuda'
    assert torch.cuda.is_available()
    arrays = Tensors(fit, device)
    run_reports = {}
    for seed in protocol['seeds']:
        for kind in ('binary', 'joint'):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = Model()
            model.initialize(old, torch.from_numpy(fit['pair'][fit['known_ids']]), torch.from_numpy(fit['row']))
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
            folder = args.out/f'{kind}-{seed}'
            folder.mkdir()
            started = monotonic()
            history, best, selected = [], float('inf'), None
            for step in range(protocol['steps']+1):
                if monotonic()-started > protocol['budget']['seconds_per_model']:
                    raise TimeoutError('Fixed per-model training/validation time budget exceeded')
                if step in protocol['checkpoints']:
                    model.eval()
                    pair, null = predict(model, val, device)
                    metrics = evaluate(val, pair, null, reference)
                    criterion = metrics['joint_loss' if kind == 'joint' else 'binary_loss']
                    record = {'step': step, 'seconds': monotonic()-started, **metrics}
                    history.append(record)
                    if criterion < best:
                        best, selected = criterion, record
                        torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            'step': step, 'seed': seed, 'kind': kind, 'protocol_sha256': sha(args.protocol),
                            'model_source_sha256': sha(ROOT/'experiments/competition_learning/model.py'),
                            'feature_source_sha256': sha(ROOT/'experiments/competition_learning/features.py')}, folder/'selected.pt')
                    save(folder/'history.json', history)
                    print(json.dumps({'kind': kind, 'seed': seed, 'step': step, 'criterion': criterion,
                        'wrong_columns': metrics['fixed_coverage_column']['wrong'],
                        'positive_recall': metrics['positive_row_match_recall'],
                        'false_accept': metrics['absent_row_false_accept_rate']}), flush=True)
                if step == protocol['steps']:break
                model.train()
                loss = objective(model, arrays.batch(), kind == 'joint')
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            report = {'kind': kind, 'seed': seed, 'selected': selected, 'history': history,
                'selected_weights_sha256': sha(folder/'selected.pt'), 'protocol_sha256': sha(args.protocol),
                'elapsed_seconds': monotonic()-started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated()}
            save(folder/'report.json', report)
            run_reports[f'{kind}-{seed}'] = report
            del model, optimizer
            torch.cuda.empty_cache()
    comparisons = {}
    rule = protocol['offline_gate']
    sample_counts = [row['counts'] for row in fit_meta+val_meta]
    oversized = sum(row.get('oversize_columns', 0) for row in sample_counts)
    all_columns = sum(row['columns'] for row in sample_counts)
    coverage_guard = oversized/max(1, all_columns) <= rule['oversize_column_fraction_max']
    for seed in protocol['seeds']:
        b = run_reports[f'binary-{seed}']['selected']
        c = run_reports[f'joint-{seed}']['selected']
        checks = {'column_wrong_reduction': c['fixed_coverage_column']['wrong'] <=
            b['fixed_coverage_column']['wrong']*rule['wrong_column_ratio_max'],
            'row_recall': c['positive_row_match_recall'] >= b['positive_row_match_recall']-rule['positive_row_recall_drop_max'],
            'false_accept': c['absent_row_false_accept_rate'] <= b['absent_row_false_accept_rate']*rule['false_accept_ratio_max']}
        assert c['fixed_coverage_column']['selected'] == b['fixed_coverage_column']['selected']
        comparisons[str(seed)] = {'checks': checks, 'passed': all(checks.values())}
    result = {'complete': True, 'old_affinity': old_metrics, 'models': run_reports, 'comparisons': comparisons,
        'offline_passed': coverage_guard and all(row['passed'] for row in comparisons.values()), 'protocol_sha256': sha(args.protocol),
        'oversize_column_fraction': oversized/max(1, all_columns), 'coverage_guard': coverage_guard,
        'fit_samples': {'rows': len(fit['row']), 'edges': len(fit['pair'])},
        'validation_samples': {'rows': len(val['row']), 'edges': len(val['pair'])},
        'data_role': 'head-only scene validation inside parent fit64; not whole-system unseen validation'}
    save(args.out/'report.json', result)
    print(json.dumps({'offline_passed': result['offline_passed'], 'comparisons': comparisons}), flush=True)


if __name__ == '__main__':main()
