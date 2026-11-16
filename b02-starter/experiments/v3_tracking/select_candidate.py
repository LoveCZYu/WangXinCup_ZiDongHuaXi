"""Choose one frozen candidate using only the eight calibration scenes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.v2_association.run_stage1 import save
from experiments.v3_tracking.freeze_candidate import feasibility


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'baseline', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    plan = read(ROOT/'splits/v3_tracking.json')
    baseline = read(args.baseline)
    regimes = {}
    for regime in ('memory', 'ownership'):
        runs = []
        for seed in plan['seeds']:
            run = args.runs/f'train-{regime}-{seed}'
            selection = read(run/'selected.json')['selected']
            report_path = args.runs/f'full-{regime}-{seed}'/'report.json'
            report = read(report_path)
            assert report['complete'] and report['scenes'] == plan['calibration']
            runs.append({'seed': seed, 'step': selection['step'], 'checkpoint': str(run/f"step-{selection['step']:05d}.pt"),
                         'report': str(report_path), 'metrics': report['metrics'], 'identity': report['identity'],
                         'feasibility': feasibility(report, baseline, plan['calibration_feasible'])})
        regimes[regime] = runs
    prior = regimes['memory'][0]
    extra = {}
    for row in regimes['ownership']:
        checks = feasibility(row, prior, plan['calibration_feasible'])
        # Require an actual gain on both identity sub-scores compared to the
        # parent common to both residual seeds. Training-row accuracy is irrelevant here.
        for key in ('fragmentation_score', 'continuous_score'):
            checks[key] = row['metrics'][key] > prior['metrics'][key]
        extra[str(row['seed'])] = checks
    ownership_passed = all(all(row['feasibility'].values()) and all(extra[str(row['seed'])].values())
                           for row in regimes['ownership'])
    selected = 'ownership' if ownership_passed else 'memory'
    save(args.out, {'selected': selected, 'regimes': regimes, 'ownership_vs_common_prior': extra,
                    'fit_scenes': len(plan['fit64']), 'calibration_scenes': plan['calibration']})
    print(selected, flush=True)


if __name__ == '__main__':
    main()
