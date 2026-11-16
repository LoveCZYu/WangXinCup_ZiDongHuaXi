"""Read-only decomposition of one/two-point fragments and hindsight repairs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b02.scorecard import build_scorecard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    parts = sorted(args.root.glob('audit_only-*/train_*.json'))
    assert len(parts) == 8
    events, proposals = Counter(), Counter()
    variants = {}
    facts = Counter()
    for path in parts:
        item = json.loads(path.read_text(encoding='utf-8'))
        breakdown = item['fragment_audit']
        events.update(breakdown['short2_first_events'])
        proposals.update(item['plans'])
        for fragment in breakdown['fragments']:
            if fragment['points'] <= 2:
                facts['short2_total'] += 1
                facts['first_published'] += fragment['first_published_id']
                facts['previously_published'] += not fragment['first_published_id']
                facts['fresh_with_prior_same_truth'] += fragment['first_published_id'] and fragment['old'] is not None
                if not fragment['first_published_id']:
                    facts['previously_published_other_true_target' if fragment['new_id_prior_other_truth']
                          else 'previously_published_no_other_true_target'] += 1
            if fragment['first_published_id']:
                facts['fresh_all_lengths'] += 1
        for name, variant in item['diagnostic_oracles'].items():
            variants.setdefault(name, {'kind': 'local_diagnostics', 'scenes': {}})['scenes'][item['scene']] = variant['summary']
    reference = json.loads((args.root/'control'/'summary.json').read_text(encoding='utf-8'))
    oracles = {name: build_scorecard(summary, name, reference, 'combined')['overall'] for name, summary in variants.items()}
    result = {'scope': 'calibration8 only; retrospective truth-assisted diagnostics, not deployable models',
              'short2_first_events': dict(events), 'facts': dict(facts),
              'prebirth_proposals': {k:v for k,v in proposals.items() if k.startswith(('prebirth', 'proposal', 'eligible', 'first_output'))},
              'hindsight_repairs': oracles, 'sealed_test_used': False}
    (args.root/'birth-diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
