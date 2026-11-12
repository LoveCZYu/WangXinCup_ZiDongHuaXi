"""Produce reproducible seed means and full-scene score comparisons."""
import argparse
import json
from pathlib import Path
from statistics import mean

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
a = p.parse_args()
r = json.loads((a.run/'report.json').read_text())
assert r['complete'], 'Training comparison did not complete'
final = [e for e in r['evaluations'] if e['step'] == r['manifest']['steps_per_arm']]
assert len(final) == 4
metrics = ('recall', 'precision', 'point_f1')
result = {'seed_aggregation': 'Arithmetic mean across both final seeds; no best-seed selection.',
          'training_and_short_eval_seconds': r['elapsed_seconds'],
          'updates': sum(arm['steps'] for arm in r['arms']),
          'pure_training_seconds': sum(arm['training_seconds'] for arm in r['arms']),
          'short': {}, 'full': {}}
for variant in r['manifest']['variants']:
    chosen = [e for e in final if e['variant'] == variant]
    result['short'][variant] = {
        group: {mode: {key: mean(e['metrics'][group][mode][key] for e in chosen) for key in metrics}
                for mode in ('selected_points', 'filtered_tracker')}
        for group in r['manifest']['evaluation']}
for name in ('full-dev', 'full-dev-postfilter'):
    full = json.loads((a.run/name/'report.json').read_text())
    assert full['complete'], f'{name} did not complete'
    rows = []
    for row in full['runs']:
        values = json.loads((a.run/name/row['variant']/'scorecard.json').read_text(encoding='utf-8'))['overall']['values']
        rows.append({**row, 'local_scorecard': values})
    summary = {'elapsed_seconds': full['elapsed_seconds'], 'individual': rows, 'means': {}}
    for variant in ['baseline', *r['manifest']['variants']]:
        selected = [row for row in rows if row['variant'] == variant or row['variant'].endswith('-'+variant)]
        summary['means'][variant] = {
            **{key: mean(row[key] for row in selected)
               for key in (*metrics, 'clutter_outputs', 'split_truth_tracks', 'matched_truth_tracks')},
            'local_scorecard': {key: mean(row['local_scorecard'][key] for row in selected)
                               for key in selected[0]['local_scorecard']},
        }
    result['full'][name] = summary
result['total_computation_seconds'] = r['elapsed_seconds'] + sum(x['elapsed_seconds'] for x in result['full'].values())
(a.run/'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
for name, item in result['full'].items():
    print(name)
    for variant, values in item['means'].items():
        print(variant, {k: round(v, 6) for k, v in values.items() if k != 'local_scorecard'},
              'local_total', round(values['local_scorecard']['weighted_total'], 6))
print('Total computation seconds:', result['total_computation_seconds'])
