"""Aggregate fixed dev comparisons, retaining seed and scene results."""
import argparse
import json
from pathlib import Path
from statistics import mean

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
a = p.parse_args()
report = json.loads((a.run/'report.json').read_text())
assert report['complete'] and report['dev_complete']
rows = report['dev']; selection = report['selection']
policy = selection['selected']['name']
groups = {
    'baseline': ['baseline'],
    'small_fixed': [f'small_{seed}__fixed_0.5' for seed in (20261112, 20261113)],
    'expanded_fixed': [f'expanded_{seed}__fixed_0.5' for seed in (20261112, 20261113)],
    'expanded_selected': [f'expanded_{seed}__{policy}' for seed in (20261112, 20261113)],
}
def strict_mean(values):
    values = list(values)
    return None if any(v is None for v in values) else mean(values)

summary = {'elapsed_seconds': report['elapsed_seconds'], 'selection': selection,
           'seed_aggregation': 'Arithmetic mean of both prespecified seeds; no best-seed selection.',
           'overall': {}, 'by_scene': {}}
for name, keys in groups.items():
    selected = [rows[key] for key in keys]
    result = {key: mean(row[key] for row in selected) for key in selected[0] if key != 'local_scorecard'}
    result['local_scorecard'] = {key: mean(row['local_scorecard'][key] for row in selected)
                                for key in selected[0]['local_scorecard']}
    summary['overall'][name] = result
    cards = [json.loads((a.run/'dev'/key/'scorecard.json').read_text(encoding='utf-8')) for key in keys]
    summary['by_scene'][name] = {scene: {key: strict_mean(card['scenes'][scene]['values'][key] for card in cards)
                                       for key in cards[0]['scenes'][scene]['values']}
                                for scene in cards[0]['scenes']}
for name in ('small_fixed', 'expanded_fixed', 'expanded_selected'):
    row = summary['overall'][name]; baseline = summary['overall']['baseline']
    row['clutter_reduction_vs_baseline'] = 1-row['clutter_outputs']/baseline['clutter_outputs']
    row['recall_change_pp_vs_baseline'] = 100*(row['recall']-baseline['recall'])
    row['continuous_change_vs_baseline'] = row['local_scorecard']['continuous_score']-baseline['local_scorecard']['continuous_score']
    row['weighted_total_change_vs_baseline'] = row['local_scorecard']['weighted_total']-baseline['local_scorecard']['weighted_total']
    differences = []
    for scene, values in summary['by_scene'][name].items():
        reference = summary['by_scene']['baseline'][scene]
        differences.append({'scene': scene, 'recall_change_pp': values['point_recall']-reference['point_recall'],
                            'continuous_change': values['continuous_score']-reference['continuous_score'],
                            'score_change': values['weighted_total']-reference['weighted_total'],
                            'clutter_change': values['clutter_outputs']-reference['clutter_outputs']})
    row['scene_changes'] = differences
    row['scenes_with_higher_score'] = sum(x['score_change'] > 0 for x in differences)
    row['scenes_with_less_clutter'] = sum(x['clutter_change'] < 0 for x in differences)
save = a.run/'summary-comparison.json'
save.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
for name, row in summary['overall'].items():
    print(name, 'recall', round(row['recall']*100, 4), 'precision', round(row['precision']*100, 4),
          'clutter', row['clutter_outputs'], 'continuous', round(row['local_scorecard']['continuous_score'],4),
          'total', round(row['local_scorecard']['weighted_total'],4))
print('Seconds:', summary['elapsed_seconds'], 'selected:', policy)
