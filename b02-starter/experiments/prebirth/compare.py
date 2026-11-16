"""Aggregate first-output audits beside the ordinary tracking scorecards."""
import argparse
from collections import Counter
import csv
from html import escape
import json
from pathlib import Path

NAMES = {'control': '组合方案对照', 'live': '起批前接回活跃旧 ID', 'both': '起批前接回活跃及退役 ID',
         'both_relaxed': '起批前接回（较宽）', 'short_live': '纳入短旧 ID（活跃）',
         'short_both': '纳入短旧 ID（活跃＋退役）', 'short_relaxed': '纳入短旧 ID（较宽）',
         'short_wide': '纳入短旧 ID（扩大运动容差）', 'wide_confident': '扩大容差＋更高置信门槛'}
METRICS = {'fragmentation_score': '零碎度', 'continuous_score': '连续跟踪', 'point_recall': '点命中率 %',
           'track_recall': '逐轨命中率 %', 'timeliness_score': '起批及时性',
           'clutter_outputs': '杂点输出', 'weighted_total': '本地估计总分'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    baseline = {}
    for path in sorted(root.glob('control-*/train_*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        baseline[row['scene']] = row['birth_audit']['first_truth_delays']
    assert len(baseline) == 8
    comparison, table = {}, []
    for mode, name in NAMES.items():
        if not (root/mode/'report.json').exists():
            continue
        report = json.loads((root/mode/'report.json').read_text(encoding='utf-8'))
        audits, delays = Counter(), Counter()
        scenes = {}
        for path in sorted(root.glob(mode+'-*/train_*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            audit = row['birth_audit']
            audits.update(audit['extra_id_lengths'])
            current = audit['first_truth_delays']
            pairs = [(current[t]-d) for t, d in baseline[row['scene']].items() if t in current]
            check = dict(later=sum(d>0 for d in pairs), later_over_one_scan=sum(d>1 for d in pairs),
                         earlier=sum(d<0 for d in pairs), lost=len(set(baseline[row['scene']])-set(current)),
                         max_extra_delay=max([0, *pairs]))
            for key in ('later', 'later_over_one_scan', 'earlier', 'lost'):
                delays[key] += check[key]
            delays['max_extra_delay'] = max(delays['max_extra_delay'], check['max_extra_delay'])
            scenes[row['scene']] = {'extra_id_lengths': audit['extra_id_lengths'], 'new_target_delays': check}
        assert len(scenes) == 8
        values = report['variants'][mode]['metrics']
        report['birth_audit'] = {'extra_id_lengths': dict(audits), 'extra_ids_1or2': audits['1']+audits['2'],
                                 'new_target_delays': dict(delays), 'scenes': scenes}
        report['target70_reached'] = values['fragmentation_score'] >= 70.
        comparison[mode] = report
        table.append((name, *(values[key] for key in METRICS), audits['1']+audits['2'], delays['later_over_one_scan'],
                      report['identity']['mixed_minority_truth_points']))
    (root/'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    headings = ('方案', *METRICS.values(), '额外 ID 仅命中 1–2 点', '新目标起批推迟超过一圈', '混轨少数身份点')
    with (root/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as target:
        writer = csv.writer(target); writer.writerow(headings); writer.writerows(table)
    rows = ''.join('<tr>'+''.join(f'<td>{escape(f"{v:.3f}" if isinstance(v, float) else str(v))}</td>' for v in row)+'</tr>' for row in table)
    links = ' · '.join(f'<a href="{escape(path.name, quote=True)}">{escape(path.stem)}</a>' for path in root.glob('train_*.html'))
    (root/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>B02 首次输出前 ID 转接</title>
<style>body{{font:15px/1.6 system-ui;background:#f6f8fa;color:#192530;padding:24px}}main{{max-width:1500px;margin:auto}}
table{{border-collapse:collapse;background:white}}td,th{{padding:10px;white-space:nowrap;border-bottom:1px solid #dce3e9;text-align:right}}
th{{background:#e7eff5}}td:first-child,th:first-child{{text-align:left}}</style><main><h1>首次输出前 ID 转接：校准对照</h1>
<p>固定八文件校准集，本地估分；封存测试集未使用。其余关联、点分类器、输出阈值保持组合方案设置。</p>
<p><a href="comparison.csv">分数表 CSV</a> · <a href="comparison.json">完整结果 JSON</a> · {links}</p>
<div style="overflow:auto"><table><thead><tr>{''.join(f'<th>{escape(s)}</th>' for s in headings)}</tr></thead><tbody>{rows}</tbody></table></div>
<p>短片段按每个真目标的非主 ID 命中点数统计；同一输出 ID 混入多个目标时，分别计数。目标 70 分及保护指标均在实验前固定。</p></main></html>''', encoding='utf-8')
    print(json.dumps({mode: {'fragmentation': row['variants'][mode]['metrics']['fragmentation_score'],
                            'extra_ids_1or2': row['birth_audit']['extra_ids_1or2'],
                            'new_target_delays': row['birth_audit']['new_target_delays']}
                      for mode, row in comparison.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
